"""FastAPI-based webhook server to accept MT4/EA signals and feed the existing scheduler.

POST /signal accepts JSON like:
{
  "asset": "EURUSD-OTC",             # or "EURUSD_otc" (normalized to *_otc)
  "direction": "call",               # "call" | "put"
  "timeframe_min": 1,                 # int, optional (default 1)
  "trade_time": "13:50",             # HH:MM[:SS], optional; if missing, immediate
  "forecast_pct": 73.5,               # optional float
  "payout_pct": 82.0,                 # optional float (informational)
  "tz_offset_min": 330,               # minutes east of UTC (default IST +330)
  "raw": "free text from MT4"        # optional string for logging
}

It uses the same schedule_signal + Strategy19-like filtering and CSV logging as telegram_signal_live.
"""

from __future__ import annotations
import os
import time
import csv
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Reuse existing parser types and scheduler
try:
    from .signal_reader_telegram import TelegramSignal, schedule_signal, ScheduledTelegramSignal  # type: ignore
except Exception:
    from signal_reader_telegram import TelegramSignal, schedule_signal, ScheduledTelegramSignal  # type: ignore

# Reuse S19-like filtering + logging helpers inline (kept minimal to avoid import cycles)

LOG_CSV = os.path.join(os.path.dirname(__file__), "..", "strategy19_signals.csv")

def _ensure_log(path: str):
    try:
        new = not os.path.exists(path)
        if new:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", newline="", encoding="utf-8") as f:
            if new:
                csv.writer(f).writerow([
                    "ts_local","asset","direction","timeframe_min","trade_time",
                    "forecast_pct","payout_pct","trade_epoch","seconds_until_entry",
                    "ignored","reason","raw",
                ])
    except Exception:
        pass

def _log_signal(sig: ScheduledTelegramSignal):
    try:
        _ensure_log(LOG_CSV)
        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                sig.asset,
                sig.direction,
                sig.timeframe_min or "",
                sig.trade_time or "",
                sig.forecast_pct or "",
                sig.payout_pct or "",
                sig.trade_epoch or "",
                sig.seconds_until_entry(),
                int(sig.ignored),
                sig.reason,
                (sig.raw_block or "").replace("\n", " | ")[:200],
            ])
    except Exception:
        pass


class SignalIn(BaseModel):
    asset: str = Field(..., description="Symbol, e.g., EURUSD-OTC or EURUSD_otc")
    direction: str = Field(..., description="call | put")
    timeframe_min: Optional[int] = Field(1, description="Expiry minutes (default 1)")
    trade_time: Optional[str] = Field(None, description="HH:MM[:SS] local to tz_offset_min")
    forecast_pct: Optional[float] = None
    payout_pct: Optional[float] = None
    tz_offset_min: Optional[int] = Field(330, description="Minutes east of UTC (IST=+330)")
    raw: Optional[str] = Field(None, description="Free-form notes")

    def norm(self) -> "SignalIn":
        # Normalize asset to *_otc if -OTC suffix
        a = (self.asset or "").strip()
        if a.lower().endswith("-otc"):
            core = a[:-4].replace(" ", "").replace("-", "").upper()
            a = f"{core}_otc"
        elif a.endswith("_OTC"):
            a = a[:-4] + "_otc"
        self.asset = a
        # Normalize direction
        d = (self.direction or "").strip().lower()
        if d not in ("call", "put"):
            # allow arrows or words
            if "🔼" in d or "up" in d:
                d = "call"
            elif "🔽" in d or "down" in d:
                d = "put"
        self.direction = d
        return self


def _filter(sig: ScheduledTelegramSignal, min_forecast: float = 0.0, allow_grace_s: int = 8) -> ScheduledTelegramSignal:
    # Forecast filter
    if sig.forecast_pct is not None and sig.forecast_pct < float(min_forecast):
        sig.ignored = True
        sig.reason = f"forecast<{min_forecast}"
        return sig
    # Late arrival check (entry moment passed beyond grace)
    now = time.time()
    if sig.trade_epoch is not None and (sig.trade_epoch - sig.entry_lead_s) < now:
        if now - (sig.trade_epoch - sig.entry_lead_s) > allow_grace_s:
            sig.ignored = True
            sig.reason = "missed_entry"
            return sig
    sig.reason = "ok"
    return sig


app = FastAPI(title="Quotex MT4 Webhook", version="1.0")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "time": int(time.time())}


# --- Optional live execution wiring (Quotex client) ---
_TRADE_LOCK = None  # lazy-created asyncio.Lock

class _QuotexManager:
    def __init__(self) -> None:
        self.qx = None
        self.connected = False
        self.account_mode = os.environ.get("QX_ACCOUNT", "PRACTICE").upper()

    async def connect(self) -> None:
        if self.connected:
            return
        try:
            from pyquotex.stable_api import Quotex  # type: ignore
        except Exception as e:  # pragma: no cover
            print(f"[WEBHOOK][QX] import error: {e}")
            return
        email = os.environ.get("QX_EMAIL") or os.environ.get("QUOTEX_EMAIL")
        password = os.environ.get("QX_PASSWORD") or os.environ.get("QUOTEX_PASSWORD")
        # Fallback to local settings/config.ini if env missing
        if not (email and password):
            try:
                import configparser
                cfg = configparser.ConfigParser()
                cfg.read(os.path.join(os.path.dirname(__file__), "..", "settings", "config.ini"), encoding="utf-8")
                email = email or cfg.get("settings", "email", fallback=None)
                password = password or cfg.get("settings", "password", fallback=None)
            except Exception:
                pass
        if not (email and password):
            print("[WEBHOOK][QX] Missing credentials; set QX_EMAIL/QX_PASSWORD env vars")
            return
        try:
            self.qx = Quotex(email=email, password=password, lang="en")
            ok, reason = await self.qx.connect()
            if not ok:
                print(f"[WEBHOOK][QX] connect failed: {reason}")
                return
            # Switch account mode if REAL requested
            try:
                if self.account_mode.startswith("R"):
                    await self.qx.change_account('REAL')
                else:
                    await self.qx.change_account('PRACTICE')
            except Exception:
                pass
            self.connected = True
            print("[WEBHOOK][QX] Connected")
        except Exception as e:
            print(f"[WEBHOOK][QX] connect exception: {e}")

    async def ensure(self) -> None:
        if not self.connected:
            await self.connect()

    async def trend_ok(self, asset: str, direction: str) -> bool:
        """Simple trend guard: EMA50 on 60s, price aligned and EMA slope in direction."""
        try:
            if not self.qx:
                return False
            import time as _t
            candles = await self.qx.get_candles(asset, _t.time(), 60 * 240, 60)
            if not candles or len(candles) < 60:
                return False
            closes = [float(c.get("close", 0)) for c in candles]
            # lightweight EMA50
            ema = []
            period = 50
            k = 2 / (period + 1)
            if len(closes) < period:
                prev = closes[0]
                ema.append(prev)
                for v in closes[1:]:
                    prev = (v - prev) * k + prev
                    ema.append(prev)
            else:
                prev = sum(closes[:period]) / period
                ema.append(prev)
                for v in closes[period:]:
                    prev = (v - prev) * k + prev
                    ema.append(prev)
            if len(ema) < 2:
                return False
            ema_last = ema[-1]
            ema_prev = ema[-2]
            close_last = closes[-1]
            if direction == "call":
                return close_last > ema_last and ema_last >= ema_prev
            else:
                return close_last < ema_last and ema_last <= ema_prev
        except Exception:
            return False

    async def execute(self, asset: str, direction: str, expiry_min: int, amount: float) -> dict:
        await self.ensure()
        if not self.connected or not self.qx:
            return {"executed": False, "status": "no_connection"}
        duration_s = int(max(1, expiry_min) * 60)
        # Prefer robust helper from main if available
        try:
            try:
                from .main import place_and_wait  # type: ignore
            except Exception:
                from app.main import place_and_wait  # type: ignore
            won, delta = await place_and_wait(self.qx, amount, asset, direction, duration_s)
            status = getattr(self.qx, "last_exec_status", "win" if won else "loss")
            return {"executed": True, "status": status, "won": bool(won), "delta": float(delta)}
        except Exception:
            # Minimal fallback
            try:
                ok, _ = await self.qx.buy(amount, asset, direction, duration_s, time_mode=("TIMER" if asset.endswith("_otc") else "TIME"))
                if not ok:
                    return {"executed": False, "status": "buy_rejected"}
                # Best-effort track
                win = await self.qx.check_win(getattr(self.qx.api, 'buy_id', None))
                return {"executed": True, "status": ("win" if win else "loss"), "won": bool(win)}
            except Exception as e:
                return {"executed": False, "status": f"exec_error:{e}"}


QX_MANAGER = _QuotexManager()


@app.on_event("startup")
async def _on_startup():
    # Connect proactively only if trading is enabled
    if os.environ.get("WEBHOOK_TRADE", "0").lower() in ("1", "true", "yes"):
        try:
            import asyncio as _aio
            await QX_MANAGER.connect()
            # init lock
            global _TRADE_LOCK
            _TRADE_LOCK = _TRADE_LOCK or __import__("asyncio").Lock()
        except Exception:
            pass


@app.post("/signal")
async def post_signal(payload: SignalIn) -> dict:
    p = payload.norm()

    if not p.asset:
        raise HTTPException(400, detail="asset required")
    if p.direction not in ("call", "put"):
        raise HTTPException(400, detail="direction must be 'call' or 'put'")

    # Build TelegramSignal equivalent
    t_sig = TelegramSignal(
        asset=p.asset,
        direction=p.direction,
        timeframe_min=int(p.timeframe_min or 1),
        trade_time=p.trade_time,
        trend=None,
        forecast_pct=p.forecast_pct,
        payout_pct=p.payout_pct,
        raw_block=p.raw or "",
    )
    # Schedule to epoch using provided tz or env override
    sched = schedule_signal(
        t_sig,
        default_expiry_min=int(p.timeframe_min or 1),
        lead_s=int(os.environ.get("S19_LEAD_S", "5") or 5),
        tz_offset_min=int(p.tz_offset_min or int(os.environ.get("S19_TZ_OFFSET_MIN", "330"))),
    )

    # Apply lightweight S19 filters (min forecast, lateness). Cooldowns can be added here if needed.
    min_fc = float(os.environ.get("S19_MIN_FORECAST", "0") or 0)
    sched = _filter(sched, min_forecast=min_fc, allow_grace_s=int(os.environ.get("S19_ALLOW_PAST_GRACE_S", "8") or 8))

    # Log CSV row like S19
    _log_signal(sched)

    # If immediate-execute mode enabled, ignore trade_time and place now after trend check
    do_trade = os.environ.get("WEBHOOK_TRADE", "0").lower() in ("1", "true", "yes")
    amount = float(os.environ.get("TRADE_AMOUNT", os.environ.get("WEBHOOK_AMOUNT", "1.0")) or 1.0)
    exec_result: dict | None = None
    trend_pass = None
    if do_trade and not sched.ignored:
        try:
            # Serialize trades to avoid overlap
            global _TRADE_LOCK
            _TRADE_LOCK = _TRADE_LOCK or __import__("asyncio").Lock()
            async with _TRADE_LOCK:
                trend_pass = await QX_MANAGER.trend_ok(sched.asset, sched.direction)
                if not trend_pass:
                    exec_result = {"executed": False, "status": "trend_block"}
                else:
                    exec_result = await QX_MANAGER.execute(sched.asset, sched.direction, int(sched.timeframe_min or 1), amount)
        except Exception as e:
            exec_result = {"executed": False, "status": f"exception:{e}"}

    # Quick summary for caller
    eta = sched.seconds_until_entry()
    resp = {
        "ok": True,
        "asset": sched.asset,
        "direction": sched.direction,
        "timeframe_min": sched.timeframe_min,
        "trade_time": sched.trade_time,
        "trade_epoch": sched.trade_epoch,
        "eta_to_entry_s": eta,
        "ignored": bool(sched.ignored),
        "reason": sched.reason,
        "executed": (exec_result or {}).get("executed") if exec_result is not None else False,
        "exec_status": (exec_result or {}).get("status") if exec_result is not None else None,
        "trend_pass": trend_pass,
    }
    # include outcome hints when available
    if exec_result is not None:
        if "won" in exec_result:
            resp["won"] = bool(exec_result["won"])  # type: ignore[index]
        if "delta" in exec_result:
            resp["delta"] = float(exec_result["delta"])  # type: ignore[index]
    return resp


@app.get("/")
def root() -> dict:
    return {
        "service": "Quotex MT4 Webhook",
        "endpoints": ["GET /health", "POST /signal"],
        "hint": "POST JSON: {asset, direction, timeframe_min, trade_time, tz_offset_min}",
    }


if __name__ == "__main__":
    # dev-run convenience: python app/webhook_server.py
    import uvicorn  # type: ignore
    port = int(os.environ.get("WEBHOOK_PORT", "8000"))
    uvicorn.run("app.webhook_server:app", host="0.0.0.0", port=port, reload=False)
