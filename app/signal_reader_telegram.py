import asyncio
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import AsyncIterator, Dict, List, Optional, Set

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None  # type: ignore


# Telegram-style signal blocks sample (from histories.txt):
# WIN ✅
# 💳 USDBDT-OTC
# 🔥 M1
# ⌛ 23:10:00
# 🔽 put
#
# 🚦 Tend: Sell
# 📈 Forecast: 73.35%
# 💸 Payout: 82.0%
#
# We parse asset, timeframe (minutes), trade_time (HH:MM[:SS]), direction, trend, forecast%, payout%.


@dataclass
class TelegramSignal:
    asset: str
    direction: str  # "call" | "put"
    timeframe_min: Optional[int] = None
    trade_time: Optional[str] = None  # HH:MM[:SS]
    trend: Optional[str] = None  # Buy/Sell
    forecast_pct: Optional[float] = None
    payout_pct: Optional[float] = None
    raw_block: str = ""

    def key(self) -> str:
        return f"{self.asset}|{self.trade_time or ''}|{self.direction}"


@dataclass
class ScheduledTelegramSignal(TelegramSignal):
    """Extended signal with scheduling metadata for Strategy19 (Telegram follower)."""

    trade_epoch: Optional[int] = None      # epoch seconds (IST based) when candle opens
    expiry_s: int = 60                     # default 1 minute expiry
    entry_lead_s: int = 5                  # place trade this many seconds *before* trade_epoch
    scheduled_at: Optional[int] = None     # epoch seconds when we decided to schedule
    reason: str = ""                        # filtering / gating notes
    ignored: bool = False                  # if True, do not trade (explain in reason)

    def seconds_until_entry(self, now_ts: Optional[float] = None) -> Optional[int]:
        if self.trade_epoch is None:
            return None
        import time as _t
        now_ts = now_ts or _t.time()
        return int(self.trade_epoch - self.entry_lead_s - now_ts)


_ASSET_RE = re.compile(r"^\s*[💳]?\s*([A-Z0-9\-_/]+)(?:\s*)$")
_ASSET_LINE_RE = re.compile(r"^\s*💳\s*([A-Z0-9\-_/]+)\s*$", re.IGNORECASE)
_TIMEFRAME_RE = re.compile(r"^\s*🔥\s*M(\d+)\s*$", re.IGNORECASE)
_TRADE_TIME_RE = re.compile(r"^\s*⌛\s*([0-2]?\d:\d{2}(?::\d{2})?)\s*$", re.IGNORECASE)
_DIRECTION_WORD_RE = re.compile(r"\b(call|put)\b", re.IGNORECASE)
_TREND_RE = re.compile(r"^\s*🚦\s*T(?:r|)end\s*:\s*(Buy|Sell)\s*$", re.IGNORECASE)
_FORECAST_RE = re.compile(r"^\s*📈\s*Forecast\s*:\s*([0-9]+(?:\.[0-9]+)?)%\s*$", re.IGNORECASE)
_PAYOUT_RE = re.compile(r"^\s*💸\s*Payout\s*:\s*([0-9]+(?:\.[0-9]+)?)%\s*$", re.IGNORECASE)


def _normalize_direction_from_line(s: str) -> Optional[str]:
    s_low = s.lower()
    if "🔼" in s_low or " call" in s_low:
        return "call"
    if "🔽" in s_low or " put" in s_low:
        return "put"
    m = _DIRECTION_WORD_RE.search(s)
    if m:
        v = m.group(1).lower()
        return "call" if v == "call" else "put"
    return None


class TelegramSignalParser:
    """Stateful line-by-line parser that emits TelegramSignal when sufficient fields are captured."""

    def __init__(self) -> None:
        self._state: Dict[str, Optional[str]] = {}
        self._block_lines: List[str] = []

    def reset(self) -> None:
        self._state.clear()
        self._block_lines.clear()

    def feed_line(self, line: str) -> Optional[TelegramSignal]:
        s = line.strip()
        if not s:
            # blank lines delimit blocks; try emitting if complete
            sig = self._maybe_emit()
            if sig:
                return sig
            # else keep waiting
            return None

        # Some channels compress multiple tokens (asset, timeframe, time, direction, trend, forecast, payout)
        # into one long line. We attempt to split such a composite line into pseudo-lines so existing
        # regex patterns match without large refactor. Heuristic: look for known emoji markers and
        # re-feed segments individually (excluding the current raw accumulation to avoid recursion loops).
        if any(em in s for em in ("💳","🔥","⌛","🚦","📈","💸")) and ' ' in s and '\n' not in s:
            # Split by emoji boundaries keeping the emoji at start of each segment
            parts = re.split(r"(?=(💳|🔥|⌛|🚦|📈|💸))", s)
            segs: List[str] = []
            cur = ""
            for p in parts:
                if not p:
                    continue
                if p in ("💳","🔥","⌛","🚦","📈","💸"):
                    if cur:
                        segs.append(cur.strip())
                    cur = p
                else:
                    cur += p
            if cur:
                segs.append(cur.strip())
            # If we produced multiple segments with at least 2 different emoji, treat as composite
            if len(segs) >= 2:
                out_sig: Optional[TelegramSignal] = None
                for seg in segs:
                    r = self.feed_line(seg + "\n")  # recursive feed per segment
                    if r:
                        out_sig = r
                return out_sig

        # Accumulate raw block text
        self._block_lines.append(s)

        # Recognize a new block start by an Asset line or WIN line
        if s.upper().startswith("WIN") or s.startswith("💳"):
            # If prior block was complete, emit before starting fresh
            prev = self._maybe_emit()
            # Reset for new block if asset line; WIN just marks outcome
            if s.startswith("💳"):
                self._state.clear()
                self._block_lines[:] = [s]
            if prev:
                return prev

        # Asset
        m = _ASSET_LINE_RE.match(s)
        if m:
            self._state["asset"] = m.group(1).strip()

        # Timeframe
        m = _TIMEFRAME_RE.match(s)
        if m:
            self._state["timeframe_min"] = m.group(1)

        # Trade time
        m = _TRADE_TIME_RE.match(s)
        if m:
            tt = m.group(1).strip().strip('"')
            self._state["trade_time"] = tt

        # Direction (arrow or word)
        dire = _normalize_direction_from_line(s)
        if dire:
            self._state["direction"] = dire

        # Trend
        m = _TREND_RE.match(s)
        if m:
            self._state["trend"] = m.group(1).title()

        # Forecast
        m = _FORECAST_RE.match(s)
        if m:
            try:
                self._state["forecast_pct"] = str(float(m.group(1)))
            except Exception:
                pass

        # Payout
        m = _PAYOUT_RE.match(s)
        if m:
            try:
                self._state["payout_pct"] = str(float(m.group(1)))
            except Exception:
                pass

        # Emit ASAP when core fields are present
        return self._maybe_emit(partial_ok=True)

    def _maybe_emit(self, partial_ok: bool = False) -> Optional[TelegramSignal]:
        asset = self._state.get("asset")
        direction = self._state.get("direction")
        trade_time = self._state.get("trade_time")
        if not asset or not direction:
            return None
        # For partial_ok, allow missing trade_time/timeframe; emit when direction+asset set
        if not partial_ok and (asset and direction and trade_time is None):
            return None

        tf = self._state.get("timeframe_min")
        trend = self._state.get("trend")
        f = self._state.get("forecast_pct")
        p = self._state.get("payout_pct")
        raw = "\n".join(self._block_lines[-12:])
        sig = TelegramSignal(
            asset=asset,
            direction=direction,
            timeframe_min=int(tf) if tf else None,
            trade_time=trade_time,
            trend=trend,
            forecast_pct=float(f) if f is not None else None,
            payout_pct=float(p) if p is not None else None,
            raw_block=raw,
        )
        # After emitting once per block, clear direction to avoid duplicates
        self._state.pop("direction", None)
        return sig


async def tail_file(path: str) -> AsyncIterator[TelegramSignal]:
    """Tail a UTF-8 file and yield TelegramSignal as they are parsed.
    Tolerates emoji and partial lines; does not seek to end to allow initial backlog.
    """
    parser = TelegramSignalParser()
    seen: Set[str] = set()

    # Wait for file to appear
    for _ in range(200):
        if os.path.exists(path):
            break
        await asyncio.sleep(0.1)

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                line = f.readline()
                if not line:
                    await asyncio.sleep(0.1)
                    continue
                sig = parser.feed_line(line)
                if sig:
                    k = sig.key()
                    if k in seen:
                        continue
                    seen.add(k)
                    yield sig
    except asyncio.CancelledError:
        raise


def parse_file_once(path: str) -> List[TelegramSignal]:
    """Parse entire file once (no tail) and return all signals, de-duplicated by (asset,time,direction)."""
    parser = TelegramSignalParser()
    seen: Set[str] = set()
    out: List[TelegramSignal] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            sig = parser.feed_line(line)
            if sig:
                k = sig.key()
                if k in seen:
                    continue
                seen.add(k)
                out.append(sig)
    return out


def ist_epoch_for_trade_time(hhmm: str, ref_ts: Optional[float] = None) -> int:
    """Convert HH:MM[:SS] assumed IST clock to epoch seconds.
    If time already passed today, schedule for next day.
    """
    try:
        parts = hhmm.split(":")
        h, m = int(parts[0]), int(parts[1])
        s = int(parts[2]) if len(parts) > 2 else 0
        tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else timezone.utc
        now = datetime.now(tz) if ref_ts is None else datetime.fromtimestamp(ref_ts, tz)
        trade_dt = now.replace(hour=h, minute=m, second=s, microsecond=0)
        if trade_dt <= now:
            # next day
            trade_dt = trade_dt.replace(day=trade_dt.day) + timedelta(days=1)  # type: ignore[name-defined]
        return int(trade_dt.timestamp())
    except Exception:
        return 0


# ---- Strategy19 Helpers (non-invasive; safe to import) ----
def epoch_for_trade_time(hhmm: str, tz_offset_min: int, ref_ts: Optional[float] = None) -> int:
    """Generic helper: interpret HH:MM[:SS] in a fixed offset timezone, return UTC epoch.

    tz_offset_min = minutes east of UTC (e.g., IST=+330, UTC-3 = -180).
    Rolls to next day if time already passed in that local offset.
    Returns 0 on error.
    """
    try:
        from datetime import timezone as _tz, timedelta as _td, datetime as _dt
        parts = hhmm.split(":")
        h, m = int(parts[0]), int(parts[1])
        s = int(parts[2]) if len(parts) > 2 else 0
        base_ts = ref_ts or __import__("time").time()
        offset = _td(minutes=tz_offset_min)
        tz = _tz(offset)
        now_local = _dt.fromtimestamp(base_ts, tz)
        trade_local = now_local.replace(hour=h, minute=m, second=s, microsecond=0)
        if trade_local <= now_local:
            trade_local = trade_local + _td(days=1)
        return int(trade_local.astimezone(_tz.utc).timestamp())
    except Exception:
        return 0


def schedule_signal(
    sig: TelegramSignal,
    default_expiry_min: int = 1,
    lead_s: int = 5,
    ref_ts: Optional[float] = None,
    tz_offset_min: int = 330,
) -> ScheduledTelegramSignal:
    """Create a ScheduledTelegramSignal computing trade_epoch from trade_time.

    tz_offset_min controls interpretation of trade_time (default IST +330).
    If trade_time missing: immediate (now + lead_s).
    """
    import time as _t
    now_ts = ref_ts or _t.time()
    # Allow runtime override of timezone offset used to interpret trade_time
    _env_tz = os.environ.get("S19_TT_TZ_OFFSET_MIN")
    if _env_tz:
        try:
            tz_offset_min = int(_env_tz)
        except Exception:
            pass

    # Past-time handling policy: roll | immediate | next_minute | skip
    past_policy = os.environ.get("S19_PAST_TIME_POLICY", "roll").lower()
    te_skip = False
    if sig.trade_time:
        # Optional timezone remap: signal time label belongs to source tz
        # but we execute at SAME HH:MM in destination tz (e.g. IST -> UTC-3).
        # Enable: S19_TT_REMAP_TO_DST=1; set S19_TT_DST_TZ_OFFSET_MIN
        remap_enabled = os.environ.get(
            "S19_TT_REMAP_TO_DST", "0"
        ).lower() in ("1", "true", "yes")
        if remap_enabled:
            try:
                dst_off = int(
                    os.environ.get("S19_TT_DST_TZ_OFFSET_MIN", "-180")
                )
            except Exception:
                dst_off = -180
            # Parse components
            parts_rm = sig.trade_time.split(":")
            try:
                _rh = int(parts_rm[0])
                _rm = int(parts_rm[1])
                _rs = int(parts_rm[2]) if len(parts_rm) > 2 else 0
            except Exception:
                _rh = 0
                _rm = 0
                _rs = 0
            from datetime import timezone as _tz
            from datetime import timedelta as _td, datetime as _dt
            dst_tz = _tz(_td(minutes=dst_off))
            # Build target time in destination tz with same HH:MM[:SS]
            dst_dt = _dt.fromtimestamp(now_ts, dst_tz).replace(
                hour=_rh, minute=_rm, second=_rs, microsecond=0
            )
            if dst_dt <= _dt.fromtimestamp(now_ts, dst_tz):
                dst_dt = dst_dt + _td(days=1)
            te_candidate = int(dst_dt.astimezone(_tz.utc).timestamp())
            # Append reason later via sanity_reason if large shift occurred
        else:
            te_candidate = epoch_for_trade_time(
                sig.trade_time, tz_offset_min, ref_ts=now_ts
            )
    # If candidate is >30m ahead & label already past: treat as past.
    # Heuristic: diff > 1800s => next-day roll likely happened.
        if te_candidate - now_ts > 1800 and past_policy != "roll":
            if past_policy == "immediate":
                te = int(now_ts + max(0, lead_s - 1))  # near-immediate
            elif past_policy == "next_minute":
                te = (int(now_ts // 60) * 60) + 60
            elif past_policy == "skip":
                te = te_candidate  # set but mark skip below
                te_skip = True
            else:
                te = te_candidate
        else:
            te = te_candidate
    else:
        te = int(now_ts + lead_s)

    # --- ETA sanity correction block ---
    # Problem: tt=HH:MM produced huge eta (> hours) though trade was imminent.
    # Cause: unwanted day rollover / timezone mismatch. Detect & correct.
    # and attempt same-day correction or fallback policy.
    if sig.trade_time:
        try:
            # Max acceptable eta before sanity logic (default 15m)
            max_eta = int(os.environ.get("S19_TT_MAX_ETA", "900"))
        except Exception:
            max_eta = 900
        eta_now = te - now_ts
        if eta_now > max_eta:
            sanity_policy = os.environ.get(
                "S19_TT_SANITY_POLICY", "adjust"
            ).lower()
            # Re-parse same-day (without forced rollover) and see if closer
            try:
                from datetime import timezone as _tz
                from datetime import timedelta as _td, datetime as _dt
                parts = sig.trade_time.split(":")
                _h = int(parts[0])
                _m = int(parts[1])
                _s = int(parts[2]) if len(parts) > 2 else 0
                offset = _td(minutes=tz_offset_min)
                tz = _tz(offset)
                now_local = _dt.fromtimestamp(now_ts, tz)
                same_day_dt = now_local.replace(
                    hour=_h, minute=_m, second=_s, microsecond=0
                )
                alt_te = int(same_day_dt.astimezone(_tz.utc).timestamp())
                alt_eta = alt_te - now_ts
            except Exception:
                alt_te = te
                alt_eta = eta_now
            applied = False
            # Fast one-day-back heuristic: if eta is huge but going back a day
            # yields a reasonable near-term eta, adopt it immediately.
            try:
                day_back_threshold = int(
                    os.environ.get("S19_TT_DAY_BACK_THRESHOLD_S", "3600")
                )
            except Exception:
                day_back_threshold = 3600
            if eta_now > day_back_threshold and not applied:
                alt_te2 = te - 86400
                alt_eta2 = alt_te2 - now_ts
                if 0 <= alt_eta2 <= max_eta:
                    te = alt_te2
                    applied = True
                    sanity_reason = "eta_sanity_day_back_fast"
            # If alt_eta is small positive (0< <= max_eta) prefer it
            if 0 <= alt_eta <= max_eta and alt_te <= te:
                te = alt_te
                applied = True
                sanity_reason = "eta_sanity_same_day"
            # If rollover made alt_eta negative but within 2m past.
            if (
                not applied
                and -120 <= alt_eta < 0
                and sanity_policy in ("adjust", "immediate", "next_minute")
            ):
                # We expected this minute but we are already a few seconds late
                applied = True
                sanity_reason = "eta_sanity_late"
                te = int(now_ts + max(0, lead_s - 1))
            if not applied:
                if sanity_policy in ("immediate", "now"):
                    te = int(now_ts + max(0, lead_s - 1))
                    sanity_reason = "eta_sanity_immediate"
                elif sanity_policy == "next_minute":
                    te = (int(now_ts // 60) * 60) + 60
                    sanity_reason = "eta_sanity_next_minute"
                elif sanity_policy == "keep":
                    sanity_reason = "eta_sanity_keep"
                else:  # fallback: eta >12h maybe wrong day -> -1d
                    if eta_now > 43200:  # >12h means likely wrong day
                        te2 = te - 86400
                        if te2 - now_ts > 0:
                            te = te2
                            sanity_reason = "eta_sanity_day_back"
                        else:
                            sanity_reason = "eta_sanity_noop"
                    else:
                        sanity_reason = "eta_sanity_noop"
            # Attach reason later after sched created if not skip
        else:
            sanity_reason = None
    else:
        sanity_reason = None
    # Normalize asset naming for OTC pairs: convert trailing '-OTC' to '_otc'
    try:
        import re as _re
        a = sig.asset.strip()
        if _re.search(r"-OTC$", a, _re.IGNORECASE):
            core = _re.sub(r"-OTC$", "", a, flags=_re.IGNORECASE)
            core = core.replace(" ", "").replace("-", "").upper()
            sig.asset = f"{core}_otc"
        elif _re.search(r"_OTC$", a):  # upper suffix -> lower
            sig.asset = a[:-4] + "_otc"
    except Exception:
        pass
    expiry_s = (sig.timeframe_min or default_expiry_min) * 60
    sched = ScheduledTelegramSignal(
        **sig.__dict__,  # type: ignore[arg-type]
        trade_epoch=te,
        expiry_s=expiry_s,
        entry_lead_s=lead_s,
        scheduled_at=int(now_ts),
    )
    if te_skip:
        sched.ignored = True
        sched.reason = "past_skip"
    elif past_policy in ("immediate", "next_minute") and sig.trade_time:
        sched.reason = f"past_{past_policy}"
    if sanity_reason and not sched.reason:
        sched.reason = sanity_reason
    return sched


async def tail_file_scheduled(
    path: str,
    lead_s: int = 5,
    default_expiry_min: int = 1,
    tz_offset_min: int = 330,
) -> AsyncIterator[ScheduledTelegramSignal]:
    """Wrapper over tail_file that yields ScheduledTelegramSignal objects.

    Deduplicates using base key().
    """
    async for base in tail_file(path):
        yield schedule_signal(
            base,
            default_expiry_min=default_expiry_min,
            lead_s=lead_s,
            tz_offset_min=tz_offset_min,
        )


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Parse histories.txt blocks into structured signals"
    )
    ap.add_argument(
        "--file",
        default=os.path.join(os.path.dirname(__file__), "..", "histories.txt"),
    )
    ap.add_argument(
        "--tail", action="store_true", help="Tail the file and stream signals"
    )
    args = ap.parse_args()

    path = os.path.abspath(args.file)
    if args.tail:
        async def _run():
            async for sig in tail_file(path):
                print(json.dumps(sig.__dict__, ensure_ascii=False))
        asyncio.run(_run())
    else:
        sigs = parse_file_once(path)
        for s in sigs:
            print(json.dumps(s.__dict__, ensure_ascii=False))

