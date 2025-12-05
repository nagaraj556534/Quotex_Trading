import os
import csv
import asyncio
import time
from typing import Optional
from dataclasses import dataclass

# Reuse existing parser + scheduling helpers (support direct + package import)
try:
    from signal_reader_telegram import (
        TelegramSignalParser,
        schedule_signal,
        ScheduledTelegramSignal,
    )
except Exception:
    try:
        from .signal_reader_telegram import (  # type: ignore
            TelegramSignalParser,
            schedule_signal,
            ScheduledTelegramSignal,
        )
    except Exception as _e:  # pragma: no cover
        raise ImportError(f"Cannot import signal_reader_telegram: {_e}")

# Lazy import Telethon only when running
try:
    from telethon import TelegramClient, events
except Exception:  # pragma: no cover - telethon optional at import time
    TelegramClient = None  # type: ignore
    events = None  # type: ignore


@dataclass
class S19Config:
    api_id: int
    api_hash: str
    phone: str
    group: str  # username ("mygroup") OR numeric id ("-1001234567890")
    session_name: str = "s19_session"
    # optional future filter (broker lookup needed)
    min_payout: float = 0.0
    # forecast pct filter
    min_forecast: float = 0.0
    # seconds before trade_time to send order
    lead_s: int = 5
    log_csv: str = os.path.join(
        os.path.dirname(__file__), "..", "strategy19_signals.csv"
    )
    # future toggle for actual trade execution
    trade: bool = False
    default_expiry_min: int = 1
    cooldown_same_asset_s: int = 90
    # if already passed by <= this, still attempt
    allow_past_grace_s: int = 8
    tz_offset_min: int = 330  # default IST (+330). Set -180 for UTC-3.
    # auto reconnect loop (placeholder – compatibility with main.py)
    auto_reconnect: bool = True


_HEADER = [
    "ts_local",
    "asset",
    "direction",
    "timeframe_min",
    "trade_time",
    "forecast_pct",
    "payout_pct",
    "trade_epoch",
    "seconds_until_entry",
    "ignored",
    "reason",
    "raw",
]


def _ensure_log(path: str):
    try:
        new = not os.path.exists(path)
        if new:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", newline="", encoding="utf-8") as f:
            if new:
                csv.writer(f).writerow(_HEADER)
    except Exception:
        pass


def _log_signal(path: str, sig: ScheduledTelegramSignal):
    try:
        _ensure_log(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
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
                sig.raw_block.replace("\n", " | ")[:200],
            ])
    except Exception:
        pass


class Strategy19Follower:
    def __init__(self, cfg: S19Config, execute_cb=None):
        self.cfg = cfg
        self.parser = TelegramSignalParser()
        self.seen_keys: set[str] = set()
        self.cooldowns: dict[str, float] = {}
        self.execute_cb = execute_cb  # async function(sig) -> None
        # Fallback type if telethon missing
        self.client: Optional[object] = None
        # --- Two-step (base text then sticker) direction mode ---
        # Enable with env S19_TWO_STEP_STICKER=1
        # Base message: pair + timeframe (no direction).
        # Next sticker (⬇️ PUT / ⬆️ CALL) supplies direction.
    # Tamil: முதலில் pair+timeframe msg; அடுத்த sticker PUT/CALL trade.
        self.two_step_enabled = (
            os.environ.get("S19_TWO_STEP_STICKER", "0").lower()
            in ("1", "true", "yes")
        )
        try:
            self.two_step_ttl = int(
                os.environ.get("S19_TWO_STEP_STICKER_TTL", "60")
            )
        except Exception:
            self.two_step_ttl = 60
        # Emoji mapping lists (comma separated) env override
        put_list_env = os.environ.get(
            "S19_STICKER_PUT_EMOJI_LIST",
            # include plain arrow (no variation) + extra down symbols
            "⬇️,⬇,🔻,👇",
        ).split(",")
        call_list_env = os.environ.get(
            "S19_STICKER_CALL_EMOJI_LIST",
            # include plain arrow (no variation) + extra up symbols
            "⬆️,⬆,🔺,👆",
        ).split(",")
        self.two_step_put_emojis = {
            e.strip() for e in put_list_env if e.strip()
        }
        self.two_step_call_emojis = {
            e.strip() for e in call_list_env if e.strip()
        }
    # Fallback: any media (sticker/photo) with no emoji
    # Env S19_TWO_STEP_FALLBACK_MEDIA_DIRECTION=put|call
    # Tamil: emoji identify ஆகவில்லை என்றால் default direction
        fb_dir = os.environ.get(
            "S19_TWO_STEP_FALLBACK_MEDIA_DIRECTION", ""
        ).lower().strip()
        self.two_step_fallback_media_direction = (
            fb_dir if fb_dir in ("put", "call") else ""
        )
        # Debug toggle
        self.two_step_debug = (
            os.environ.get("S19_TWO_STEP_DEBUG", "0").lower()
            in ("1", "true", "yes")
        )
        # Default direction if color / other heuristics ambiguous
        dd = os.environ.get(
            "S19_TWO_STEP_STICKER_DEFAULT_DIRECTION", ""
        ).lower().strip()
        self.two_step_sticker_default_direction = (
            dd if dd in ("put", "call") else ""
        )
        # Hold pending base messages awaiting direction sticker
        # each pending dict: asset,str timeframe_min,int|None raw,str ts,float
        self._pending_bases: list[dict] = []

    def _filter(self, sig: ScheduledTelegramSignal) -> ScheduledTelegramSignal:
        # Forecast filter
        if (
            sig.forecast_pct is not None
            and sig.forecast_pct < self.cfg.min_forecast
        ):
            sig.ignored = True
            sig.reason = f"forecast<{self.cfg.min_forecast}"
            return sig
        # Cooldown
        now = time.time()
        cd_until = self.cooldowns.get(sig.asset, 0.0)
        if now < cd_until:
            sig.ignored = True
            sig.reason = "asset_cooldown"
            return sig
        # Late arrival check
        if (
            sig.trade_epoch is not None
            and (sig.trade_epoch - sig.entry_lead_s) < now
        ):
            # Already past intended entry moment
            if (
                now - (sig.trade_epoch - sig.entry_lead_s)
                > self.cfg.allow_past_grace_s
            ):
                sig.ignored = True
                sig.reason = "missed_entry"
                return sig
        sig.reason = "ok"
        return sig

    async def _handle_signal(self, sig: ScheduledTelegramSignal):
        k = sig.key()
        if k in self.seen_keys:
            return
        self.seen_keys.add(k)
        sig = self._filter(sig)
        _log_signal(self.cfg.log_csv, sig)
        # Print summary
        eta = sig.seconds_until_entry()
        print(
            f"[S19] {sig.asset} {sig.direction} tt={sig.trade_time} "
            f"eta={eta}s fore={sig.forecast_pct} pay={sig.payout_pct} "
            f"ignored={sig.ignored} reason={sig.reason}"
        )
        # Execute via injected callback
        if self.cfg.trade and not sig.ignored and self.execute_cb:
            try:
                await self.execute_cb(sig)
            except Exception as e:  # pragma: no cover
                print(f"[S19][EXEC_ERR] {e}")
        # Set cooldown after scheduling (even if not traded) to prevent spam
        self.cooldowns[sig.asset] = (
            time.time() + self.cfg.cooldown_same_asset_s
        )

    async def start(self):
        if TelegramClient is None:
            raise RuntimeError(
                "Telethon not installed. Run: pip install telethon"
            )
        self.client = TelegramClient(
            self.cfg.session_name, self.cfg.api_id, self.cfg.api_hash
        )
        await self.client.start(phone=self.cfg.phone)

        group = self.cfg.group
        
        @self.client.on(events.NewMessage(chats=group))  # type: ignore
        async def _on_msg(event):  # noqa: N802
            try:
                txt = event.raw_text or ""

                # --- Two-step direction handling ---
                if self.two_step_enabled:
                    # 1) Direction sticker? (raw_text empty usually).
                    # Telethon may expose event.message.sticker.emoji
                    direction_from_sticker: Optional[str] = None
                    try:
                        # Try attribute paths defensively
                        msg_obj = getattr(event, "message", None)
                        if self.two_step_debug and msg_obj is not None:
                            try:
                                print(
                                    "[S19][2STEP][DBG] has_sticker=", bool(getattr(msg_obj, "sticker", None)),
                                    "has_photo=", bool(getattr(msg_obj, "photo", None)),
                                    "has_doc=", bool(getattr(msg_obj, "document", None)),
                                    "file_name=", getattr(getattr(msg_obj, "file", None), "name", None),
                                    "text_len=", len(txt),
                                )
                            except Exception:
                                pass
                        # sticker emoji maybe at .sticker.emoji or .file.emoji
                        sticker_emoji = None
                        if msg_obj is not None:
                            sticker_attr = getattr(msg_obj, "sticker", None)
                            if sticker_attr is not None:
                                sticker_emoji = (
                                    getattr(sticker_attr, "emoji", None)
                                    or getattr(sticker_attr, "emoticon", None)
                                )
                            if sticker_emoji is None:
                                file_attr = getattr(msg_obj, "file", None)
                                sticker_emoji = (
                                    getattr(file_attr, "emoji", None)
                                    if file_attr
                                    else None
                                )
                        # Exported HTML shows emojis like ⬇️ for PUT
                        if sticker_emoji in self.two_step_put_emojis:
                            direction_from_sticker = "put"
                        elif sticker_emoji in self.two_step_call_emojis:
                            direction_from_sticker = "call"
                        # If still unknown, look for words in caption/text
                        if not direction_from_sticker and txt:
                            low = txt.lower()
                            if " put" in f" {low}" or low.strip() == "put":
                                direction_from_sticker = "put"
                            elif " call" in f" {low}" or low.strip() == "call":
                                direction_from_sticker = "call"
                        # Inspect sticker/document attributes (alt text)
                        if not direction_from_sticker and msg_obj is not None:
                            try:
                                doc = getattr(msg_obj, "document", None)
                                if doc and getattr(doc, "attributes", None):
                                    for att in doc.attributes:
                                        alt = getattr(att, "alt", "")
                                        if isinstance(alt, str) and alt:
                                            al = alt.lower()
                                            if "put" in al:
                                                direction_from_sticker = "put"
                                                break
                                            if "call" in al:
                                                direction_from_sticker = "call"
                                                break
                            except Exception:
                                pass
                        # Inspect file name
                        if not direction_from_sticker and msg_obj is not None:
                            try:
                                fobj = getattr(msg_obj, "file", None)
                                fname = getattr(fobj, "name", None)
                                if isinstance(fname, str):
                                    fl = fname.lower()
                                    if "put" in fl:
                                        direction_from_sticker = "put"
                                    elif "call" in fl:
                                        direction_from_sticker = "call"
                            except Exception:
                                pass
                        # Color heuristic always attempted last if still unknown
                        if not direction_from_sticker and msg_obj is not None:
                            try:
                                media_obj = (
                                    getattr(msg_obj, "photo", None)
                                    or getattr(msg_obj, "sticker", None)
                                    or getattr(msg_obj, "document", None)
                                )
                                if media_obj is not None:
                                    import tempfile, shutil
                                    tmpdir = tempfile.gettempdir()
                                    tmp_path = os.path.join(
                                        tmpdir,
                                        f"s19_media_{int(time.time()*1000)}.bin",
                                    )
                                    await event.download_media(file=tmp_path)
                                    try:
                                        from PIL import Image  # type: ignore
                                        with Image.open(tmp_path) as im:
                                            im = im.convert("RGB").resize((32, 32))
                                            pixels = list(im.getdata())
                                        r = sum(p[0] for p in pixels) / len(pixels)
                                        g = sum(p[1] for p in pixels) / len(pixels)
                                        if r - g > 18:  # threshold tuned a bit higher
                                            direction_from_sticker = "put"
                                            if self.two_step_debug:
                                                print(
                                                    f"[S19][2STEP][COLOR] avgR={r:.1f} avgG={g:.1f} -> put"
                                                )
                                        elif g - r > 18:
                                            direction_from_sticker = "call"
                                            if self.two_step_debug:
                                                print(
                                                    f"[S19][2STEP][COLOR] avgR={r:.1f} avgG={g:.1f} -> call"
                                                )
                                        else:
                                            if (
                                                self.two_step_sticker_default_direction
                                            ):
                                                direction_from_sticker = (
                                                    self.two_step_sticker_default_direction
                                                )
                                                if self.two_step_debug:
                                                    print(
                                                        f"[S19][2STEP][COLOR] ambiguous -> default "
                                                        f"{direction_from_sticker}"
                                                    )
                                            elif self.two_step_debug:
                                                print(
                                                    f"[S19][2STEP][COLOR] ambiguous avgR={r:.1f} avgG={g:.1f}"
                                                )
                                    except ImportError:
                                        if self.two_step_debug:
                                            print(
                                                "[S19][2STEP][COLOR] Pillow missing (restart after install)"
                                            )
                                    except Exception as _ce:
                                        if self.two_step_debug:
                                            print(
                                                f"[S19][2STEP][COLOR] err {_ce}"
                                            )
                                    finally:
                                        try:
                                            if os.path.exists(tmp_path):
                                                os.remove(tmp_path)
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                    except Exception:
                        direction_from_sticker = None

                    # Raw dump (structure only) if enabled and still undecided
                    if (
                        not direction_from_sticker
                        and self.two_step_debug
                        and os.environ.get(
                            "S19_TWO_STEP_RAW_DUMP", "0"
                        ).lower()
                        in ("1", "true", "yes")
                    ):
                        try:
                            msg_obj = getattr(event, "message", None)
                            d = msg_obj.to_dict() if msg_obj else {}
                            # Trim heavy fields
                            keys_show = [
                                k
                                for k in d.keys()
                                if k
                                not in (
                                    "photo",
                                    "media",
                                    "thumbs",
                                    "file",
                                )
                            ]
                            subset = {k: d[k] for k in keys_show[:15]}
                            attrs_val = d.get("attributes", "")
                            if isinstance(attrs_val, (list, tuple)):
                                attrs_val = str(attrs_val)[:120]
                            else:
                                attrs_val = str(attrs_val)[:120]
                            print(
                                f"[S19][2STEP][RAW] keys={list(subset.keys())}"
                            )
                            print(f"[S19][2STEP][RAW] attrs={attrs_val}")
                        except Exception:
                            pass

                    # Last-resort default direction if configured
                    if (
                        not direction_from_sticker
                        and self.two_step_sticker_default_direction
                    ):
                        direction_from_sticker = (
                            self.two_step_sticker_default_direction
                        )
                        if self.two_step_debug:
                            print("[S19][2STEP][DEFAULT] -> " + direction_from_sticker)

                    # Fallback: media present & fallback configured
                    if (
                        not direction_from_sticker
                        and self.two_step_fallback_media_direction
                    ):
                        try:
                            msg_obj = getattr(event, "message", None)
                            if msg_obj is not None and getattr(
                                msg_obj, "media", None
                            ) is not None and not txt.strip():
                                # Assume configured direction
                                direction_from_sticker = (
                                    self.two_step_fallback_media_direction
                                )
                                print(
                                    f"[S19][2STEP] Fallback media -> "
                                    f"{direction_from_sticker}"
                                )
                        except Exception:
                            pass

                    if direction_from_sticker:
                        # Match last pending base within TTL
                        now_ts = time.time()
                        # purge expired first
                        self._pending_bases = [
                            p
                            for p in self._pending_bases
                            if now_ts - p["ts"] <= self.two_step_ttl
                        ]
                        for base in reversed(self._pending_bases):
                            # Use and remove
                            try:
                                # local import to avoid circular at top level
                                from signal_reader_telegram import (
                                    TelegramSignal,
                                )
                            except Exception:
                                from .signal_reader_telegram import (  # noqa
                                    TelegramSignal,
                                )
                            # Inject trade_time if captured in pending
                            trade_time_val = base.get("trade_time")
                            sig = TelegramSignal(
                                asset=base["asset"],
                                direction=direction_from_sticker,
                                timeframe_min=base["timeframe_min"],
                                trade_time=trade_time_val,
                                trend=None,
                                forecast_pct=None,
                                payout_pct=None,
                                raw_block=(
                                    base["raw"]
                                    + f"\n[STICKER->{direction_from_sticker}]"
                                ),
                            )
                            # tz override: temporarily set env for scheduling
                            tz_env_backup = os.environ.get(
                                "S19_TT_TZ_OFFSET_MIN"
                            )
                            tz_override = base.get("tz_override")
                            if tz_override is not None:
                                os.environ["S19_TT_TZ_OFFSET_MIN"] = str(
                                    tz_override
                                )
                            sched = schedule_signal(
                                sig,
                                default_expiry_min=self.cfg.default_expiry_min,
                                lead_s=self.cfg.lead_s,
                                tz_offset_min=self.cfg.tz_offset_min,
                            )
                            # Restore env
                            if tz_override is not None:
                                if tz_env_backup is not None:
                                    os.environ["S19_TT_TZ_OFFSET_MIN"] = (
                                        tz_env_backup
                                    )
                                else:
                                    os.environ.pop(
                                        "S19_TT_TZ_OFFSET_MIN", None
                                    )
                            await self._handle_signal(sched)
                            self._pending_bases.remove(base)
                            break
                        return  # processed sticker; skip normal parser

                    # 2) Base message detection: 'USD DZD  OTC' + '1 MIN'
                    base_lines = [
                        ln.strip()
                        for ln in txt.splitlines()
                        if ln.strip()
                    ]
                    if base_lines:
                        # New extended format example:
                        # Time Zone UTC:-5:30
                        # 🔔One Minute Signal🔔
                        # USD PKR OTC  Time now 13:50
                        # 35$ Capital ... (ignored)
                        # Capture: asset=USDPKR_otc tf=1 trade_time=13:50
                        # tz override from 'Time Zone UTC:-5:30'
                        # Tamil: அந்த line இருந்தா local tz override செய்கிறோம்
                        import re as _re
                        tz_override_min: Optional[int] = None
                        for bl in base_lines[:3]:
                            m_tz = _re.search(
                                r"Time\s*Zone\s*UTC:?\s*([+-]?\d{1,2})(?::(\d{1,2}))?",
                                bl,
                                _re.IGNORECASE,
                            )
                            if m_tz:
                                try:
                                    hh = int(m_tz.group(1))
                                    mm = int(m_tz.group(2) or 0)
                                    sign = 1 if hh >= 0 else -1
                                    tz_override_min = hh * 60 + sign * mm
                                except Exception:
                                    tz_override_min = None
                        # Detect timeframe from bell line
                        if any("One Minute" in bl for bl in base_lines[:3]):
                            bell_tf = 1
                        else:
                            bell_tf = None
                        # Asset + time line detection with 'Time now'
                        time_line = None
                        for bl in base_lines:
                            if "Time now" in bl or "Time Now" in bl:
                                time_line = bl
                                break
                        # If time_line present, parse asset pairs & HH:MM
                        if time_line:
                            # Remove duplicate spaces
                            tmp = _re.sub(r"\s+", " ", time_line)
                            # Extract time
                            m_tm = _re.search(r"(\d{1,2}:\d{2})", tmp)
                            trade_time_val = m_tm.group(1) if m_tm else None
                            # Extract first two currency codes before 'OTC'
                            m_asset2 = _re.search(
                                r"^([A-Z]{3})\s+([A-Z]{3})\s+OTC",
                                tmp,
                            )
                            if m_asset2:
                                raw_asset_new = (
                                    m_asset2.group(1) + m_asset2.group(2)
                                ).upper() + "_otc"
                                tf_use = bell_tf or 1
                                # Save pending with trade_time for schedule
                                self._pending_bases.append(
                                    {
                                        "asset": raw_asset_new,
                                        "timeframe_min": tf_use,
                                        "raw": txt.strip(),
                                        "ts": time.time(),
                                        "trade_time": trade_time_val,
                                        "tz_override": tz_override_min,
                                    }
                                )
                                print(
                                    f"[S19][2STEP] Base asset={raw_asset_new} "
                                    f"tf={tf_use} time={trade_time_val} "
                                    f"tz={tz_override_min} "
                                    f"wait sticker <={self.two_step_ttl}s"
                                )
                                return
                        # Heuristic: first line has 2 currency codes + OTC
                        asset_line = base_lines[0]
                        m_asset = _re.match(
                            r"^([A-Z]{3})\s+([A-Z]{3})\s+OTC", asset_line
                        )
                        if m_asset:
                            raw_asset = (
                                m_asset.group(1) + m_asset.group(2)
                            ).upper() + "_otc"
                            # Find timeframe line
                            timeframe_min = None
                            for bl in base_lines[1:3]:
                                m_tf = _re.search(r"(\d+)\s*MIN", bl)
                                if m_tf:
                                    try:
                                        timeframe_min = int(m_tf.group(1))
                                        break
                                    except Exception:
                                        pass
                            # Accept only if no direction hint in message
                            low_txt = txt.lower()
                            if (
                                "put" not in low_txt
                                and "call" not in low_txt
                                and "⬇" not in low_txt
                                and "⬆" not in low_txt
                            ):
                                self._pending_bases.append(
                                    {
                                        "asset": raw_asset,
                                        "timeframe_min": timeframe_min,
                                        "raw": txt.strip(),
                                        "ts": time.time(),
                                        "trade_time": None,
                                        "tz_override": None,
                                    }
                                )
                                print(
                                    f"[S19][2STEP] Base asset={raw_asset} "
                                    f"tf={timeframe_min} wait sticker <="
                                    f"{self.two_step_ttl}s"
                                )
                                return  # hold until direction sticker

                # --- Default single-message parsing path ---
                for line in txt.splitlines():
                    sig = self.parser.feed_line(line + "\n")
                    if sig:
                        sched = schedule_signal(
                            sig,
                            default_expiry_min=self.cfg.default_expiry_min,
                            lead_s=self.cfg.lead_s,
                            tz_offset_min=self.cfg.tz_offset_min,
                        )
                        await self._handle_signal(sched)
            except Exception as e:  # pragma: no cover
                print(f"[S19][ERR] handler: {e}")

        # Fresh start listening banner
        try:
            from colorama import Fore, Style  # type: ignore
        except Exception:  # pragma: no cover
            class _No:
                RED = YELLOW = MAGENTA = CYAN = GREEN = RESET = BLUE = ""
            Fore = Style = _No()  # type: ignore
        print(
            Fore.CYAN
            + f"[S19] Listening group={group} session={self.cfg.session_name}"
            + Style.RESET_ALL
        )
        print(Fore.CYAN + "[S19] (Ctrl+C to stop)" + Style.RESET_ALL)
        await self.client.run_until_disconnected()


def load_cfg_from_env() -> S19Config:
    try:
        api_id = int(os.environ.get("TELEGRAM_API_ID", "0"))
    except Exception:
        api_id = 0
    session_name = os.environ.get(
        "TELEGRAM_SESSION", "s19_session"
    ).strip() or "s19_session"
    return S19Config(
        api_id=api_id,
        api_hash=os.environ.get("TELEGRAM_API_HASH", ""),
        phone=os.environ.get("TELEGRAM_PHONE", ""),
        group=os.environ.get("TELEGRAM_GROUP", ""),
        session_name=session_name,
        min_payout=float(os.environ.get("S19_MIN_PAYOUT", "0") or 0),
        min_forecast=float(os.environ.get("S19_MIN_FORECAST", "0") or 0),
        lead_s=int(os.environ.get("S19_LEAD_S", "5") or 5),
        trade=os.environ.get("S19_TRADE", "0").lower() in ("1", "true", "yes"),
    )


async def main():
    cfg = load_cfg_from_env()
    if not (cfg.api_id and cfg.api_hash and cfg.phone and cfg.group):
        print(
            "Missing TELEGRAM_* env vars. Need TELEGRAM_API_ID, "
            "TELEGRAM_API_HASH, TELEGRAM_PHONE, TELEGRAM_GROUP"
        )
        return
    follower = Strategy19Follower(cfg)
    await follower.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[S19] Stopped by user.")
