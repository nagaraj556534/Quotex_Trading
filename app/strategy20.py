import re
import unicodedata
import time
import asyncio
import os
import csv
from typing import Optional, List, Dict
from dataclasses import dataclass

# Try to import Telethon
try:
    from telethon import TelegramClient, events
except ImportError:
    TelegramClient = None
    events = None

# ---------------------------------------------------------------------------
# Config Data Class
# ---------------------------------------------------------------------------
@dataclass
class S20Config:
    api_id: int
    api_hash: str
    phone: str
    group: str
    session_name: str = "s20_session"
    loss_target: float = 0.0  # User provided stop loss
    stake_amount: float = 5.0 # Default or from config
    martingale_multiplier: float = 2.0 # Default per instructions
    martingale_steps: int = 3 # Default
    log_csv: str = os.path.join(os.path.dirname(__file__), "..", "strategy20_signals.csv")


# ---------------------------------------------------------------------------
# Parser Logic for "Mathematical Monospace" and Special Fonts
# ---------------------------------------------------------------------------
class TelegramSignalParserS20:
    """
    Parses signals with special unicode fonts like:
    💷 𝙽𝚉𝙳𝙹𝙿𝚈-𝙾𝚃𝙲𝚚
    ⏳  17:35
    ⌚️  𝙼𝟷
    🟢 𝙲𝙰𝙻𝙻
    """

    def normalize_text(self, text: str) -> str:
        """Converts mathematical monospace/bold unicode chars to standard ASCII."""
        out = []
        for char in text:
            # Check for Mathematical Alphanumeric Symbols (U+1D400 - U+1D7FF)
            # We can use unicodedata.normalize('NFKD', char) but it might not catch all math fonts perfectly 
            # without explicit ranges, but let's try standard normalization first.
            # Many of these "𝙽" are effectively "N" in compatibility decomposition.
            normalized = unicodedata.normalize('NFKD', char)
            if normalized:
                out.append(normalized)
            else:
                out.append(char)
        return "".join(out)

    def parse_signal(self, raw_text: str) -> Optional[Dict]:
        """
        Returns dict with {asset, direction, timeframe, time} or None.
        """
        # Strict Filtering: Check for required emojis
        # User required: 📊 (Asset), ⏰ (Time), ⌛️ (Duration), 🟢/🔴 (Direction)
        # Note: ⌛️ might be ⌛ in some clients, checking both or just base.
        # We check for presence of these characters in raw_text.
        
        has_asset_emoji = "📊" in raw_text
        has_time_emoji = "⏰" in raw_text
        has_dur_emoji = "⌛" in raw_text # Matches ⌛ and ⌛️
        has_dir_emoji = "🟢" in raw_text or "🔴" in raw_text
        
        if not (has_asset_emoji and has_time_emoji and has_dur_emoji and has_dir_emoji):
            # print(f"[S20] Ignored message (missing emojis): {raw_text[:20]}...")
            return None

        clean_text = self.normalize_text(raw_text)
        lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
        
        asset = None
        direction = None
        timeframe = None
        trade_time = None
        
        # Regex patterns for normalized text
        # Asset: 3 letters + 3 letters + optional OTC/Q/etc.
        # Example: NZDJPY-OTCQ -> NZDJPY_otc
        asset_pattern = re.compile(r"([A-Z]{3}[A-Z]{3}).*?(OTC)?", re.IGNORECASE)
        
        # Time: HH:MM
        time_pattern = re.compile(r"(\d{1,2}:\d{2})")
        
        # Timeframe: M1, 1M, etc.
        tf_pattern = re.compile(r"(?:M|MIN)\s*(\d+)|(\d+)\s*(?:M|MIN)", re.IGNORECASE)
        
        # Direction: CALL/PUT
        dir_pattern = re.compile(r"(CALL|PUT|UP|DOWN)", re.IGNORECASE)

        for line in lines:
            # Asset detection
            if not asset:
                # Look for currency pairs
                m_asset = asset_pattern.search(line)
                if m_asset:
                    # Check if it looks like a pair line (often has emojis like 💷 or 📊)
                    # We assume the first valid pair found is the asset
                    base = m_asset.group(1).upper()
                    is_otc = "OTC" in line.upper()
                    asset = f"{base}_otc" if is_otc else base
            
            # Time detection
            if not trade_time:
                m_time = time_pattern.search(line)
                if m_time:
                    trade_time = m_time.group(1)
            
            # Timeframe detection
            if not timeframe:
                m_tf = tf_pattern.search(line)
                if m_tf:
                    # Group 1 or 2 depending on order (M1 vs 1M)
                    val = m_tf.group(1) or m_tf.group(2)
                    try:
                        timeframe = int(val)
                    except:
                        pass
            
            # Direction detection
            if not direction:
                m_dir = dir_pattern.search(line)
                if m_dir:
                    d = m_dir.group(1).upper()
                    if d in ("CALL", "UP"):
                        direction = "call"
                    elif d in ("PUT", "DOWN"):
                        direction = "put"

        if asset and direction:
            # Default timeframe if missing
            if not timeframe:
                timeframe = 1 
            
            return {
                "asset": asset,
                "direction": direction,
                "timeframe": timeframe,
                "trade_time": trade_time,
                "raw": raw_text
            }
        return None

from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Strategy Follower Class
# ---------------------------------------------------------------------------
class Strategy20Follower:
    def __init__(self, cfg: S20Config, execute_cb):
        self.cfg = cfg
        self.execute_cb = execute_cb # async func(asset, direction, amount, expiry)
        self.parser = TelegramSignalParserS20()
        self.client = None
        self.current_loss = 0.0
        self.stop_trading = False
        self.processed_messages = set()

    async def start(self):
        if not TelegramClient:
            print("Telethon not installed.")
            return

        self.client = TelegramClient(self.cfg.session_name, self.cfg.api_id, self.cfg.api_hash)
        await self.client.start(phone=self.cfg.phone)
        
        # Fix: Convert group ID to int if it's a string number (e.g. "-100...")
        # Telethon needs int for IDs, string for usernames
        try:
            if isinstance(self.cfg.group, str) and (self.cfg.group.startswith("-") or self.cfg.group.isdigit()):
                self.cfg.group = int(self.cfg.group)
        except:
            pass

        # Fix: Populate entity cache by fetching dialogs
        # This resolves the "Cannot find any entity" error
        print("[S20] Fetching dialogs to resolve entities...")
        await self.client.get_dialogs()
        
        print(f"[S20] Listening to group: {self.cfg.group}")
        print(f"[S20] Stop Loss Target: {self.cfg.loss_target}")
        print(f"[S20] Stake: {self.cfg.stake_amount} | Martingale: x{self.cfg.martingale_multiplier}")

        @self.client.on(events.NewMessage(chats=self.cfg.group))
        async def handler(event):
            if self.stop_trading:
                return

            text = event.raw_text
            if not text:
                return
            
            # Deduplicate by message ID
            msg_id = getattr(event, 'id', 0)
            if msg_id in self.processed_messages:
                return
            self.processed_messages.add(msg_id)

            # Parse
            sig = self.parser.parse_signal(text)
            if sig:
                print(f"\n[S20] Signal Detected: {sig['asset']} {sig['direction']} {sig['timeframe']}m Time: {sig['trade_time']}")
                
                # Timing Logic
                wait_seconds = 0
                if sig['trade_time']:
                    try:
                        # Parse HH:MM
                        hh, mm = map(int, sig['trade_time'].split(':'))
                        now = datetime.now()
                        target_dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                        
                        # Handle day rollover? 
                        # Usually signals are for the current day. If target is way in the past (e.g. > 12 hours), maybe it's for tomorrow?
                        # But simpler assumption: if target < now, it's late. If target > now, wait.
                        # User said: "suppos time exipery anathukku appuram signal vanthathan imidiate ag trate edukkanum"
                        # (If signal comes after expiry time, take trade immediately).
                        # Wait, "time exipery anathukku appuram" -> "after time expiry". 
                        # Actually, usually "17:35" means "Start trade at 17:35".
                        # If now is 17:36, we are late. User said "imidiate ag trate edukkanum" (take immediate).
                        # So: If now < target, WAIT. If now >= target, EXECUTE IMMEDIATE.
                        
                        delta = (target_dt - now).total_seconds()
                        if delta > 0:
                            print(f"[S20] Scheduled for {sig['trade_time']} (Wait {delta:.1f}s)...")
                            wait_seconds = delta
                        else:
                            print(f"[S20] Signal time {sig['trade_time']} passed. Executing immediately.")
                            
                    except Exception as e:
                        print(f"[S20] Time parse error: {e}. Executing immediately.")

                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)

                # Execute Trade
                try:
                    # Cross-Signal Recovery: Add accumulated loss to base stake
                    current_stake = self.cfg.stake_amount
                    if self.current_loss > 0:
                        current_stake += self.current_loss
                        print(f"[S20] Recovery Mode: Base {self.cfg.stake_amount} + Loss {self.current_loss:.2f} = Stake {current_stake:.2f}")

                    pnl = await self.execute_cb(
                        asset=sig['asset'],
                        direction=sig['direction'],
                        timeframe=sig['timeframe'],
                        stake=current_stake,
                        martingale_mult=self.cfg.martingale_multiplier,
                        martingale_steps=self.cfg.martingale_steps
                    )
                    
                    if pnl < 0:
                        self.current_loss += abs(pnl)
                    elif pnl > 0:
                        self.current_loss -= pnl 
                        if self.current_loss < 0:
                            self.current_loss = 0 

                    print(f"[S20] Session Net Loss: {self.current_loss:.2f} / Target: {self.cfg.loss_target}")

                    if self.current_loss >= self.cfg.loss_target:
                        print(f"🔴 [S20] LOSS TARGET HIT ({self.current_loss} >= {self.cfg.loss_target}). STOPPING TRADING.")
                        self.stop_trading = True
                        await self.client.disconnect()
                        
                except Exception as e:
                    print(f"[S20] Execution Error: {e}")

        # Auto-Reconnect Loop
        print("[S20] Starting event loop with auto-reconnect...")
        while True:
            try:
                if not self.client.is_connected():
                    print("[S20] Connecting to Telegram...")
                    await self.client.connect()
                
                await self.client.run_until_disconnected()
                
                # If we are here, it means we disconnected.
                if self.stop_trading:
                    print("[S20] Trading stopped cleanly.")
                    break
                
                print("[S20] Disconnected unexpectedly. Reconnecting in 5s...")
                await asyncio.sleep(5)

            except Exception as e:
                print(f"[S20] Connection Error: {e}")
                print("[S20] Reconnecting in 5s...")
                await asyncio.sleep(5)
