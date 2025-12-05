
import asyncio
import os
import sys
from pyquotex.stable_api import Quotex

async def test_live_profit():
    # Use environment variables or hardcoded for test
    email = os.environ.get("QX_EMAIL") or "test@example.com"
    password = os.environ.get("QX_PASSWORD") or "password"
    
    # Try to load from session first if possible, or just connect
    # For this test, we assume user has valid session or env vars. 
    # But since I can't interact, I'll rely on the existing session.json if present?
    # Or better, I'll use the user's config if I can import it.
    
    # Let's just try to connect using the same logic as main.py
    try:
        from pyquotex.config import credentials
        email, password = credentials()
    except:
        pass
        
    if not email or not password:
        print("No credentials found for test.")
        return

    qx = Quotex(email=email, password=password)
    await qx.connect()
    print("Connected.")
    
    # Place a demo trade
    # Find an open asset first
    print("Checking available assets...")
    asset = None
    all_assets = await qx.get_all_assets()
    for a in all_assets:
        if "_otc" in a: # Prefer OTC for testing as they are usually open
            _, open_info = await qx.check_asset_open(a)
            if open_info[2]: # boolean for is_open
                asset = a
                break
    
    if not asset:
        print("No open OTC assets found. Trying any open asset...")
        for a in all_assets:
             _, open_info = await qx.check_asset_open(a)
             if open_info[2]:
                asset = a
                break
    
    if not asset:
        print("No open assets found at all.")
        await qx.close()
        return

    print(f"Selected asset: {asset}")

    try:
        qx.set_account_mode("PRACTICE")
        print(f"Placing demo trade on {asset}...")
        status, buy_info = await qx.buy(100, asset, "call", 60)
        if status:
            print(f"Trade placed: {buy_info}")
            buy_id = buy_info['id']
            
            # Monitor profit for 30 seconds
            print("Monitoring profit...")
            for i in range(30):
                profit = qx.get_profit()
                profit_op = getattr(qx.api, 'profit_in_operation', None)
                print(f"Time {i}: get_profit={profit}, profit_in_op={profit_op}")
                await asyncio.sleep(1)
        else:
            print(f"Trade failed. Reason: {getattr(qx, 'last_exec_error_reason', 'Unknown')}")
            # Also print buy_info if available (it's the second return value)
            print(f"Buy Info/Error: {buy_info}")
    finally:
        await qx.close()

if __name__ == "__main__":
    asyncio.run(test_live_profit())
