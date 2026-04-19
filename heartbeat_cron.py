"""
heartbeat_cron.py

Calls the /heartbeat endpoint every 5 minutes.
Run this on any machine:
    python heartbeat_cron.py

Later: change INTERVAL_MINUTES to 60 or 120 for production.
"""

import time
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# ── Config ──
VERCEL_URL = os.getenv("VERCEL_URL", "https://your-app.vercel.app")
CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")
BYPASS_TOKEN = os.getenv("VERCEL_AUTOMATION_BYPASS_SECRET")  # add this to .env
INTERVAL_MINUTES = 1  # change to 60 or 120 for production

def trigger_heartbeat():
    try:
        response = requests.post(
            f"{VERCEL_URL}/heartbeat",
            json={"channel_id": CHANNEL_ID},
            headers={"x-vercel-protection-bypass": BYPASS_TOKEN} if BYPASS_TOKEN else {},
            timeout=10
        )
        print(f"[CRON] Heartbeat triggered → {response.status_code} {response.json()}")
    except Exception as e:
        print(f"[CRON] Heartbeat failed: {e}")

if __name__ == "__main__":
    print(f"[CRON] Starting heartbeat cron — every {INTERVAL_MINUTES} min for channel {CHANNEL_ID}")
    while True:
        trigger_heartbeat()
        time.sleep(INTERVAL_MINUTES * 60)