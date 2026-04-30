"""
heartbeat_cron.py

Calls the /heartbeat endpoint every 15 minutes for ALL channels.
Run this on any machine:
    python heartbeat_cron.py
"""

import time
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# ── Config ──
VERCEL_URL = os.getenv("VERCEL_URL", "https://meeting-aware-persona-based-instructor-agent-k9bv-2w5v247fm.vercel.app")
BYPASS_TOKEN = os.getenv("VERCEL_AUTOMATION_BYPASS_SECRET")
INTERVAL_MINUTES = 15  # every 15 minutes

def trigger_heartbeat():
    try:
        response = requests.post(
            f"{VERCEL_URL}/heartbeat",
            json={},  # no channel_id needed — server handles all channels
            headers={"x-vercel-protection-bypass": BYPASS_TOKEN} if BYPASS_TOKEN else {},
            timeout=10
        )
        print(f"[CRON] Heartbeat triggered → {response.status_code} {response.json()}")
    except Exception as e:
        print(f"[CRON] Heartbeat failed: {e}")

if __name__ == "__main__":
    print(f"[CRON] Starting heartbeat cron — every {INTERVAL_MINUTES} min for ALL channels")
    while True:
        trigger_heartbeat()
        time.sleep(INTERVAL_MINUTES * 60)