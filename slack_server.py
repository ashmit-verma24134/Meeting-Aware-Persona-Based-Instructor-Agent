import os
import json
import threading
import re
from datetime import datetime
from threading import Lock

from fastapi import FastAPI, Request
from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier
from dotenv import load_dotenv

from agents.supervisor_agent import supervisor


# ───────────────────────────────────────
# ENV + SLACK SETUP
# ───────────────────────────────────────

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")

if not SLACK_BOT_TOKEN or not SLACK_SIGNING_SECRET:
    raise RuntimeError("Missing Slack environment variables")

slack_client = WebClient(token=SLACK_BOT_TOKEN)
signature_verifier = SignatureVerifier(SLACK_SIGNING_SECRET)

app = FastAPI()


# ───────────────────────────────────────
# CONSTANTS / STATE
# ───────────────────────────────────────

SAFE_ABSTAIN = "This was not clearly discussed in the meeting."

SLACK_SESSIONS = {}          # slack_user_id → session info
PROCESSED_EVENTS = set()
EVENT_LOCK = Lock()

STAGE_AWAITING_USER_ID = "AWAITING_USER_ID"
STAGE_ACTIVE = "ACTIVE"


# ───────────────────────────────────────
# SLACK EVENTS ENDPOINT
# ───────────────────────────────────────

@app.post("/slack/events")
async def slack_events(request: Request):
    body = await request.body()
    payload = json.loads(body)

    # Slack URL verification
    if payload.get("type") == "url_verification":
        return {"challenge": payload["challenge"]}

    # Verify signature
    if not signature_verifier.is_valid_request(body, request.headers):
        return {"error": "invalid signature"}

    event = payload.get("event", {})

    # Process async (avoid 3 sec Slack timeout)
    threading.Thread(
        target=process_event,
        args=(event,),
        daemon=True
    ).start()

    return {"ok": True}


# ───────────────────────────────────────
# EVENT PROCESSOR
# ───────────────────────────────────────

def process_event(event: dict):
    if not event:
        return

    # Ignore bot messages
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return

    event_id = event.get("event_ts")
    if not event_id:
        return

    with EVENT_LOCK:
        if event_id in PROCESSED_EVENTS:
            return
        PROCESSED_EVENTS.add(event_id)
        if len(PROCESSED_EVENTS) > 10_000:
            PROCESSED_EVENTS.clear()

    event_type = event.get("type")
    if event_type not in ["message", "app_mention"]:
        return

    slack_user_id = event.get("user")
    channel_id = event.get("channel")
    text = (event.get("text") or "").strip()

    # Remove bot mention formatting
    if event_type == "app_mention":
        text = re.sub(r"<@[^>]+>", "", text).strip()

    if not slack_user_id or not channel_id:
        return

    # In public channels: only respond if mentioned OR session exists
    if event_type == "message" and event.get("channel_type") == "channel":
        if "<@" not in event.get("text", "") and slack_user_id not in SLACK_SESSIONS:
            return

    handle_user_message(slack_user_id, channel_id, text)


# ───────────────────────────────────────
# USER HANDLER
# ───────────────────────────────────────

def handle_user_message(slack_user_id: str, channel_id: str, text: str):
    session = SLACK_SESSIONS.get(slack_user_id)

    # First interaction → ask for user_id
    if session is None:
        SLACK_SESSIONS[slack_user_id] = {"stage": STAGE_AWAITING_USER_ID}
        send_message(channel_id, "Hello 👋\nPlease enter your *user_id* to begin.")
        return

    # Awaiting user_id
    if session["stage"] == STAGE_AWAITING_USER_ID:
        user_id = text.strip()
        start = datetime.now()
        sid = f"{user_id}_slack_{start.strftime('%Y-%m-%d_%H-%M-%S')}"

        SLACK_SESSIONS[slack_user_id] = {
            "stage": STAGE_ACTIVE,
            "user_id": user_id,
            "session_id": sid,
            "session_start": start
        }

        send_message(
            channel_id,
            f"Session started for *{user_id}* ✅\n"
            f"Ask your questions.\n"
            f"Type *exit* to end."
        )
        return

    # Active session
    if session["stage"] == STAGE_ACTIVE:

        if text.lower().strip() == "exit":
            del SLACK_SESSIONS[slack_user_id]
            send_message(channel_id, "Session ended ✅")
            return

        send_message(channel_id, "⏳ Checking meeting notes...")

        try:
            result = supervisor(
                user_id=session["user_id"],
                session_id=session["session_id"],
                question=text
            )

            answer = result.get("answer") if isinstance(result, dict) else None

            if not answer or answer.strip() == SAFE_ABSTAIN:
                send_message(
                    channel_id,
                    "Sorry 😅 I couldn’t find a clear answer in the meeting transcript."
                )
                return

            # Terminal-style formatting
            formatted = (
                f"{session['user_id']}@meeting-bot:~$ {text}\n\n"
                f"{answer}"
            )

            send_message(channel_id, formatted)

        except Exception:
            send_message(
                channel_id,
                "⚠️ Something went wrong while processing your question."
            )


# ───────────────────────────────────────
# SLACK SEND
# ───────────────────────────────────────

def send_message(channel_id: str, text: str):
    slack_client.chat_postMessage(channel=channel_id, text=text)
