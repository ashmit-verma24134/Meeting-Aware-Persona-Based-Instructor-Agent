import os
import threading
from datetime import datetime
from fastapi import FastAPI, Request
from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier
from dotenv import load_dotenv

from agents.supervisor_agent import supervisor
from memory.session_persistence import save_session
from memory.session_memory import session_memory
from fastapi import BackgroundTasks

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")

if not SLACK_BOT_TOKEN or not SLACK_SIGNING_SECRET:
    raise RuntimeError("Missing Slack environment variables")

slack_client = WebClient(token=SLACK_BOT_TOKEN)
signature_verifier = SignatureVerifier(SLACK_SIGNING_SECRET)

app = FastAPI()

SAFE_ABSTAIN = "This was not clearly discussed in the meeting."


SLACK_SESSIONS = {}      # slack_user_id → session
PROCESSED_EVENTS = set() # event_ts dedup

STAGE_AWAITING_USER_ID = "AWAITING_USER_ID"
STAGE_ACTIVE = "ACTIVE"

@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()

    if not signature_verifier.is_valid_request(body, request.headers):
        return {"error": "invalid signature"}

    payload = await request.json()

    if payload.get("type") == "url_verification":
        return {"challenge": payload["challenge"]}

    if payload.get("type") != "event_callback":
        return {"ok": True}

    event = payload.get("event", {})
    if not event or event.get("bot_id"):
        return {"ok": True}

    background_tasks.add_task(process_event, event)
    return {"ok": True}




def process_event(event: dict):
    print("\n=== EVENT RECEIVED ===")
    print(event)
    event_id = event.get("event_ts")
    if not event_id or event_id in PROCESSED_EVENTS:
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

    # If channel mention, remove bot mention
    if event.get("type") == "app_mention":
        import re
        text = re.sub(r"<@[^>]+>", "", text).strip()



    if not slack_user_id or not channel_id:
        return

    # Ignore normal channel messages unless user already has a session
    if event_type == "message" and event.get("channel_type") == "channel":
        # allow if this message mentions the bot
        if "<@" in event.get("text", ""):
            pass
        elif slack_user_id not in SLACK_SESSIONS:
            return


    print("PASSING TO HANDLER:", {
    "event_type": event_type,
    "user": slack_user_id,
    "channel": channel_id,
    "text": text,
    "has_session": slack_user_id in SLACK_SESSIONS
    })



    handle_user_message(slack_user_id, channel_id, text)



def handle_user_message(slack_user_id: str, channel_id: str, text: str):
    session = SLACK_SESSIONS.get(slack_user_id)

    if session is None:
        SLACK_SESSIONS[slack_user_id] = {
            "stage": STAGE_AWAITING_USER_ID
        }
        send_message(channel_id, "Hello 👋\nPlease enter your *user_id* to begin.")
        return

    if session["stage"] == STAGE_AWAITING_USER_ID:
        user_id = text
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
            f"Session started for *{user_id}* \n"
            f"Ask your questions.\n"
            f"Type *exit* to save and end."
        )
        return


    if session["stage"] == STAGE_ACTIVE:
        user_id = session["user_id"]
        session_id = session["session_id"]

        if text.lower().strip() == "exit":

            finalize_session(slack_user_id, channel_id)
            return

        # EXACTLY SAME AS CLI
        result = supervisor(
            user_id=user_id,
            session_id=session_id,
            question=text
        )

        answer = result.get("answer")

        # Polite Slack decline (CLI prints SAFE_ABSTAIN, Slack phrases it)
        if not answer or answer.strip() == SAFE_ABSTAIN:
            send_message(
                channel_id,
                "Sorry 😅 I checked the meeting transcript, but this wasn’t clearly discussed."
            )
            return

        send_message(channel_id, answer)



def finalize_session(slack_user_id: str, channel_id: str):
    session = SLACK_SESSIONS.get(slack_user_id)
    if not session:
        return

    user_id = session["user_id"]
    session_id = session["session_id"]
    start = session["session_start"]
    end = datetime.now()

    history = session_memory.get_recent_context(session_id, k=10_000)
    conversation = [
        {
            "question": t.get("question"),
            "answer": t.get("answer"),
            "timestamp": t.get("timestamp")
        }
        for t in history
    ]

    save_session(
        session_id,
        {
            "user_id": user_id,
            "meeting_name": "slack_session",
            "session_start_time": start.isoformat(),
            "session_end_time": end.isoformat(),
            "conversation": conversation
        }
    )

    del SLACK_SESSIONS[slack_user_id]

    send_message(channel_id, "Session ended and saved successfully")



def send_message(channel_id: str, text: str):
    slack_client.chat_postMessage(
        channel=channel_id,
        text=text
    )
