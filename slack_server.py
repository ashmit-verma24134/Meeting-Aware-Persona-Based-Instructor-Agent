import os
import json
import threading
import re
from datetime import datetime
from threading import Lock
from services.supabase_service import get_supabase_client
from scripts.hf_json_to_txt import convert_hf_json_to_txt
from scripts.ingest_to_supabase import ingest_single_file
from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier
from dotenv import load_dotenv
from fastapi import FastAPI, Request

from graphs.meeting_graph import meeting_graph
from services.hf_api_service import HFAPIService


from scripts.ingest_to_supabase import ingest_single_file


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
hf_service = HFAPIService()



# ───────────────────────────────────────
# CONSTANTS / STATE
# ───────────────────────────────────────

SAFE_ABSTAIN = "This was not clearly discussed in the meeting."

SLACK_SESSIONS = {}  # channel_id → session info]
PROCESSED_EVENTS = set()
EVENT_LOCK = Lock()

STAGE_ACTIVE = "ACTIVE"


# ───────────────────────────────────────
# SLACK EVENTS ENDPOINT
# ───────────────────────────────────────

@app.post("/slack/events")
async def slack_events(request: Request):

    content_type = request.headers.get("content-type", "")

    # ======================================================
    # 1️⃣ HANDLE SLASH COMMANDS
    # ======================================================
    if "application/x-www-form-urlencoded" in content_type:

        form = await request.form()
        command = form.get("command")
        text = form.get("text")
        channel_id = form.get("channel_id")

        # ----------------------------------
        # /new_meeting
        # ----------------------------------
        if command == "/new_meeting":

            if not text:
                return {"text": "Please provide a video URL."}

            video_url = text.strip()

            # 🔥 Start background thread
            threading.Thread(
                target=start_meeting_background,
                args=(channel_id, video_url),
                daemon=True
            ).start()

            # ⚡ Immediate response (avoid Slack timeout)
            return {
                "response_type": "in_channel",
                "text": "🚀 Starting meeting pipeline... please wait."
            }

        # ----------------------------------
        # /state
        # ----------------------------------
 # ----------------------------------
# /state
# ----------------------------------
        if command == "/state":

            run_id = text.strip() if text else None

            # If run_id provided manually
            if run_id:
                threading.Thread(
                    target=check_status_background,
                    args=(channel_id, run_id),
                    daemon=True
                ).start()

                return {
                    "response_type": "in_channel",
                    "text": f"🔎 Checking status for `{run_id}`..."
                }

            # Otherwise use stored one (if exists)
            session = SLACK_SESSIONS.get(channel_id)

            if not session or "current_run_id" not in session:
                return {
                    "response_type": "in_channel",
                    "text": "❌ No meeting running in this channel."
                }

            run_id = session["current_run_id"]

            threading.Thread(
                target=check_status_background,
                args=(channel_id, run_id),
                daemon=True
            ).start()

            return {
                "response_type": "in_channel",
                "text": f"🔎 Checking status for `{run_id}`..."
            }

# ───────────────────────────────────────
# EVENT PROCESSOR
# ───────────────────────────────────────
def check_status_background(channel_id, run_id):
    try:
        status = hf_service.check_status(run_id)

        slack_client.chat_postMessage(
            channel=channel_id,
            text=f"📊 Status for `{run_id}`: {status}"
        )

    except Exception as e:
        slack_client.chat_postMessage(
            channel=channel_id,
            text=f"❌ Failed to check status:\n{str(e)}"
        )

        
def process_event(event: dict):
    if not event:
        return

    # Ignore bot messages
    # Ignore bot messages and message edits/deletes
    if event.get("bot_id"):
        return

    if event.get("subtype") in [
        "bot_message",
        "message_changed",
        "message_deleted",
        "thread_broadcast"
    ]:
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
    # Ignore blank messages
    if not text:
        return


    # Remove bot mention formatting
    if event_type == "app_mention":
        # Remove bot mention
        text = re.sub(r"<@[^>]+>", "", text).strip()

        # 🔥 Ignore pure mention (no command)
        if text == "":
            return


    if not slack_user_id or not channel_id:
        return

    # In public channels → only respond if bot is mentioned
    if event.get("channel_type") == "channel":
        if event_type != "app_mention":
            return

    handle_user_message(slack_user_id, channel_id, text)

def start_meeting_background(channel_id, video_url):
    try:
        run_id = hf_service.start_pipeline(video_url)

        # Save run_id per channel
        session = SLACK_SESSIONS.get(channel_id, {})
        run_ids = session.get("run_ids", [])
        run_ids.append(run_id)

        SLACK_SESSIONS[channel_id] = {
            "run_ids": run_ids
        }

        slack_client.chat_postMessage(
            channel=channel_id,
            text=f"✅ Meeting started!\nRun ID: `{run_id}`\nUse `/state` to check progress."
        )

    except Exception as e:
        slack_client.chat_postMessage(
            channel=channel_id,
            text=f"❌ Failed to start meeting:\n{str(e)}"
        )
# ───────────────────────────────────────
# USER HANDLER
# ───────────────────────────────────────

def handle_user_message(slack_user_id: str, channel_id: str, text: str):

    supabase = get_supabase_client()

    # ----------------------------------------
    # Ensure Channel-Based User Exists
    # ----------------------------------------
    user = supabase.get_user_by_username(channel_id)

    if not user:
        user = supabase.create_user(channel_id)

    user_id = user["id"]

    session = SLACK_SESSIONS.get(channel_id)

    # ----------------------------------------
    # CREATE SESSION IF NOT EXISTS (Per Channel)
    # ----------------------------------------
    if session is None:
        sid = f"{channel_id}_slack_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        supabase.create_session(sid, user_id)

        SLACK_SESSIONS[channel_id] = {
            "user_id": user_id,
            "session_id": sid,
        }

        send_message(channel_id, "Channel session started ✅")
        return

    # ----------------------------------------
    # EXIT
    # ----------------------------------------
    if text.lower().strip() == "exit":
        del SLACK_SESSIONS[channel_id]
        send_message(channel_id, "Session ended ✅")
        return

    # ----------------------------------------
    # NEW MEETING
    # ----------------------------------------
    if text.startswith("new_meeting"):

        parts = text.split()

        if len(parts) < 2:
            send_message(channel_id, "Please provide a video URL.")
            return

        video_url = parts[1]

        send_message(channel_id, "🚀 Starting meeting pipeline...")

        run_id = hf_service.start_pipeline(video_url)

        session["current_run_id"] = run_id

        send_message(
            channel_id,
            f"✅ Meeting started.\nRun ID: `{run_id}`\nUse `status` to check progress."
        )
        return

    # ----------------------------------------
    # STATUS
    # ----------------------------------------
    if text.startswith("status"):

        run_id = session.get("current_run_id")

        if not run_id:
            send_message(channel_id, "No meeting has been started yet.")
            return

        state = hf_service.check_status(run_id)
        send_message(channel_id, f"📊 Status: {state}")
        return

    # ----------------------------------------
    # PROCESS
    # ----------------------------------------
    if text.startswith("process"):

        run_id = session.get("current_run_id")

        if not run_id:
            send_message(channel_id, "No meeting has been started yet.")
            return

        result = hf_service.fetch_result(run_id)

        if not result or result.get("status") != "complete":
            send_message(channel_id, "⏳ Pipeline still running...")
            return

        send_message(channel_id, "📥 Processing transcript...")

        txt_path = convert_hf_json_to_txt(
            result,
            username=channel_id,   # 🔥 Channel used as username
            meeting_name=f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        ingest_single_file(txt_path)

        send_message(channel_id, "✅ Meeting processed and stored successfully.")
        return

    # ----------------------------------------
    # NORMAL QUESTION FLOW
    # ----------------------------------------
    try:
        initial_state = {
            "user_id": session["user_id"],   # 🔥 Channel-based user
            "session_id": session["session_id"],
            "question": text,
            "decision": None,
            "standalone_query": text,
            "confidence": None,
            "temporal_constraint": None,
            "domain_constraint": None,
            "retrieved_chunks": [],
            "meeting_indices": None,
            "_all_meeting_indices": None,
            "question_intent": None,
            "time_scope": None,
            "candidate_answer": None,
            "final_answer": None,
            "method": "",
            "context_extended": False,
            "path": [],
        }

        result = meeting_graph.invoke(initial_state)
        answer = result.get("final_answer")

        if not answer or answer.strip() == SAFE_ABSTAIN:
            send_message(
                channel_id,
                "Sorry 😅 I couldn’t find a clear answer in the meeting transcript."
            )
            return

        send_message(channel_id, answer)

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
