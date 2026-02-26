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

    # ===============================
    # 1️⃣ EVENTS API (JSON)
    # ===============================
    if "application/json" in content_type:

        body = await request.json()

        # Slack URL verification
        if body.get("type") == "url_verification":
            return {"challenge": body.get("challenge")}

        # Handle events
        if body.get("type") == "event_callback":
            event = body.get("event")
            process_event(event)

        return {"status": "ok"}

    # ===============================
    # 2️⃣ SLASH COMMANDS
    # ===============================
    if "application/x-www-form-urlencoded" in content_type:

        form = await request.form()

        command = form.get("command")
        text = form.get("text")
        channel_id = form.get("channel_id")

        # ---------- /new_meeting ----------
        if command == "/new_meeting":

            if not text:
                return {"text": "Please provide a video URL."}

            video_url = text.strip()

            threading.Thread(
                target=start_meeting_background,
                args=(channel_id, video_url),
                daemon=True
            ).start()

            return {
                "response_type": "in_channel",
                "text": "🚀 Starting meeting pipeline..."
            }

        # ---------- /ingest ----------
# ---------- /ingest ----------
        if command == "/ingest":

            text = (form.get("text") or "").strip()

            if not text:
                return {
                    "response_type": "in_channel",
                    "text": "❌ Please provide a run_id.\nUsage: `/ingest <run_id>`"
                }

            # 🔥 REMOVE BACKTICKS + SPACES
            run_id = text.replace("`", "").strip()

            print("CLEAN INGEST RUN ID:", run_id)

            threading.Thread(
                target=ingest_background,
                args=(channel_id, run_id),
                daemon=True
            ).start()

            return {
                "response_type": "in_channel",
                "text": f"📥 Starting ingestion for `{run_id}`..."
            }
        # ---------- /state ----------
        if command == "/state":

            supabase = get_supabase_client()

            meeting = supabase.get_latest_meeting_by_channel(channel_id)

            if not meeting:
                return {
                    "response_type": "in_channel",
                    "text": "❌ No meeting found in this channel."
                }

            run_id = text.replace("`", "").strip()

            print("STATE RUN ID:", run_id)

            # 🔥 Run status check in background thread
            threading.Thread(
                target=check_status_background,
                args=(channel_id, run_id),
                daemon=True
            ).start()

            # 🔥 Immediate response (must be under 3 seconds)
            return {
                "response_type": "in_channel",
                "text": f"🔎 Checking status for `{run_id}`..."
            }
# ───────────────────────────────────────
# EVENT PROCESSOR
# ───────────────────────────────────────
def check_status_background(channel_id, run_id):
    try:
        supabase = get_supabase_client()

        state = hf_service.check_status(run_id)
        status = state.get("status", "unknown")

        if status == "completed":
            supabase.update_meeting_status(run_id, "completed")

            slack_client.chat_postMessage(
                channel=channel_id,
                text=f"✅ Meeting `{run_id}` completed.\nYou can now run `/ingest`."
            )
            return

        logs = state.get("logs", "")

        slack_client.chat_postMessage(
            channel=channel_id,
            text=f"📊 Status for `{run_id}`: {status}\n\nLast Logs:\n{logs[-1000:]}"
        )

    except Exception as e:
        slack_client.chat_postMessage(
            channel=channel_id,
            text=f"❌ Failed to check status:\n{str(e)}"
        )

def ingest_background(channel_id, run_id):
    try:
        supabase = get_supabase_client()

        # 1️⃣ Fetch HF result
        result = hf_service.fetch_result(run_id)

        if result["state"] != "completed":
            slack_client.chat_postMessage(
                channel=channel_id,
                text="⏳ Pipeline not finished yet."
            )
            return

        data = result["data"]

        # 🔥 SAFELY EXTRACT KEYFRAMES
        if "keyframes" in data:
            keyframes = data["keyframes"]
        elif "output" in data and "keyframes" in data["output"]:
            keyframes = data["output"]["keyframes"]
        else:
            slack_client.chat_postMessage(
                channel=channel_id,
                text=f"❌ Unexpected HF format.\nKeys: {list(data.keys())}"
            )
            return

        if not keyframes:
            slack_client.chat_postMessage(
                channel=channel_id,
                text="❌ Keyframes list is empty."
            )
            return

        # 2️⃣ Extract ONLY combined_summary
        lines = []

        for frame in keyframes:
            summary = frame.get("combined_summary")
            timestamp = frame.get("timestamp")

            if summary:
                lines.append(f"[{timestamp}] {summary}")

        if not lines:
            slack_client.chat_postMessage(
                channel=channel_id,
                text="❌ No combined summaries found."
            )
            return

        full_text = "\n\n".join(lines)

        # 3️⃣ Save transcript file
        meeting_name = f"meeting_{run_id}"
        file_path = f"{meeting_name}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        # 4️⃣ 🔥 Ingest into Supabase (CORRECT CALL)
        ingest_single_file(
            file_path=file_path,
            username=channel_id,   # Slack channel = user
            run_id=run_id          # Unique meeting id
        )

        # 5️⃣ Update meeting status
        supabase.update_meeting_status(run_id, "ingested")

        slack_client.chat_postMessage(
            channel=channel_id,
            text=f"✅ Ingestion complete for `{run_id}`."
        )

    except Exception as e:
        slack_client.chat_postMessage(
            channel=channel_id,
            text=f"❌ Ingestion failed:\n{str(e)}"
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
# In public channels, respond only if bot is mentioned
    if event.get("channel_type") == "channel":
        if "<@" not in (event.get("text") or ""):
            return
    handle_user_message(slack_user_id, channel_id, text)

def start_meeting_background(channel_id, video_url):
    try:
        supabase = get_supabase_client()

        run_id = hf_service.start_pipeline(video_url)

        # Get user
        user = supabase.get_user_by_username(channel_id)
        if not user:
            user = supabase.create_user(channel_id)

        user_id = user["id"]

        meeting_name = f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 🔥 Store in DB
        supabase.create_meeting({
            "meeting_name": meeting_name,
            "user_id": user_id,
            "run_id": run_id,
            "channel_id": channel_id,
            "status": "running"
        })

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

        send_message(channel_id, "🆕 New session started for this channel.")

        return  # 🔥 STOP HERE — do not continue to Q&A

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

        # CASE 1: Still Running
        if result["state"] == "running":
            send_message(channel_id, "⏳ Pipeline is still running...")
            return

        # CASE 2: Completed
        if result["state"] == "completed":
            send_message(channel_id, "📥 Output received. Processing transcript...")

            txt_path = convert_hf_json_to_txt(
                result["data"],
                username=channel_id,
                meeting_name=f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            ingest_single_file(txt_path)

            send_message(channel_id, "✅ Meeting processed and stored successfully.")
            return

        send_message(channel_id, "⚠️ Unexpected pipeline response.")
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
