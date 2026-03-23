import os
import json
import threading
import requests as http_requests
import re

from datetime import datetime
from threading import Lock
from services.supabase_service import get_supabase_client
from scripts.hf_json_to_txt import convert_hf_json_to_txt
from scripts.ingest_to_supabase import ingest_single_file
from scripts.ingest_slack_to_supabase import ingest_slack_history
from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier
from dotenv import load_dotenv
from fastapi import FastAPI, Request, BackgroundTasks
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
# SLACK MESSAGE AUTO-STORE (Real-time RAG)
# ───────────────────────────────────────

def store_slack_message(channel_id: str, user_id: str, text: str):
    """
    Append a live incoming Slack message to today's daily chunk.
    If today's chunk doesn't exist yet, create it.
    This keeps live messages consistent with the daily-chunk structure
    created by ingest_slack_history.
    """
    from services.embedding_api import get_embedding
    from datetime import datetime, timezone

    if not text or len(text.strip()) < 10:
        return

    supabase = get_supabase_client()

    # ── Ensure user exists ──
    user = supabase.get_user_by_username(channel_id)
    if not user:
        user = supabase.create_user(channel_id)
    user_uuid = user["id"]

    # ── Ensure slack meeting record exists ──
    meeting_name = f"slack_{channel_id}"
    meeting = supabase.get_meeting_by_name(meeting_name)
    if not meeting:
        meeting = supabase.create_meeting({
            "meeting_name": meeting_name,
            "user_id": user_uuid,
            "channel_id": channel_id,
            "run_id": f"slack_{channel_id}",
            "status": "ingested",
        })
    meeting_id = meeting["id"]

    # ── Today's date key ──
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    formatted_line = f"[{user_id}]: {text}"

    try:
        # ── Look for today's existing chunk ──
        existing = supabase.client.table("chunks") \
            .select("id, chunk_index, chunk_text") \
            .eq("meeting_id", meeting_id) \
            .like("chunk_text", f"--- {today} ---%") \
            .limit(1) \
            .execute()

        if existing.data:
            # ── Append to today's chunk ──
            row = existing.data[0]
            updated_text = row["chunk_text"] + f"\n\n{formatted_line}"
            new_embedding = get_embedding(updated_text)

            supabase.client.table("chunks") \
                .update({
                    "chunk_text": updated_text,
                    "embedding": new_embedding,
                }) \
                .eq("id", row["id"]) \
                .execute()

            print(f"[SLACK STORE] Appended to {today} chunk for {channel_id}")

        else:
            # ── No chunk for today yet — create it ──
            # Get next chunk index
            all_chunks = supabase.client.table("chunks") \
                .select("chunk_index") \
                .eq("meeting_id", meeting_id) \
                .order("chunk_index", desc=True) \
                .limit(1) \
                .execute()
            next_index = (all_chunks.data[0]["chunk_index"] + 1) if all_chunks.data else 0

            chunk_text = f"--- {today} ---\n{formatted_line}"
            embedding = get_embedding(chunk_text)

            supabase.insert_chunks([{
                "meeting_id": meeting_id,
                "chunk_index": next_index,
                "chunk_text": chunk_text,
                "embedding": embedding,
                "source": "slack",
            }])

            print(f"[SLACK STORE] Created new {today} chunk for {channel_id}")

    except Exception as e:
        print(f"[SLACK STORE] Failed: {e}")



# ───────────────────────────────────────
# CONSTANTS / STATE
# ───────────────────────────────────────

SAFE_ABSTAIN = "This was not clearly discussed in the meeting."

SLACK_SESSIONS = {}  # channel_id → session info
PROCESSED_EVENTS = set()
EVENT_LOCK = Lock()

STAGE_ACTIVE = "ACTIVE"


def ingest_and_reply(channel_id, run_id, response_url):
    try:
        result = hf_service.fetch_result(run_id)

        if result["state"] != "completed":
            http_requests.post(response_url, json={
                "response_type": "in_channel",
                "text": "Pipeline not finished yet."
            })
            return

        data = result["data"]
        keyframes = data.get("keyframes") or data.get("output", {}).get("keyframes", [])

        if not keyframes:
            http_requests.post(response_url, json={
                "response_type": "in_channel",
                "text": "No keyframes found."
            })
            return

        lines = [
            f"[{f.get('timestamp')}] {f.get('combined_summary')}"
            for f in keyframes if f.get("combined_summary")
        ]

        full_text = "\n\n".join(lines)
        file_path = f"/tmp/meeting_{run_id}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        ingest_single_file(
            file_path=file_path,
            username=channel_id,
            run_id=run_id
        )

        http_requests.post(response_url, json={
            "response_type": "in_channel",
            "text": f"Ingestion complete for `{run_id}`."
        })

    except Exception as e:
        http_requests.post(response_url, json={
            "response_type": "in_channel",
            "text": f"Ingestion failed: {str(e)}"
        })

def check_state_and_reply(run_id, response_url):
    try:
        state = hf_service.check_status(run_id)
        status = state.get("status", "unknown")
        http_requests.post(response_url, json={
            "response_type": "in_channel",
            "text": f"Status for `{run_id}`: `{status}`"
        })
    except Exception as e:
        http_requests.post(response_url, json={
            "response_type": "in_channel",
            "text": f"Failed: {str(e)}"
        })

def start_and_reply(channel_id, video_url, response_url):
    try:
        run_id = hf_service.start_pipeline(video_url)

        supabase = get_supabase_client()
        user = supabase.get_user_by_username(channel_id)
        if not user:
            user = supabase.create_user(channel_id)

        supabase.create_meeting({
            "meeting_name": f"meeting_{run_id}",
            "user_id": user["id"],
            "run_id": run_id,
            "channel_id": channel_id,
            "status": "running"
        })

        http_requests.post(response_url, json={
            "response_type": "in_channel",
            "text": f"Meeting started! Run ID: `{run_id}`\nUse `/state {run_id}` to check."
        })

    except Exception as e:
        http_requests.post(response_url, json={
            "response_type": "in_channel",
            "text": f"Failed: {str(e)}"
        })

def sync_slack_and_reply(channel_id, response_url):
    """Background task: ingest Slack channel history and reply."""
    try:
        ingest_slack_history(channel_id=channel_id)
        http_requests.post(response_url, json={
            "response_type": "in_channel",
            "text": f"✅ Slack history synced for this channel."
        })
    except Exception as e:
        http_requests.post(response_url, json={
            "response_type": "in_channel",
            "text": f"❌ Slack sync failed: {str(e)}"
        })

# ───────────────────────────────────────
# SLACK EVENTS ENDPOINT
# ───────────────────────────────────────


@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    content_type = request.headers.get("content-type", "")


    # ===============================
    # 1️⃣ EVENTS API (JSON)
    if "application/json" in content_type:
        body = await request.json()

        if body.get("type") == "url_verification":
            return {"challenge": body.get("challenge")}

        if body.get("type") == "event_callback":
            event = body.get("event")
            background_tasks.add_task(process_event, event)

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
            response_url = form.get("response_url")

            background_tasks.add_task(
                start_and_reply,
                channel_id, video_url, response_url
            )

            return {
                "response_type": "in_channel",
                "text": "Pipeline starting... Run ID will appear shortly."
            }

        # ---------- /ingest ----------
        if command == "/ingest":
            run_id = (form.get("text") or "").replace("`", "").strip()

            if not run_id:
                return {
                    "response_type": "in_channel",
                    "text": "Please provide a run_id.\nUsage: `/ingest <run_id>`"
                }

            response_url = form.get("response_url")
            background_tasks.add_task(ingest_and_reply, channel_id, run_id, response_url)

            return {
                "response_type": "in_channel",
                "text": f"Starting ingestion for `{run_id}`..."
            }

        # ---------- /state ----------
        if command == "/state":
            run_id = (text or "").replace("`", "").strip()

            if not run_id:
                return {
                    "response_type": "in_channel",
                    "text": "Please provide a run_id."
                }

            response_url = form.get("response_url")
            background_tasks.add_task(check_state_and_reply, run_id, response_url)

            return {
                "response_type": "in_channel",
                "text": f"Checking status for `{run_id}`..."
            }

        # ---------- /sync_slack ----------
        if command == "/sync_slack":
            response_url = form.get("response_url")
            background_tasks.add_task(
                sync_slack_and_reply,
                channel_id, response_url
            )
            return {
                "response_type": "in_channel",
                "text": "⏳ Syncing Slack history for this channel..."
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
                text=f"Meeting `{run_id}` completed.\nYou can now run `/ingest`."
            )
            return

        logs = state.get("logs", "")

        slack_client.chat_postMessage(
            channel=channel_id,
            text=f"Status for `{run_id}`: {status}\n\nLast Logs:\n{logs[-1000:]}"
        )

    except Exception as e:
        slack_client.chat_postMessage(
            channel=channel_id,
            text=f"Failed to check status:\n{str(e)}"
        )

def ingest_background(channel_id, run_id):
    try:
        supabase = get_supabase_client()

        # 1️⃣ Fetch HF result
        result = hf_service.fetch_result(run_id)

        if result["state"] != "completed":
            slack_client.chat_postMessage(
                channel=channel_id,
                text="Pipeline not finished yet."
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
                text=f" Unexpected HF format.\nKeys: {list(data.keys())}"
            )
            return

        if not keyframes:
            slack_client.chat_postMessage(
                channel=channel_id,
                text=" Keyframes list is empty."
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
                text=" No combined summaries found."
            )
            return

        full_text = "\n\n".join(lines)

        # 3️⃣ Save transcript file
        meeting_name = f"meeting_{run_id}"
        file_path = f"{meeting_name}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        # 4️⃣ Ingest into Supabase
        ingest_single_file(
            file_path=file_path,
            username=channel_id,
            run_id=run_id
        )

        # 5️⃣ Update meeting status
        supabase.update_meeting_status(run_id, "ingested")

        slack_client.chat_postMessage(
            channel=channel_id,
            text=f"Ingestion complete for `{run_id}`."
        )

    except Exception as e:
        slack_client.chat_postMessage(
            channel=channel_id,
            text=f"Ingestion failed:\n{str(e)}"
        )

def process_event(event: dict):
    if not event:
        return

    # ── 1. Skip system subtypes ──
    if event.get("subtype") in [
        "bot_message", "message_changed", "message_deleted",
        "thread_broadcast", "channel_join", "channel_leave",
        "channel_purpose", "channel_topic",
    ]:
        return

    # ── 2. Skip system join/leave text messages ──
    text_raw = (event.get("text") or "").strip()
    if any(phrase in text_raw for phrase in [
        "has been added to",
        "joined the channel",
        "left the channel",
        "MMA AGENT has joined",
        "joined #",
    ]):
        return

    # ── 3. Store bot replies, skip bot commands ──
    if event.get("bot_id"):
        bot_text = (event.get("text") or "").strip()
        bot_channel = event.get("channel")
        bot_name = event.get("username") or "MMA AGENT"
        if bot_text and bot_channel:
            try:
                store_slack_message(bot_channel, bot_name, bot_text)
            except Exception as e:
                print(f"[SLACK STORE BOT] Failed: {e}")
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
        text = re.sub(r"<@[^>]+>", "", text).strip()
        if text == "":
            return

    if not slack_user_id or not channel_id:
        return

    # ── AUTO-STORE every incoming Slack message ──
    try:
        store_slack_message(channel_id, slack_user_id, text)
    except Exception as e:
        print(f"[SLACK STORE] Failed: {e}")

    # In public channels → only respond if bot is mentioned
    if event.get("channel_type") == "channel":
        if "<@" not in (event.get("text") or ""):
            return

    handle_user_message(slack_user_id, channel_id, text)

def start_meeting_background(channel_id, video_url):
    try:
        supabase = get_supabase_client()

        run_id = hf_service.start_pipeline(video_url)

        user = supabase.get_user_by_username(channel_id)
        if not user:
            user = supabase.create_user(channel_id)

        user_id = user["id"]

        meeting_name = f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        supabase.create_meeting({
            "meeting_name": meeting_name,
            "user_id": user_id,
            "run_id": run_id,
            "channel_id": channel_id,
            "status": "running"
        })

        slack_client.chat_postMessage(
            channel=channel_id,
            text=f"Meeting started!\nRun ID: `{run_id}`\nUse `/state` to check progress."
        )

    except Exception as e:
        slack_client.chat_postMessage(
            channel=channel_id,
            text=f"Failed to start meeting:\n{str(e)}"
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

        session = SLACK_SESSIONS[channel_id]  # use the newly created session
        send_message(channel_id, "🤖 New session started for this channel.") 

        # ── AUTO-INGEST Slack history if this channel has never been indexed ──
        try:
            if not supabase.get_slack_ingestion_status(channel_id):
                threading.Thread(
                    target=ingest_slack_history,
                    args=(channel_id,),
                    daemon=True
                ).start()
                print(f"[AUTO-INGEST] Started Slack history ingest for {channel_id}")
        except Exception as e:
            print(f"[AUTO-INGEST] Failed to start: {e}")

        # NOTE: No return here — fall through so the first message gets answered

    # ----------------------------------------
    # EXIT
    # ----------------------------------------
    if text.lower().strip() == "exit":
        del SLACK_SESSIONS[channel_id]
        send_message(channel_id, "Session ended.")
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

        send_message(channel_id, "Starting meeting pipeline...")

        run_id = hf_service.start_pipeline(video_url)

        session["current_run_id"] = run_id

        send_message(
            channel_id,
            f"Meeting started.\nRun ID: `{run_id}`\nUse `status` to check progress."
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
        send_message(channel_id, f"Status: {state}")
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

        if result["state"] == "running":
            send_message(channel_id, "Pipeline is still running...")
            return

        if result["state"] == "completed":
            send_message(channel_id, "Output received. Processing transcript...")

            txt_path = convert_hf_json_to_txt(
                result["data"],
                username=channel_id,
                meeting_name=f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            ingest_single_file(
                file_path=txt_path,
                username=channel_id,
                run_id=run_id
            )

            send_message(channel_id, "Meeting processed and stored successfully.")
            return

        send_message(channel_id, "Unexpected pipeline response.")
        return

    # ----------------------------------------
    # NORMAL QUESTION FLOW
    # ----------------------------------------
    try:
        initial_state = {
            "user_id": session["user_id"],
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
                "Sorry, I couldn't find a clear answer for that."
            )
            return

        send_message(channel_id, answer)

    except Exception as e:
        print(f"[HANDLE ERROR] {e}")
        send_message(
            channel_id,
            "Something went wrong while processing your question."
        )


# ───────────────────────────────────────
# SLACK SEND
# ───────────────────────────────────────

def send_message(channel_id: str, text: str):
    slack_client.chat_postMessage(channel=channel_id, text=text)

@app.get("/")
def health():
    return {"status": "meeting-agent-running"}