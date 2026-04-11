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
# CONSTANTS / STATE
# ───────────────────────────────────────

SAFE_ABSTAIN = "This was not clearly discussed in the meeting."

SLACK_SESSIONS = {}  # channel_id → session info]
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

        # 4️⃣Ingest into Supabase (CORRECT CALL)
        ingest_single_file(
            file_path=file_path,
            username=channel_id,   # Slack channel = user
            run_id=run_id          # Unique meeting id
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

        # Ignore pure mention (no command)
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

        # Store in DB
        supabase.create_meeting({
            "meeting_name": meeting_name,
            "user_id": user_id,
            "run_id": run_id,
            "channel_id": channel_id,
            "status": "running"
        })

        slack_client.chat_postMessage(
            channel=channel_id,
            text=f" Meeting started!\nRun ID: `{run_id}`\nUse `/state` to check progress."
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
    # ----------------------------------------
    # EXIT
    # ----------------------------------------
    if text.lower().strip() == "exit":
        del SLACK_SESSIONS[channel_id]
        send_message(channel_id, "Session ended")
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

        # CASE 1: Still Running
        if result["state"] == "running":
            send_message(channel_id, "Pipeline is still running...")
            return

        # CASE 2: Completed
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

            send_message(channel_id, " Meeting processed and stored successfully.")
            return

        send_message(channel_id, "Unexpected pipeline response.")
        return

    # ----------------------------------------
    # NORMAL QUESTION FLOW
    # ----------------------------------------
    try:
        initial_state = {
            "user_id": session["user_id"],   #  Channel-based user
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
                "Sorry  I couldn’t find a clear answer in the meeting transcript."
            )
            return

        send_message(channel_id, answer)

    except Exception:
        send_message(
            channel_id,
            " Something went wrong while processing your question."
        )
# ───────────────────────────────────────
# SLACK SEND
# ───────────────────────────────────────

def send_message(channel_id: str, text: str):
    slack_client.chat_postMessage(channel=channel_id, text=text)

@app.get("/")
def health():
    return {"status": "meeting-agent-running"}

# ───────────────────────────────────────
# HEARTBEAT — Proactive Supervisor
# Called by cron job every X minutes
# Reviews recent activity and posts
# next steps to the channel as ICAPP Agent
# ───────────────────────────────────────

HEARTBEAT_COMPRESSION_PROMPT = """
You are summarizing recent project activity.

Summarize everything into the most important points.
Preserve ALL specific details — credentials, run IDs,
file names, commands, errors, decisions, questions asked,
answers given, what was built, what failed, what was planned.

The summary must be accurate and dense — nothing important should be lost.
Output as bullet points. Max 300 words.

RECENT ACTIVITY:
{raw_context}
""".strip()

HEARTBEAT_SUPERVISOR_PROMPT = """
You are ICAPP Agent, acting as Professor Gautam Shroff —
a demanding project supervisor who expects results, not excuses.

PROJECT CONTEXT:
{context}

YOUR TASK:
1. STATUS (1 sentence): What has the student actually shipped or completed recently?
   If nothing concrete — say so bluntly.

2. GAP (1 sentence): What was supposed to be done but isn't? Be direct.

3. NEXT STEPS (2-3 bullet points): What MUST be done today. No fluff.
   Use exact names — files, functions, features — from the context.

4. DEADLINE QUESTION (1 sentence): Ask them when exactly it will be done.
   Not "are you working on it" — ask for a specific time/date commitment.

RULES:
- Address student as "you" directly, no softening
- Short sentences. No filler words.
- NEVER say "based on the context" or "according to the transcript"
- NEVER mention Slack, chunks, embeddings, or any system internals
- NEVER praise unless something was genuinely shipped
- If nothing happened recently, say exactly:
  "No activity detected. What is blocking you and when will it be resolved?"
- Max 120 words. Every word must earn its place.
""".strip()


def run_heartbeat(channel_id: str):
    """
    Core heartbeat logic — fetch recent context, compress if needed,
    run supervisor prompt, post to Slack channel.
    """
    from groq import Groq as _Groq
    groq_client = _Groq(api_key=os.getenv("GROQ_API_KEY"))
    supabase = get_supabase_client()

    print(f"[HEARTBEAT] Running for channel: {channel_id}")

    # ── 1. Get user for this channel ──
    user = supabase.get_user_by_username(channel_id)
    if not user:
        print(f"[HEARTBEAT] No user found for channel {channel_id}, skipping.")
        return
    user_id = user["id"]

    # ── 2. Get slack meeting id for this channel ──
    meeting_name = f"slack_{channel_id}"
    meeting = supabase.get_meeting_by_name(meeting_name)

    # ── 3. Fetch last 5 slack chunks ──
    slack_chunks_text = ""
    if meeting:
        try:
            result = supabase.client.table("chunks") \
                .select("chunk_text, created_at") \
                .eq("meeting_id", meeting["id"]) \
                .eq("source", "slack") \
                .order("created_at", desc=True) \
                .limit(5) \
                .execute()

            if result.data:
                chunks = list(reversed(result.data))  # chronological order
                slack_chunks_text = "\n\n".join(
                    c["chunk_text"] for c in chunks if c.get("chunk_text")
                )
                print(f"[HEARTBEAT] Fetched {len(result.data)} slack chunks")
        except Exception as e:
            print(f"[HEARTBEAT] Slack chunks fetch failed: {e}")

    # ── 4. Fetch last 20 chat turns ──
    chat_turns_text = ""
    try:
        # Get most recent session for this user
        session_result = supabase.client.table("sessions") \
            .select("session_id") \
            .eq("user_id", user_id) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()

        if session_result.data:
            session_id = session_result.data[0]["session_id"]
            turns = supabase.get_recent_chat_turns(session_id=session_id, limit=20)
            if turns:
                chat_turns_text = "\n".join(turns)
                print(f"[HEARTBEAT] Fetched {len(turns)} chat turns")
    except Exception as e:
        print(f"[HEARTBEAT] Chat turns fetch failed: {e}")

    # ── 5. Fetch last 3 transcript chunks ──
    transcript_text = ""
    try:
        transcript_result = supabase.client.table("chunks") \
            .select("chunk_text, created_at") \
            .eq("source", "transcript") \
            .order("created_at", desc=True) \
            .limit(3) \
            .execute()

        if transcript_result.data:
            chunks = list(reversed(transcript_result.data))
            transcript_text = "\n\n".join(
                c["chunk_text"] for c in chunks if c.get("chunk_text")
            )
            print(f"[HEARTBEAT] Fetched {len(transcript_result.data)} transcript chunks")
    except Exception as e:
        print(f"[HEARTBEAT] Transcript fetch failed: {e}")

    # ── 6. Combine all context ──
    raw_parts = []
    if slack_chunks_text:
        raw_parts.append(f"=== RECENT SLACK ACTIVITY ===\n{slack_chunks_text}")
    if chat_turns_text:
        raw_parts.append(f"=== RECENT Q&A WITH BOT ===\n{chat_turns_text}")
    if transcript_text:
        raw_parts.append(f"=== LATEST MEETING ===\n{transcript_text}")

    if not raw_parts:
        print("[HEARTBEAT] No context found, posting default message.")
        slack_client.chat_postMessage(
            channel=channel_id,
            text="No activity detected. What is blocking you and when will it be resolved?"
        )
        return

    raw_context = "\n\n".join(raw_parts)

    # ── 7. Compress if too big (> 2000 words) ──
    word_count = len(raw_context.split())
    print(f"[HEARTBEAT] Raw context word count: {word_count}")

    if word_count > 2000:
        print("[HEARTBEAT] Context too big — compressing first...")
        try:
            compress_response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{
                    "role": "user",
                    "content": HEARTBEAT_COMPRESSION_PROMPT.format(raw_context=raw_context)
                }],
                temperature=0.0,
                max_tokens=600,
            )
            context_for_supervisor = compress_response.choices[0].message.content.strip()
            print(f"[HEARTBEAT] Compressed to {len(context_for_supervisor.split())} words")
        except Exception as e:
            print(f"[HEARTBEAT] Compression failed: {e} — using raw context truncated")
            context_for_supervisor = " ".join(raw_context.split()[:2000])
    else:
        context_for_supervisor = raw_context

    # ── 8. Run supervisor prompt ──
    try:
        supervisor_response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": HEARTBEAT_SUPERVISOR_PROMPT.format(context=context_for_supervisor)
            }],
            temperature=0.2,
            max_tokens=300,
        )
        supervisor_message = supervisor_response.choices[0].message.content.strip()
        print(f"[HEARTBEAT] Supervisor message generated: {supervisor_message[:80]}...")
    except Exception as e:
        print(f"[HEARTBEAT] Supervisor prompt failed: {e}")
        return

    # ── 9. Post to Slack ──
    try:
        slack_client.chat_postMessage(
            channel=channel_id,
            text=supervisor_message
        )
        print(f"[HEARTBEAT] Posted to channel {channel_id}")
    except Exception as e:
        print(f"[HEARTBEAT] Slack post failed: {e}")


@app.post("/heartbeat")
async def heartbeat(request: Request, background_tasks: BackgroundTasks):
    """
    POST /heartbeat
    Body: { "channel_id": "C0123456789" }
    Called by cron job every 5 minutes (increase later).
    """
    try:
        body = await request.json()
        channel_id = body.get("channel_id")

        if not channel_id:
            return {"status": "error", "message": "channel_id required"}

        background_tasks.add_task(run_heartbeat, channel_id)
        return {"status": "ok", "message": f"Heartbeat triggered for {channel_id}"}

    except Exception as e:
        return {"status": "error", "message": str(e)}