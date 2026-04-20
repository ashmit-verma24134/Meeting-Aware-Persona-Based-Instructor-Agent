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
from scripts.ingest_slack_to_supabase import ingest_slack_history


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

SLACK_SESSIONS = {}  # channel_id → session info
# NOTE: PROCESSED_EVENTS is kept as a local cache for within-process dedup,
# but the primary dedup is now Supabase-based (see is_event_processed).
PROCESSED_EVENTS = set()
EVENT_LOCK = Lock()

STAGE_ACTIVE = "ACTIVE"

# ── Heartbeat cooldown tracker (in-memory) ──
_LAST_HEARTBEAT = {}  # channel_id → datetime
HEARTBEAT_COOLDOWN_MINUTES = 0


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


def sync_slack_and_reply(channel_id, response_url):
    try:
        ingest_slack_history(channel_id=channel_id, limit=5000)
        http_requests.post(response_url, json={
            "response_type": "in_channel",
            "text": "Slack history synced!"
        })
    except Exception as e:
        http_requests.post(response_url, json={
            "response_type": "in_channel",
            "text": f"Sync failed: {str(e)}"
        })

@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks):
    # ── Ignore Slack retries to prevent double responses on slow processing ──
    from fastapi.responses import JSONResponse
    if request.headers.get("x-slack-retry-num"):
        return JSONResponse({"status": "ok"}, headers={"X-Slack-No-Retry": "1"})

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
                "text": "Syncing Slack history for this channel..."
            }

    return {"status": "ok"}


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

        result = hf_service.fetch_result(run_id)

        if result["state"] != "completed":
            slack_client.chat_postMessage(
                channel=channel_id,
                text="Pipeline not finished yet."
            )
            return

        data = result["data"]

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
        meeting_name = f"meeting_{run_id}"
        file_path = f"{meeting_name}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        ingest_single_file(
            file_path=file_path,
            username=channel_id,
            run_id=run_id
        )

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

    # ── Ignore bot messages and structural events ──
    if event.get("bot_id"):
        return

    if event.get("subtype") in [
        "bot_message",
        "message_changed",
        "message_deleted",
        "thread_broadcast"
    ]:
        return

    event_id = event.get("client_msg_id") or event.get("event_ts")
    if not event_id:
        return

    # ── LOCAL dedup (same process, handles fast duplicates) ──
    with EVENT_LOCK:
        if event_id in PROCESSED_EVENTS:
            print(f"[DEDUP] Local: Skipping duplicate event {event_id}")
            return
        PROCESSED_EVENTS.add(event_id)
        if len(PROCESSED_EVENTS) > 10_000:
            PROCESSED_EVENTS.clear()

    # ── SUPABASE dedup (cross-process, handles serverless re-invocations) ──
    try:
        supabase = get_supabase_client()
        if supabase.is_event_processed(event_id):
            print(f"[DEDUP] Supabase: Skipping duplicate event {event_id}")
            return
    except Exception as e:
        print(f"[DEDUP] Supabase check failed (continuing): {e}")

    event_type = event.get("type")
    if event_type not in ["message", "app_mention"]:
        return

    slack_user_id = event.get("user")
    channel_id = event.get("channel")
    text = (event.get("text") or "").strip()

    if not text:
        return

    # ── FIX: Skip message events with bot @mentions in ANY channel type ──
    channel_type = event.get("channel_type", "")
    if event_type == "message" and "<@" in text and channel_type in ["channel", "group", "mpim"]:
        return

    # ── Remove bot mention formatting ──
    if event_type == "app_mention":
        text = re.sub(r"<@[^>]+>", "", text).strip()
        if text == "":
            return

# ── In ALL channel types → store all messages, only reply if mentioned ──
    if event_type == "message" and channel_type in ["channel", "group", "mpim"]:
        if "<@" not in (event.get("text") or ""):
            # Store to chunk as context even if bot not mentioned
            try:
                _supabase = get_supabase_client()
                _user = _supabase.get_user_by_username(channel_id)
                if _user:
                    _supabase.append_live_chat_to_slack_chunk(
                                            _user["id"], text, None
                                        )
                                        print(f"[CONTEXT STORE] Stored non-mention message as context")
                                        # Also store to chat_turns so chat_answer_node can see it
                                        session = SLACK_SESSIONS.get(channel_id)
                                        if session:
                                            _supabase.save_chat_turn(
                                                session_id=session["session_id"],
                                                user_id=session["user_id"],
                                                question=text,
                                                answer="[context]",
                                                source="user_context",
                                            )
            except Exception as e:
                print(f"[CONTEXT STORE] Failed: {e}")
            return  # Don't reply, just store

    if not slack_user_id or not channel_id:
        return

# ── Store user message to chunk + chat_turns immediately ──
    try:
        _supabase = get_supabase_client()
        _user = _supabase.get_user_by_username(channel_id)
        if _user:
            _supabase.append_live_chat_to_slack_chunk(
                _user["id"], text, None
            )
            session = SLACK_SESSIONS.get(channel_id)
            if session:
                _supabase.save_chat_turn(
                    session_id=session["session_id"],
                    user_id=session["user_id"],
                    question=text,
                    answer="[context]",
                    source="user_context",
                )
    except Exception as e:
        print(f"[STORE USER MSG] Failed: {e}")

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
            text=f"Meeting started! Run ID: `{run_id}`"
        )

    except Exception as e:
        slack_client.chat_postMessage(
            channel=channel_id,
            text=f"Failed to start: {str(e)}"
        )


# ───────────────────────────────────────
# USER HANDLER
# ───────────────────────────────────────

def handle_user_message(slack_user_id: str, channel_id: str, text: str):

    supabase = get_supabase_client()

    # ── Ensure channel-based user exists ──
    user = supabase.get_user_by_username(channel_id)
    if not user:
        user = supabase.create_user(channel_id)

    user_id = user["id"]

    session = SLACK_SESSIONS.get(channel_id)

    # ── Create session if not exists ──
    if session is None:
        sid = f"{channel_id}_slack_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        supabase.create_session(sid, user_id)

        session = {
            "user_id": user_id,
            "session_id": sid,
        }
        SLACK_SESSIONS[channel_id] = session

    # ── EXIT ──
    if text.lower().strip() == "exit":
        del SLACK_SESSIONS[channel_id]
        send_message(channel_id, "Session ended")
        return

    # ── NEW MEETING ──
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

    # ── STATUS ──
    if text.startswith("status"):

        run_id = session.get("current_run_id")

        if not run_id:
            send_message(channel_id, "No meeting has been started yet.")
            return

        state = hf_service.check_status(run_id)
        send_message(channel_id, f"Status: {state}")
        return

    # ── PROCESS ──
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

    # ── NORMAL QUESTION FLOW ──
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
            "is_hunting_for_exact_value": False,
            "context_extended": False,
            "path": [],
        }

        result = meeting_graph.invoke(initial_state)
        answer = result.get("final_answer")

        if not answer or answer.strip() == SAFE_ABSTAIN:
            send_message(
                channel_id,
                "Sorry  I couldn't find a clear answer in the meeting transcript."
            )
            return

        send_message(channel_id, answer)

    except Exception as e:
        import traceback
        traceback.print_exc()
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


# ───────────────────────────────────────
# HEARTBEAT — Proactive Supervisor
# ───────────────────────────────────────

HEARTBEAT_COMPRESSION_PROMPT = """
You are summarizing recent project activity.

Summarize everything into the most important points.
Preserve ALL specific details — credentials, run IDs,
file names, commands, errors.
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
1. GOAL PROGRESS (1 sentence): Look at the CURRENT PROJECT GOAL. Compare it to what has actually been done recently (from Slack activity and latest meetings). Be highly critical — has the student actually moved closer to the goal, or just been busy without progress?
2. STATUS (1 sentence): What concrete thing has the student shipped or completed? If nothing is done — say so bluntly.
3. NEXT STEPS (1–2 bullet points): What MUST be done TODAY to advance toward the current goal. Name exact files, functions, or features from the context.
4. GOAL PROXIMITY (1 sentence): Directly ask — "How close are you to completing [specific goal from context]? When exactly will it be done?"

RULES:
- Address student directly as "you". No softening.
- Short sentences. No filler words.
- NEVER say "based on the context" or "according to the transcript".
- NEVER mention Slack, chunks, embeddings, or any system internals.
- NEVER praise unless something was genuinely shipped.
- If nothing new has happened since the last check-in, say:
  "No new activity detected. Are you blocked on the current goal? What is stopping you and when will it be resolved?"
- Max 150 words. Every word must earn its place.
""".strip()


def run_heartbeat(channel_id: str):
    from groq import Groq as _Groq
    from datetime import timedelta
    groq_client = _Groq(api_key=os.getenv("GROQ_API_KEY"))
    supabase = get_supabase_client()

    print(f"[HEARTBEAT] Running for channel: {channel_id}")

    # ── Cooldown check ──
    now = datetime.utcnow()
    last = _LAST_HEARTBEAT.get(channel_id)
    if last and (now - last) < timedelta(minutes=HEARTBEAT_COOLDOWN_MINUTES):
        print(f"[HEARTBEAT] Skipping — last fired {(now-last).seconds}s ago, cooldown={HEARTBEAT_COOLDOWN_MINUTES}min")
        return
    _LAST_HEARTBEAT[channel_id] = now

    # ── 1. Get user ──
    user = supabase.get_user_by_username(channel_id)
    if not user:
        print(f"[HEARTBEAT] No user found for channel {channel_id}, skipping.")
        return
    user_id = user["id"]
    print(f"[HEARTBEAT] User found: {user_id}")

    # ── 2. Get slack meeting ──
    # NOTE: Auto-sync removed — heartbeat reads from DB directly.
    # Slack chunks are populated by /sync_slack command, not re-ingested here.
    meeting_name = f"slack_{channel_id}"
    meeting = supabase.get_meeting_by_name(meeting_name)
    print(f"[HEARTBEAT] Slack meeting found: {meeting is not None} | name={meeting_name}")

    # ── 3. Fetch last 2 dates of slack chunks ──
    summary_slack_chunks_count = 0
    slack_chunks_text = ""
    slack_dates_found = []
    if meeting:
        try:
            result = supabase.client.table("chunks") \
                .select("chunk_text, created_at") \
                .eq("meeting_id", meeting["id"]) \
                .eq("source", "slack") \
                .order("id", desc=True) \
                .limit(50) \
                .execute()
            print(f"[HEARTBEAT] Slack chunks in DB: {len(result.data) if result.data else 0}", flush=True)
            if result.data:
                collected_chunks = []
                date_pattern = re.compile(r'---\s*(\d{4}-\d{2}-\d{2})\s*---')
                seen_dates = set()

                for c in result.data:
                    text = c.get("chunk_text", "")
                    if text:
                        match = date_pattern.search(text)
                        if match:
                            date_str = match.group(1)
                            seen_dates.add(date_str)
                            if len(seen_dates) > 2:
                                break
                        collected_chunks.append(text)

                chunks = list(reversed(collected_chunks))
                slack_chunks_text = "\n\n".join(chunks)
                summary_slack_chunks_count = len(chunks)
                slack_dates_found = sorted(seen_dates)
                print(f"[HEARTBEAT] SLACK CONTEXT → {summary_slack_chunks_count} chunks | dates: {slack_dates_found}", flush=True)
                print(f"[HEARTBEAT] SLACK PREVIEW → {slack_chunks_text[:200]}...", flush=True)
        except Exception as e:
            print(f"[HEARTBEAT] Slack chunks fetch failed: {e}", flush=True)
    else:
        print(f"[HEARTBEAT] No slack meeting record — slack chunks SKIPPED. Run /sync_slack first.", flush=True)

    # ── 4. Fetch last 20 chat turns ──
    summary_chat_turns = 0
    chat_turns_text = ""
    try:
        turns = supabase.get_recent_chat_turns_by_user(user_id=user_id, limit=20)
        if turns:
            chat_turns_text = "\n".join(turns)
            summary_chat_turns = len(turns)
            print(f"[HEARTBEAT] CHAT TURNS CONTEXT → {summary_chat_turns} turns", flush=True)
            print(f"[HEARTBEAT] CHAT PREVIEW → {chat_turns_text[:200]}...", flush=True)
        else:
            print(f"[HEARTBEAT] No chat turns found for user {user_id}", flush=True)
    except Exception as e:
        print(f"[HEARTBEAT] Chat turns fetch failed: {e}", flush=True)

    # ── 5. Fetch last 2 transcript meetings ──
    summary_transcript_metrics = 0
    transcript_text = ""
    try:
        recent_meetings = supabase.get_recent_meetings(user_id=user_id, limit=2)
        if not recent_meetings:
            print("[HEARTBEAT] No recent transcript meetings found.", flush=True)
        else:
            meeting_parts = []
            for rm in recent_meetings:
                transcript_result = supabase.client.table("chunks") \
                    .select("chunk_text") \
                    .eq("meeting_id", rm["id"]) \
                    .eq("source", "transcript") \
                    .order("chunk_index", desc=False) \
                    .limit(8) \
                    .execute()

                chunk_count = len(transcript_result.data) if transcript_result.data else 0
                print(f"[HEARTBEAT] MEETING CONTEXT → '{rm.get('meeting_name')}' | {chunk_count} chunks (first 8)", flush=True)

                if transcript_result.data:
                    meeting_text = "\n\n".join(
                        c["chunk_text"] for c in transcript_result.data if c.get("chunk_text")
                    )
                    print(f"[HEARTBEAT] MEETING PREVIEW → {meeting_text[:200]}...", flush=True)
                    meeting_parts.append(f"[Meeting: {rm.get('meeting_name', 'Unknown')}]\n{meeting_text}")

            transcript_text = "\n\n".join(meeting_parts)
            summary_transcript_metrics = len(meeting_parts)
    except Exception as e:
        print(f"[HEARTBEAT] Transcript fetch failed: {e}", flush=True)

    # ── 5b. Fetch Evolving Project Goal ──
    goal_text = ""
    try:
        goal_meeting_name = f"goal_{user_id}"
        goal_meeting_res = supabase.client.table("meetings") \
            .select("id") \
            .eq("meeting_name", goal_meeting_name) \
            .limit(1) \
            .execute()

        if goal_meeting_res.data:
            goal_m = goal_meeting_res.data[0]
            goal_chunks = supabase.client.table("chunks") \
                .select("chunk_text") \
                .eq("meeting_id", goal_m["id"]) \
                .limit(1) \
                .execute()

            if goal_chunks.data:
                goal_text = goal_chunks.data[0].get("chunk_text", "")
                print(f"[HEARTBEAT] GOAL CONTEXT → {len(goal_text)} chars", flush=True)
                print(f"[HEARTBEAT] GOAL PREVIEW → {goal_text[:300]}...", flush=True)
            else:
                print(f"[HEARTBEAT] Goal meeting exists but no chunk found — run /sync_slack", flush=True)
        else:
            print(f"[HEARTBEAT] No goal record found — run /sync_slack to generate it", flush=True)
    except Exception as e:
        print(f"[HEARTBEAT] Goal fetch failed: {e}", flush=True)

    # ── 6. Combine + full context log ──
    print("\n" + "="*60, flush=True)
    print("🧠 HEARTBEAT CONTEXT SUMMARY", flush=True)
    print(f"  📌 GOAL          : {'✅ ' + str(len(goal_text)) + ' chars' if goal_text else '❌ MISSING — run /sync_slack'}", flush=True)
    print(f"  📝 MEETINGS      : {'✅ ' + str(summary_transcript_metrics) + ' meetings (first 8 chunks each)' if summary_transcript_metrics else '❌ none found'}", flush=True)
    print(f"  💬 SLACK         : {'✅ ' + str(summary_slack_chunks_count) + ' chunks | dates: ' + str(slack_dates_found) if summary_slack_chunks_count else '❌ none — run /sync_slack'}", flush=True)
    print(f"  🗨️  CHAT TURNS   : {'✅ ' + str(summary_chat_turns) + ' turns' if summary_chat_turns else '❌ none found'}", flush=True)
    print("="*60 + "\n", flush=True)

    raw_parts = []
    if goal_text:
        raw_parts.append(f"=== CURRENT PROJECT GOAL ===\n{goal_text}")
    if transcript_text:
        raw_parts.append(f"=== LATEST 2 MEETINGS ===\n{transcript_text}")
    if slack_chunks_text:
        raw_parts.append(f"=== RECENT SLACK ACTIVITY (last 2 dates) ===\n{slack_chunks_text}")
    if chat_turns_text:
        raw_parts.append(f"=== RECENT Q&A WITH BOT ===\n{chat_turns_text}")

    if not raw_parts:
        print("[HEARTBEAT] No context found, posting default message.", flush=True)
        slack_client.chat_postMessage(
            channel=channel_id,
            text="No activity detected. What is blocking you and when will it be resolved?"
        )
        return

    raw_context = "\n\n".join(raw_parts)
    word_count = len(raw_context.split())
    print(f"[HEARTBEAT] Raw context word count: {word_count}")

    if word_count > 10000:
        print("[HEARTBEAT] Context too big — compressing first...")
        try:
            compress_response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": HEARTBEAT_COMPRESSION_PROMPT.format(raw_context=raw_context)}],
                temperature=0.0,
                max_tokens=600,
            )
            context_for_supervisor = compress_response.choices[0].message.content.strip()
            print(f"[HEARTBEAT] Compressed to {len(context_for_supervisor.split())} words")
        except Exception as e:
            print(f"[HEARTBEAT] Compression failed: {e} — truncating")
            context_for_supervisor = " ".join(raw_context.split()[:10000])
    else:
        context_for_supervisor = raw_context

    try:
        supervisor_response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": HEARTBEAT_SUPERVISOR_PROMPT.format(context=context_for_supervisor)}],
            temperature=0.2,
            max_tokens=300,
        )
        supervisor_message = supervisor_response.choices[0].message.content.strip()
        print(f"[HEARTBEAT] Message generated: {supervisor_message[:80]}...")
    except Exception as e:
        print(f"[HEARTBEAT] Supervisor prompt failed: {e}")
        return

    try:
        slack_client.chat_postMessage(
            channel=channel_id,
            text="Project Status Update",
            blocks=[
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"🔔 *PROACTIVE CHECK-IN*\n\n{supervisor_message}"
                    }
                }
            ]
        )
        print(f"[HEARTBEAT] Posted to channel {channel_id}")
    except Exception as e:
        print(f"[HEARTBEAT] Slack post failed: {e}")


# ───────────────────────────────────────
# POST /heartbeat — manual trigger / internal calls
# ───────────────────────────────────────
@app.post("/heartbeat")
async def heartbeat_post(request: Request, background_tasks: BackgroundTasks):
    """
    POST /heartbeat
    Body: { "channel_id": "C0123456789" }
    Can be called manually or from heartbeat_cron.py.
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


# ───────────────────────────────────────
# GET /heartbeat — called by Vercel cron
# Vercel cron can only send GET requests with no body,
# so we read channel_id from environment variable.
# ───────────────────────────────────────
@app.get("/heartbeat")
async def heartbeat_get(background_tasks: BackgroundTasks):
    """
    GET /heartbeat
    Called automatically by Vercel cron every 5 minutes.
    Reads SLACK_CHANNEL_ID from environment — set this in your Vercel project settings.
    """
    channel_id = os.getenv("SLACK_CHANNEL_ID")
    if not channel_id:
        print("[HEARTBEAT GET] SLACK_CHANNEL_ID env var not set!")
        return {"status": "error", "message": "SLACK_CHANNEL_ID env var not set"}
    background_tasks.add_task(run_heartbeat, channel_id)
    return {"status": "ok", "message": f"Heartbeat triggered for {channel_id}"}# ─────────────────────────────────────────────────────────────────


@app.get("/debug")
async def debug_retrieval():
    """
    GET /debug
    Checks every layer of the retrieval pipeline and reports what's broken.
    Call this from browser: https://your-app.vercel.app/debug
    """
    import traceback
    report = {}

    # 1. Check env vars
    report["env"] = {
        "SUPABASE_URL": bool(os.getenv("SUPABASE_URL")),
        "SUPABASE_SERVICE_ROLE_KEY": bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY")),
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "HF_EMBED_URL": bool(os.getenv("HF_EMBED_URL")),
        "SLACK_BOT_TOKEN": bool(os.getenv("SLACK_BOT_TOKEN")),
        "SLACK_CHANNEL_ID": bool(os.getenv("SLACK_CHANNEL_ID")),
    }

    # 2. Check Supabase connection + tables
    try:
        supabase = get_supabase_client()
        users = supabase.client.table("users").select("id").limit(1).execute()
        report["supabase_users_table"] = "OK"
        report["supabase_users_count"] = len(users.data)
    except Exception as e:
        report["supabase_users_table"] = f"ERROR: {e}"

    try:
        meetings = supabase.client.table("meetings").select("id, meeting_name, user_id").limit(5).execute()
        report["supabase_meetings"] = [m["meeting_name"] for m in (meetings.data or [])]
    except Exception as e:
        report["supabase_meetings"] = f"ERROR: {e}"

    try:
        chunks = supabase.client.table("chunks").select("id, meeting_id, source").limit(5).execute()
        report["supabase_chunks_sample"] = chunks.data
    except Exception as e:
        report["supabase_chunks"] = f"ERROR: {e}"

    # 3. Check embedding API
    try:
        from services.embedding_api import get_embedding
        emb = get_embedding("test query hello world")
        report["embedding_api"] = f"OK — dim={len(emb)}"
    except Exception as e:
        report["embedding_api"] = f"ERROR: {e}"

    # 4. Check match_chunks_by_user RPC
    try:
        from services.embedding_api import get_embedding
        test_emb = get_embedding("what happened in the meeting")
        # Get first user_id from DB
        users_resp = supabase.client.table("users").select("id").limit(1).execute()
        if users_resp.data:
            test_user_id = users_resp.data[0]["id"]
            rpc_result = supabase.client.rpc(
                "match_chunks_by_user",
                {
                    "query_embedding": test_emb,
                    "match_count": 3,
                    "filter_user_id": test_user_id
                }
            ).execute()
            report["rpc_match_chunks_by_user"] = f"OK — returned {len(rpc_result.data or [])} chunks"
        else:
            report["rpc_match_chunks_by_user"] = "SKIP — no users in DB"
    except Exception as e:
        report["rpc_match_chunks_by_user"] = f"ERROR: {e}"

    # 5. Check match_chunks_bm25 RPC
    try:
        users_resp = supabase.client.table("users").select("id").limit(1).execute()
        if users_resp.data:
            test_user_id = users_resp.data[0]["id"]
            bm25_result = supabase.client.rpc(
                "match_chunks_bm25",
                {
                    "query_text": "meeting pipeline model",
                    "filter_user_id": test_user_id,
                    "match_count": 3
                }
            ).execute()
            report["rpc_match_chunks_bm25"] = f"OK — returned {len(bm25_result.data or [])} chunks"
        else:
            report["rpc_match_chunks_bm25"] = "SKIP — no users in DB"
    except Exception as e:
        report["rpc_match_chunks_bm25"] = f"ERROR: {e}"

    # 6. Check slack_events table (dedup)
    try:
        supabase.client.table("slack_events").select("event_id").limit(1).execute()
        report["slack_events_table"] = "OK"
    except Exception as e:
        report["slack_events_table"] = f"MISSING — run: CREATE TABLE slack_events (event_id TEXT PRIMARY KEY, created_at TIMESTAMPTZ DEFAULT NOW()); | Error: {e}"

    # 7. Check Groq
    try:
        from groq import Groq
        g = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = g.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5
        )
        report["groq"] = f"OK — {resp.choices[0].message.content.strip()}"
    except Exception as e:
        report["groq"] = f"ERROR: {e}"

    return report