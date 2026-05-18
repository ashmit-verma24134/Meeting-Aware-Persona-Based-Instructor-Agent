import os
import re
from dotenv import load_dotenv
from services.embedding_api import get_embedding
from services.supabase_service import SupabaseService

load_dotenv()

# ===================================================
# KEYFRAME CHUNKING
# ===================================================

MAX_CHUNK_WORDS = 400


def _split_large_text(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def chunk_by_keyframe(text: str) -> list[dict]:
    pattern = re.compile(r'(\[\d{2}:\d{2}:\d{2}(?:\.\d+)?\])')
    parts = pattern.split(text)

    raw_chunks = []
    i = 0

    while i < len(parts):
        if pattern.match(parts[i]):
            timestamp = parts[i].strip("[]")
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            i += 2

            if not content:
                continue

            speaker = None
            speaker_match = re.match(r'^\[([^\]]+)\]:\s*', content)
            if speaker_match:
                speaker = speaker_match.group(1)

            raw_chunks.append({
                "text": content,
                "timestamp_start": timestamp,
                "speaker": speaker,
            })
        else:
            i += 1

    if not raw_chunks:
        print("[CHUNK] No timestamps found — falling back to word-level chunking")
        words = text.split()
        for idx in range(0, len(words), MAX_CHUNK_WORDS):
            chunk_text = " ".join(words[idx:idx + MAX_CHUNK_WORDS])
            if chunk_text:
                raw_chunks.append({
                    "text": chunk_text,
                    "timestamp_start": None,
                    "speaker": None,
                })

    # Split any chunk that exceeds MAX_CHUNK_WORDS to keep embeddings accurate
    chunks = []
    for rc in raw_chunks:
        for sub in _split_large_text(rc["text"]):
            words = sub.split()
            chunks.append({
                "text": sub,
                "timestamp_start": rc["timestamp_start"],
                "topic": " ".join(words[:6]) if words else "",
                "speaker": rc["speaker"],
            })

    return chunks


# ===================================================
# DYNAMIC GOAL EVOLUTION
#
# Called ONLY from ingest_slack_to_supabase.py (on /sync_slack).
# NOT called on meeting ingest — goal reflects Slack activity, not raw transcripts.
#
# How it works:
# 1. Fetches ALL past transcript meeting chunks for this user
# 2. Takes a sample from each meeting (first 5 chunks)
# 3. Passes: [current goal] + [all past meetings] + [recent slack text]
# 4. LLM rebuilds the goal intelligently from full project history
# 5. Upserts back into the goal_{user_uuid} chunk
# ===================================================

def update_dynamic_project_goal(supabase: SupabaseService, user_uuid: str, new_meeting_text: str):
    try:
        from groq import Groq

        goal_meeting_name = f"goal_{user_uuid}"
        goal_meeting = supabase.get_meeting_by_name(goal_meeting_name)

        current_goal = ""
        goal_meeting_id = None

        # ── Get or create goal meeting record ──
        if not goal_meeting:
            gm = supabase.create_meeting({
                "meeting_name": goal_meeting_name,
                "user_id": user_uuid,
                "channel_id": "goal_tracker",
                "run_id": f"goal_{user_uuid}",
                "status": "ingested"
            })
            goal_meeting_id = gm["id"]
        else:
            goal_meeting_id = goal_meeting["id"]
            res = supabase.client.table("chunks") \
                .select("chunk_text") \
                .eq("meeting_id", goal_meeting_id) \
                .limit(1).execute()
            if res.data:
                current_goal = res.data[0]["chunk_text"]

        print(f"[GOAL TRACKER] Current goal length: {len(current_goal)} chars")

        # ── Fetch ALL past transcript meetings for this user ──
        all_past_meetings_text = ""
        try:
            all_meetings_res = supabase.client.table("meetings") \
                .select("id, meeting_name, created_at") \
                .eq("user_id", user_uuid) \
                .not_.like("meeting_name", "slack_%") \
                .not_.like("meeting_name", "goal_%") \
                .order("created_at", desc=False) \
                .limit(10) \
                .execute()

            meeting_parts = []
            if all_meetings_res.data:
                for m in all_meetings_res.data:
                    chunks_res = supabase.client.table("chunks") \
                        .select("chunk_text, chunk_index") \
                        .eq("meeting_id", m["id"]) \
                        .eq("source", "transcript") \
                        .order("chunk_index", desc=False) \
                        .limit(5) \
                        .execute()

                    if chunks_res.data:
                        meeting_chunk_text = "\n".join(
                            c["chunk_text"] for c in chunks_res.data if c.get("chunk_text")
                        )
                        meeting_name = m.get("meeting_name", "Unknown")
                        meeting_date = (m.get("created_at") or "")[:10]
                        meeting_parts.append(
                            f"[Meeting: {meeting_name} | Date: {meeting_date}]\n{meeting_chunk_text}"
                        )

            if meeting_parts:
                all_past_meetings_text = "\n\n---\n\n".join(meeting_parts)
                print(f"[GOAL TRACKER] Loaded {len(meeting_parts)} past meetings for goal synthesis")

        except Exception as e:
            print(f"[GOAL TRACKER] Past meetings fetch failed (continuing with new text only): {e}")

        # ── Build the prompt ──
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        combined_history = all_past_meetings_text
        if len(combined_history.split()) > 6000:
            combined_history = " ".join(combined_history.split()[:6000]) + "\n[...earlier meetings truncated...]"

        prompt = f"""
You are the master project tracker for a student capstone project.

Your job is to synthesize ONE authoritative, up-to-date PROJECT GOAL document.
This document must reflect the FULL arc of the project — from first meeting to now.

---

PREVIOUS GOAL (may be empty for first ingestion):
{current_goal if current_goal else "No previous goal recorded yet. This is the first synthesis."}

---

ALL PAST MEETING CONTENT (full project history, oldest first):
{combined_history if combined_history else "No previous meetings found."}

---

RECENT SLACK ACTIVITY (latest team communication):
{new_meeting_text[:4000]}

---

YOUR TASK:
Write the updated PROJECT GOAL document. It must:
1. CAPTURE the core technical objective of the project (what are they building?)
2. LIST the major technical decisions made so far (tools, models, architecture)
3. DESCRIBE the current scope and any pivots or changes from original plan
4. NOTE what has been completed vs what is still in progress
5. Be written as a living document — authoritative, present-tense, dense
6. Max 300 words. Every word earns its place.

DO NOT just summarize the latest Slack messages.
DO NOT repeat things already noted — evolve and update the goal state.
Write as if you are the project's single source of truth.
""".strip()

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400
        )

        new_goal = response.choices[0].message.content.strip()
        print(f"[GOAL TRACKER] New goal generated ({len(new_goal)} chars): {new_goal[:100]}...")

        # ── Embed and upsert ──
        emb = get_embedding(new_goal)

        supabase.client.table("chunks").upsert([{
            "meeting_id": goal_meeting_id,
            "chunk_index": 0,
            "chunk_text": new_goal,
            "embedding": emb,
            "source": "goal"
        }], on_conflict="meeting_id,chunk_index").execute()

        print(f"[GOAL TRACKER] Goal updated successfully.")

    except Exception as e:
        print(f"[GOAL TRACKER] Failed to update dynamic goal: {e}")


# ===================================================
# CORE INGEST LOGIC
# ===================================================

def ingest_single_file(file_path: str, username: str, run_id: str):

    if not os.path.exists(file_path):
        print("File not found:", file_path)
        return

    print(f"\nProcessing run_id: {run_id}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        print("Empty file. Skipping.")
        return

    supabase = SupabaseService()

    # ── User ──
    user = supabase.get_user_by_username(username)
    if not user:
        print(f"Creating new user: {username}")
        user = supabase.create_user(username)
    user_uuid = user["id"]

    # ── Meeting ──
    meeting = supabase.get_meeting_by_run_id(run_id)
    if meeting:
        meeting_id = meeting["id"]
        print("Meeting already exists. Reusing.")
    else:
        meeting = supabase.create_meeting({
            "meeting_name": f"meeting_{run_id}",
            "user_id": user_uuid,
            "channel_id": username,
            "run_id": run_id,
            "status": "ingested"
        })
        meeting_id = meeting["id"]
        print("Meeting created.")

    # ── Transcript ──
    if not supabase.transcript_exists(meeting_id):
        supabase.upsert_transcript(meeting_id, text)
        print("Transcript inserted.")
    else:
        print("Transcript already exists.")

    # ── Chunks ──
    if supabase.chunks_exist(meeting_id):
        print("Chunks already exist. Skipping embedding.")
        # Goal is NOT updated here — only /sync_slack updates the goal
        return

    keyframe_chunks = chunk_by_keyframe(text)
    print(f"Generated {len(keyframe_chunks)} keyframe chunks.")

    chunk_rows = []
    for idx, kf in enumerate(keyframe_chunks):
        chunk_text = kf["text"]
        embedding = get_embedding(chunk_text)
        chunk_rows.append({
            "meeting_id": meeting_id,
            "chunk_index": idx,
            "chunk_text": chunk_text,
            "embedding": embedding,
            "source": "transcript",
            "timestamp_start": kf.get("timestamp_start"),
            "topic": kf.get("topic"),
            "speaker": kf.get("speaker"),
        })

    if chunk_rows:
        supabase.insert_chunks(chunk_rows)
        print(f"Inserted {len(chunk_rows)} keyframe chunks.")

    # Goal is NOT updated here — only /sync_slack updates the goal
    print("Single file ingestion completed.")


# ===================================================
# MAIN RUNNER
# ===================================================

if __name__ == "__main__":
    username = "test_user"
    DATA_DIR = "."

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    print(f"Found {len(files)} meeting files")

    for file in sorted(files):
        run_id = os.path.splitext(file)[0]
        ingest_single_file(
            file_path=os.path.join(DATA_DIR, file),
            username=username,
            run_id=run_id
        )