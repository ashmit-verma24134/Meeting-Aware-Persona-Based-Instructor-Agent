import os
import re
from dotenv import load_dotenv
from services.embedding_api import get_embedding
from services.supabase_service import SupabaseService
from scripts.embedding_utils import build_embedding_text

load_dotenv()

# ===================================================
# KEYFRAME CHUNKING
# Split transcript by [HH:MM:SS] timestamp markers
# Each keyframe = one semantically coherent chunk
# ===================================================

def chunk_by_keyframe(text: str) -> list[dict]:
    """
    Split transcript into keyframe chunks.
    Each chunk = one [timestamp] block.
    Returns list of {text, timestamp_start, topic, speaker} dicts.
    """
    # Match lines starting with [HH:MM:SS] or [HH:MM:SS.ms]
    pattern = re.compile(r'(\[\d{2}:\d{2}:\d{2}(?:\.\d+)?\])')
    parts = pattern.split(text)

    chunks = []
    i = 0

    # parts = ['', '[00:00:00]', 'content...', '[00:00:46]', 'content...', ...]
    while i < len(parts):
        if pattern.match(parts[i]):
            timestamp = parts[i].strip("[]")
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            i += 2

            if not content:
                continue

            # Extract speaker from content if present
            # Format: "[U0AANT5PPPH]: text" or "The speaker says..."
            speaker = None
            speaker_match = re.match(r'^\[([^\]]+)\]:\s*', content)
            if speaker_match:
                speaker = speaker_match.group(1)

            # Extract topic — first 6 words of content as a label
            words = content.split()
            topic = " ".join(words[:6]) if words else ""

            chunks.append({
                "text": content,
                "timestamp_start": timestamp,
                "topic": topic,
                "speaker": speaker,
            })
        else:
            i += 1

    # Fallback: if no timestamps found, use word-level chunking
    if not chunks:
        print("[CHUNK] No timestamps found — falling back to word-level chunking")
        words = text.split()
        step = 300
        for idx in range(0, len(words), step):
            chunk_text = " ".join(words[idx:idx + step])
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "timestamp_start": None,
                    "topic": " ".join(chunk_text.split()[:6]),
                    "speaker": None,
                })

    return chunks


# ===================================================
# DYNAMIC GOAL EVOLUTION
# ===================================================

def update_dynamic_project_goal(supabase, user_uuid: str, new_meeting_text: str):
    try:
        from groq import Groq
        import os
        
        goal_meeting_name = f"goal_{user_uuid}"
        goal_meeting = supabase.get_meeting_by_name(goal_meeting_name)
        
        current_goal = ""
        goal_meeting_id = None
        
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
                
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        prompt = f"""
You are the master project tracker. 
You must update the overall PROJECT GOAL document based on a new meeting.

CURRENT GOAL:
{current_goal if current_goal else "No previous goal. This is the first meeting."}

NEW MEETING TRANSCRIPT:
{new_meeting_text[:8000]}

TASK:
Write the updated PROJECT GOAL. Keep it concise, authoritative, and strictly focus on the overarching goals, major technical decisions, and current scope. 
If the new meeting reveals a pivot or new requirement, update the goal accordingly.
Do NOT just summarize the meeting. Update the GOAL state. Max 250 words.
"""
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt.strip()}],
            temperature=0.2,
            max_tokens=300
        )
        
        new_goal = response.choices[0].message.content.strip()
        emb = get_embedding(new_goal)
        
        supabase.client.table("chunks").upsert([{
            "meeting_id": goal_meeting_id,
            "chunk_index": 0,
            "chunk_text": new_goal,
            "embedding": emb,
            "source": "transcript"
        }], on_conflict="meeting_id,chunk_index").execute()
        
        print(f"[GOAL TRACKER] Dynamic goal updated successfully. Length: {len(new_goal)} chars")
        
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

    # ================= USER =================

    user = supabase.get_user_by_username(username)

    if not user:
        print(f"Creating new user: {username}")
        user = supabase.create_user(username)

    user_uuid = user["id"]

    # ================= MEETING (BY RUN_ID) =================

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

    # ================= TRANSCRIPT =================

    if not supabase.transcript_exists(meeting_id):
        supabase.upsert_transcript(meeting_id, text)
        print("Transcript inserted.")
    else:
        print("Transcript already exists.")

    # ================= CHUNKS =================

    if supabase.chunks_exist(meeting_id):
        print("Chunks already exist. Skipping embedding.")
        return

    # Split by keyframe timestamps
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

    # ------ DYNAMIC GOAL UPDATE ------
    print("Updating dynamic project goal...")
    update_dynamic_project_goal(supabase, user_uuid, text)

    print("Single file ingestion completed.")


# ==============================================
# MAIN RUNNER
# ==============================================

if __name__ == "__main__":

    username = "test_user"
    DATA_DIR = "."

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    print(f"Found {len(files)} meeting files")

    for file in files:
        run_id = os.path.splitext(file)[0]
        ingest_single_file(
            file_path=os.path.join(DATA_DIR, file),
            username=username,
            run_id=run_id
        )