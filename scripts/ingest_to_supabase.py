import os
from dotenv import load_dotenv
from services.embedding_api import get_embedding

from services.supabase_service import SupabaseService
from scripts.embedding_utils import build_embedding_text

load_dotenv()

CHUNK_SIZE = 350
OVERLAP = 50

# ===================================================
# GLOBAL MODEL (Load once, reuse forever)
# ===================================================


# ===================================================
# CHUNKING
# ===================================================

def chunk_text(text: str, chunk_size: int, overlap: int):
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))

    return chunks


# ===================================================
# CORE INGEST LOGIC (Multi-Meeting Safe)
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

    chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
    print(f"Generated {len(chunks)} chunks.")

    prev_chunk = None
    chunk_rows = []

    for idx, chunk in enumerate(chunks):

        embedding_text = build_embedding_text(
            {"text": chunk},
            prev_chunk
        )

        embedding = get_embedding(embedding_text)

        chunk_rows.append({
            "meeting_id": meeting_id,
            "chunk_index": idx,
            "chunk_text": chunk,
            "embedding": embedding
        })

        prev_chunk = {"text": chunk}

    if chunk_rows:
        supabase.insert_chunks(chunk_rows)
        print("Chunks inserted.")

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