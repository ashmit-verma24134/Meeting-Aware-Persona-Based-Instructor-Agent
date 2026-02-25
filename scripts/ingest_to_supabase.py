import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from services.supabase_service import SupabaseService
from scripts.embedding_utils import build_embedding_text

load_dotenv()

TRANSCRIPTS_DIR = "data/transcripts/cleaned"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 350
OVERLAP = 50


# ===================================================
# GLOBAL MODEL (Load once, reuse forever)
# ===================================================

_VECTOR_MODEL = None

def get_vector_model():
    global _VECTOR_MODEL
    if _VECTOR_MODEL is None:
        print("Loading embedding model...")
        _VECTOR_MODEL = SentenceTransformer(MODEL_NAME)
        print("Embedding model loaded.")
    return _VECTOR_MODEL


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
# USERNAME EXTRACTOR
# C_utsav_live_meeting_1.txt → utsav
# ===================================================

def extract_username(filename: str):
    name = filename.replace(".txt", "")
    parts = name.split("_")

    if len(parts) < 2:
        raise ValueError(f"Invalid filename format: {filename}")

    return parts[1]


# ===================================================
# CORE INGEST LOGIC (Single File)
# ===================================================

def ingest_single_file(file_path: str):

    if not os.path.exists(file_path):
        print("File not found:", file_path)
        return

    filename = os.path.basename(file_path)
    meeting_name = filename.replace(".txt", "")
    username = extract_username(filename)

    print(f"\nProcessing: {filename}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        print("Empty file. Skipping.")
        return

    model = get_vector_model()
    supabase = SupabaseService()

    # ================= USER =================

    user = supabase.get_user_by_username(username)

    if not user:
        print(f"Creating new user: {username}")
        user = supabase.create_user(username)

    user_uuid = user["id"]

    # ================= MEETING =================

    meeting = supabase.get_meeting_by_name(meeting_name)

    if meeting and meeting["user_id"] == user_uuid:
        meeting_id = meeting["id"]
        print("Meeting exists. Reusing.")
    else:
        meeting = supabase.create_meeting(meeting_name, user_uuid)
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

        embedding = model.encode(
            embedding_text,
            normalize_embeddings=True
        ).tolist()

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


# ===================================================
# OPTIONAL: FOLDER MODE (CLI)
# ===================================================

def main():

    if not os.path.exists(TRANSCRIPTS_DIR):
        print(f"Transcript folder not found: {TRANSCRIPTS_DIR}")
        return

    files = [f for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith(".txt")]

    if not files:
        print("No transcript files found.")
        return

    for filename in files:
        file_path = os.path.join(TRANSCRIPTS_DIR, filename)
        ingest_single_file(file_path)

    print("\nFolder ingestion completed successfully.")


if __name__ == "__main__":
    main()
