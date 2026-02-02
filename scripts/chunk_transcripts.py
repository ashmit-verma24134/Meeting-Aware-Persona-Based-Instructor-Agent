import os
import json
from datetime import datetime, timezone

CLEAN_TRANSCRIPTS_DIR = "data/transcripts/cleaned"
OUTPUT_CHUNKS_FILE = "data/chunks.json"
MEETING_METADATA_FILE = "data/meeting_metadata.json"

CHUNK_SIZE = 350
OVERLAP = 50


def chunk_text(text: str, chunk_size: int, overlap: int):
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))

    return chunks


def infer_metadata_from_filename(filename: str):
    name = filename.replace(".txt", "").replace("C_", "")
    parts = name.split("_")

    if len(parts) < 3:
        raise ValueError(f"Invalid filename format: {filename}")

    user_id = parts[0]

    try:
        meeting_index = int(parts[-1])
    except ValueError:
        raise ValueError(f"{filename} must end with numeric meeting index")

    meeting_name = "_".join(parts[1:-1])
    meeting_type = "live_meeting"

    return user_id, meeting_name, meeting_type, meeting_index


def infer_project_type(meeting_name: str, text: str) -> str:
    name = meeting_name.lower()
    sample = text.lower()[:2000]

    medical_keywords = {
        "migraine", "diagnosis", "treatment",
        "patient", "symptom", "clinical"
    }

    system_keywords = {
        "agent", "architecture", "faiss",
        "embedding", "pipeline", "retrieval"
    }

    if any(k in name or k in sample for k in medical_keywords):
        return "medical"

    if any(k in name or k in sample for k in system_keywords):
        return "system_design"

    return "meeting_qa"


def main():
    all_chunks = []
    meeting_metadata = {}
    chunk_id = 0

    for filename in sorted(os.listdir(CLEAN_TRANSCRIPTS_DIR)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(CLEAN_TRANSCRIPTS_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            continue

        user_id, meeting_name, meeting_type, meeting_index = (
            infer_metadata_from_filename(filename)
        )

        project_type = infer_project_type(meeting_name, text)
        chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)

        if not chunks:
            continue

        metadata_key = f"{user_id}::meeting_{meeting_index}"

        meeting_metadata[metadata_key] = {
            "user_id": user_id,
            "meeting_index": meeting_index,
            "meeting_name": meeting_name,
            "meeting_type": meeting_type,
            "project_type": project_type,
        }

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": chunk_id,
                "user_id": user_id,
                "meeting_name": meeting_name,
                "meeting_type": meeting_type,
                "meeting_index": meeting_index,
                "project_type": project_type,
                "chunk_index": idx,
                "text": chunk,
                "created_at": datetime.now(timezone.utc).isoformat()
            })
            chunk_id += 1

    all_chunks.sort(key=lambda c: (
        c["user_id"],
        c["meeting_index"],
        c["chunk_index"]
    ))

    with open(OUTPUT_CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    with open(MEETING_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meeting_metadata, f, indent=2)

    print(f"Generated {len(all_chunks)} chunks")
    print(f"Generated metadata for {len(meeting_metadata)} meetings")


if __name__ == "__main__":
    main()
