import os
import json
import re
from datetime import datetime, timezone
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


CLEAN_TRANSCRIPTS_DIR = "data/transcripts/cleaned"
OUTPUT_CHUNKS_FILE = "data/chunks.json"
MEETING_METADATA_FILE = "data/meeting_metadata.json"


CHUNK_SIZE = 350
OVERLAP = 50



def chunk_text(text: str, chunk_size: int, overlap: int):  #chunking text
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))

    return chunks

# -----------------------------
# FILENAME → METADATA
# -----------------------------
def infer_metadata_from_filename(filename: str):

    name = filename.replace(".txt", "").replace("C_", "")
    parts = name.split("_")

    if len(parts) < 3:
        raise ValueError(f"Invalid filename format: {filename}")

    user_id = parts[0]

    try:
        meeting_index = int(parts[-1])
    except ValueError:
        raise ValueError(f"{filename} must end with a numeric meeting index")

    meeting_name = "_".join(parts[1:-1])
    meeting_type = "live_meeting"

    return user_id, meeting_name, meeting_type, meeting_index

# -----------------------------
# PROJECT TYPE INFERENCE (MANDATORY)
# -----------------------------
def infer_project_type(meeting_name: str, text: str) -> str:
    """
    Conservative domain isolation.
    Never guesses aggressively.
    """

    name = meeting_name.lower()
    sample = text.lower()[:2500]

    medical_keywords = {
        "migraine", "symptom", "diagnosis", "treatment",
        "headache", "nausea", "vomiting", "clinical", "patient"
    }

    system_keywords = {
        "agent", "architecture", "retrieval", "embedding",
        "evaluation", "faiss", "cli", "pipeline", "chunk"
    }

    if any(k in name or k in sample for k in medical_keywords):
        return "medical"

    if any(k in name or k in sample for k in system_keywords):
        return "system_design"

    return "meeting_qa"

# -----------------------------
# SAFE JSON EXTRACTION
# -----------------------------
def safe_json_from_text(text: str) -> dict:
    try:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found")
        return json.loads(match.group())
    except Exception:
        return {
            "topics": [],
            "agents": [],
            "summary": ""
        }

# -----------------------------
# STRUCTURED METADATA EXTRACTION
# -----------------------------
def extract_meeting_metadata(sample_text: str) -> dict:
    prompt = f"""
Return ONLY valid JSON.

Format:
{{
  "topics": [],
  "agents": [],
  "summary": ""
}}

Transcript:
{sample_text}
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )
        return safe_json_from_text(response.choices[0].message.content)
    except Exception:
        return {
            "topics": [],
            "agents": [],
            "summary": ""
        }

# -----------------------------
# SUMMARY FALLBACK (GUARANTEED)
# -----------------------------
def extract_summary_only(sample_text: str) -> str:
    prompt = f"""
Summarize this meeting in 2–3 sentences.
Focus on technical or evaluative discussion.

Transcript:
{sample_text}
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""

# -----------------------------
# MAIN PIPELINE
# -----------------------------
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

        sample_text = (
            "\n".join(chunks[:2] + chunks[-2:])
            if len(chunks) >= 4
            else "\n".join(chunks)
        )

        metadata = extract_meeting_metadata(sample_text)
        summary = metadata.get("summary", "").strip()
        if not summary:
            summary = extract_summary_only(sample_text)

        metadata_key = f"{user_id}::meeting_{meeting_index}"

        meeting_metadata[metadata_key] = {
            "user_id": user_id,
            "meeting_index": meeting_index,
            "meeting_name": meeting_name,
            "meeting_type": meeting_type,
            "project_type": project_type,
            "topics": metadata.get("topics", []),
            "agents": metadata.get("agents", []),
            "summary": summary
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

    print(f" Generated {len(all_chunks)} chunks")
    print(f" Generated metadata for {len(meeting_metadata)} meetings")

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    main()
