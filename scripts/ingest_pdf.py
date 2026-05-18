"""
Ingest a PDF (local file path) into Supabase as embedded chunks.
Chunks labeled source="pdf" are retrievable in normal Q&A flow.

Triggered automatically when someone uploads a PDF to the Slack channel.
Also called during /sync_slack to catch PDFs shared before bot joined.
"""

import os
import hashlib

from services.supabase_service import SupabaseService
from services.embedding_api import get_embedding

MAX_CHUNK_WORDS = 400


def _chunk_text(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
        if words[i:i + max_words]
    ]


def _extract_pdf_text(path: str) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text.strip())
    return "\n\n".join(pages)


def ingest_pdf(file_path: str, channel_id: str, source_id: str = None) -> tuple[int, str]:
    """
    Ingest a PDF from a local file path.
    source_id: stable identifier for dedup (use Slack file_id if available).
    Returns (chunk_count, meeting_name).
    """
    supabase = SupabaseService()

    user = supabase.get_user_by_username(channel_id)
    if not user:
        user = supabase.create_user(channel_id)
    user_uuid = user["id"]

    text = _extract_pdf_text(file_path)
    if not text.strip():
        raise ValueError("No text could be extracted from this PDF.")

    # Use source_id for stable naming so the same Slack file isn't re-ingested
    stable_key = source_id or file_path
    key_hash = hashlib.md5(stable_key.encode()).hexdigest()[:8]
    meeting_name = f"pdf_{key_hash}_{channel_id}"

    existing = supabase.get_meeting_by_name(meeting_name)
    if existing:
        print(f"[PDF INGEST] Already ingested {meeting_name}, skipping.")
        return 0, meeting_name

    m = supabase.create_meeting({
        "meeting_name": meeting_name,
        "user_id": user_uuid,
        "channel_id": channel_id,
        "run_id": meeting_name,
        "status": "ingested",
    })
    meeting_id = m["id"]

    supabase.upsert_transcript(meeting_id, text)

    chunks = _chunk_text(text)
    print(f"[PDF INGEST] {len(chunks)} chunks for {meeting_name}")

    chunk_rows = []
    for idx, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        chunk_rows.append({
            "meeting_id": meeting_id,
            "chunk_index": idx,
            "chunk_text": chunk,
            "embedding": emb,
            "source": "pdf",
            "topic": " ".join(chunk.split()[:6]),
        })

    for i in range(0, len(chunk_rows), 20):
        supabase.insert_chunks(chunk_rows[i:i + 20])

    print(f"[PDF INGEST] Inserted {len(chunk_rows)} chunks as {meeting_name}")
    return len(chunk_rows), meeting_name
