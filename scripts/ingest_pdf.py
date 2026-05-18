"""
Ingest a PDF (local file path) into Supabase as embedded chunks.
Chunks labeled source="pdf" are retrievable in normal Q&A flow.

Triggered automatically when someone uploads a PDF to the Slack channel.
Also called during /sync_slack to catch PDFs shared before bot joined.
"""

import os
import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from services.supabase_service import SupabaseService
from services.embedding_api import get_embedding

logger = logging.getLogger(__name__)
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
    logger.info(f"[PDF] Opening file: {path}")
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic != b"%PDF":
        raise ValueError(
            f"Downloaded file is not a valid PDF (magic bytes: {magic!r}). "
            "Slack may have returned an HTML error page or the file is corrupted."
        )
    pages = []
    with pdfplumber.open(path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"[PDF] {total_pages} pages detected")
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text.strip())
            logger.debug(f"[PDF] Page {i+1}/{total_pages} extracted ({len(page_text or '')} chars)")
    logger.info(f"[PDF] Extraction complete — {len(pages)} non-empty pages, {sum(len(p) for p in pages)} total chars")
    return "\n\n".join(pages)


def ingest_pdf(file_path: str, channel_id: str, source_id: str = None, on_progress=None) -> tuple[int, str]:
    """
    Ingest a PDF from a local file path.
    source_id: stable identifier for dedup (use Slack file_id if available).
    on_progress: optional callable(message: str) for mid-ingest status updates.
    Returns (chunk_count, meeting_name).
    """
    t_start = time.time()
    logger.info(f"[PDF INGEST] Starting — file={file_path} channel={channel_id} source_id={source_id}")

    supabase = SupabaseService()

    user = supabase.get_user_by_username(channel_id)
    if not user:
        logger.info(f"[PDF INGEST] Creating new user for channel {channel_id}")
        user = supabase.create_user(channel_id)
    user_uuid = user["id"]
    logger.info(f"[PDF INGEST] User resolved: {user_uuid}")

    logger.info("[PDF INGEST] Extracting text from PDF...")
    text = _extract_pdf_text(file_path)
    if not text.strip():
        raise ValueError("No text could be extracted from this PDF.")
    logger.info(f"[PDF INGEST] Extracted {len(text)} chars")

    stable_key = source_id or file_path
    key_hash = hashlib.md5(stable_key.encode()).hexdigest()[:8]
    meeting_name = f"pdf_{key_hash}_{channel_id}"
    logger.info(f"[PDF INGEST] Meeting name: {meeting_name}")

    existing = supabase.get_meeting_by_name(meeting_name)
    if existing:
        logger.info(f"[PDF INGEST] Already ingested {meeting_name}, skipping.")
        return 0, meeting_name

    logger.info("[PDF INGEST] Creating meeting record...")
    m = supabase.create_meeting({
        "meeting_name": meeting_name,
        "user_id": user_uuid,
        "channel_id": channel_id,
        "run_id": meeting_name,
        "status": "ingested",
    })
    meeting_id = m["id"]
    logger.info(f"[PDF INGEST] Meeting created: {meeting_id}")

    supabase.upsert_transcript(meeting_id, text)
    logger.info("[PDF INGEST] Transcript stored")

    chunks = _chunk_text(text)
    logger.info(f"[PDF INGEST] {len(chunks)} chunks to embed")

    if on_progress:
        on_progress(f"Text extracted — embedding {len(chunks)} chunks in parallel...")

    chunk_rows = [None] * len(chunks)
    completed = [0]

    def embed_one(args):
        idx, chunk = args
        t = time.time()
        emb = get_embedding(chunk)
        completed[0] += 1
        logger.info(f"[PDF INGEST] Embedded chunk {completed[0]}/{len(chunks)} (idx={idx}) in {time.time()-t:.2f}s")
        return idx, {
            "meeting_id": meeting_id,
            "chunk_index": idx,
            "chunk_text": chunk,
            "embedding": emb,
            "source": "pdf",
            "topic": " ".join(chunk.split()[:6]),
        }

    logger.info("[PDF INGEST] Starting parallel embedding...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(embed_one, (i, c)) for i, c in enumerate(chunks)]
        for future in as_completed(futures):
            idx, row = future.result()
            chunk_rows[idx] = row

    logger.info("[PDF INGEST] All embeddings done — inserting to Supabase...")
    for i in range(0, len(chunk_rows), 20):
        supabase.insert_chunks(chunk_rows[i:i + 20])
        logger.info(f"[PDF INGEST] Inserted batch {i//20 + 1}")

    elapsed = time.time() - t_start
    logger.info(f"[PDF INGEST] Complete — {len(chunk_rows)} chunks in {elapsed:.1f}s")
    return len(chunk_rows), meeting_name
