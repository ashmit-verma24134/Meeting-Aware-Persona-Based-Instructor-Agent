"""
Ingest a web page URL into Supabase as embedded chunks.
Chunks labeled source="link" are retrievable in normal Q&A flow.

Google Docs: only public docs supported ("Anyone with link can view").
Uses the plain-text export URL — no API key or OAuth needed.

Usage (slash command):  /link <url>
"""

import hashlib
import re

import requests
from bs4 import BeautifulSoup

from services.supabase_service import SupabaseService
from services.embedding_api import get_embedding

MAX_CHUNK_WORDS = 400
REQUEST_TIMEOUT = 30
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MMABot/1.0)"}

_GDOC_PATTERN = re.compile(r'https://docs\.google\.com/document/d/([a-zA-Z0-9_-]+)')


def _chunk_text(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
        if words[i:i + max_words]
    ]


def _fetch_url_text(url: str) -> str:
    gdoc_match = _GDOC_PATTERN.search(url)
    if gdoc_match:
        doc_id = gdoc_match.group(1)
        export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
        resp = requests.get(export_url, allow_redirects=True, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 403:
            raise ValueError(
                "This Google Doc is private. Set it to 'Anyone with the link can view' and try again."
            )
        resp.raise_for_status()
        text = resp.text.strip()
        if not text:
            raise ValueError("Google Doc returned empty content.")
        return text

    # Regular web page — strip boilerplate with BeautifulSoup
    resp = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()]
    return "\n".join(lines)


def ingest_url(url: str, channel_id: str) -> tuple[int, str]:
    """
    Fetch and ingest a web page.
    Returns (chunk_count, meeting_name).
    """
    supabase = SupabaseService()

    user = supabase.get_user_by_username(channel_id)
    if not user:
        user = supabase.create_user(channel_id)
    user_uuid = user["id"]

    text = _fetch_url_text(url)
    if not text.strip():
        raise ValueError("No content could be extracted from this URL.")

    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    meeting_name = f"link_{url_hash}_{channel_id}"

    existing = supabase.get_meeting_by_name(meeting_name)
    if existing:
        meeting_id = existing["id"]
        supabase.delete_chunks_by_meeting(meeting_id)
        print(f"[URL INGEST] Re-ingesting existing record {meeting_name}")
    else:
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
    print(f"[URL INGEST] {len(chunks)} chunks from {url[:80]}")

    chunk_rows = []
    for idx, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        chunk_rows.append({
            "meeting_id": meeting_id,
            "chunk_index": idx,
            "chunk_text": chunk,
            "embedding": emb,
            "source": "link",
            "topic": " ".join(chunk.split()[:6]),
        })

    for i in range(0, len(chunk_rows), 20):
        supabase.insert_chunks(chunk_rows[i:i + 20])

    print(f"[URL INGEST] Inserted {len(chunk_rows)} chunks as {meeting_name}")
    return len(chunk_rows), meeting_name
