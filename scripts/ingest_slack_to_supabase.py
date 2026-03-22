"""
Ingest Slack channel history into Supabase as embedded chunks.

Usage:
    from scripts.ingest_slack_to_supabase import ingest_slack_history
    ingest_slack_history(channel_id="C0123456789")
"""

import os
from dotenv import load_dotenv

from services.slack_history_service import SlackHistoryService
from services.supabase_service import SupabaseService
from services.embedding_api import get_embedding
from scripts.embedding_utils import build_embedding_text

load_dotenv()

CHUNK_SIZE = 350
OVERLAP = 50


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))

    return chunks


def ingest_slack_history(channel_id: str, limit: int = 500):
    """
    Fetch Slack channel messages → chunk → embed → store in Supabase.

    Uses the same `chunks` table as transcripts, with source="slack".
    Linked to a meeting record with meeting_name="slack_{channel_id}".
    """

    print(f"\n[SLACK INGEST] Fetching history for channel: {channel_id}")

    # ─── Services ───
    slack_service = SlackHistoryService()
    supabase = SupabaseService()

    # ─── Fetch messages ───
    messages = slack_service.fetch_channel_history(
        channel_id=channel_id,
        limit=limit,
    )

    if not messages:
        print("[SLACK INGEST] No messages found. Skipping.")
        return

    print(f"[SLACK INGEST] Fetched {len(messages)} messages")

    # ─── Build text blob ───
    full_text = slack_service.messages_to_text(messages)

    if not full_text.strip():
        print("[SLACK INGEST] Empty text after conversion. Skipping.")
        return

    # ─── Ensure user exists ───
    user = supabase.get_user_by_username(channel_id)
    if not user:
        user = supabase.create_user(channel_id)

    user_uuid = user["id"]

    # ─── Ensure "meeting" record for Slack history ───
    meeting_name = f"slack_{channel_id}"
    meeting = supabase.get_meeting_by_name(meeting_name)

    if meeting:
        meeting_id = meeting["id"]
        print("[SLACK INGEST] Slack meeting record exists. Re-ingesting...")
        # Delete old chunks so we can re-ingest fresh
        supabase.delete_chunks_by_meeting(meeting_id)
    else:
        meeting = supabase.create_meeting({
            "meeting_name": meeting_name,
            "user_id": user_uuid,
            "channel_id": channel_id,
            "run_id": f"slack_{channel_id}",
            "status": "ingested",
        })
        meeting_id = meeting["id"]
        print("[SLACK INGEST] Created new slack meeting record.")

    # ─── Upsert transcript (raw text) ───
    supabase.upsert_transcript(meeting_id, full_text)

    # ─── Chunk ───
    chunks = chunk_text(full_text, CHUNK_SIZE, OVERLAP)
    print(f"[SLACK INGEST] Generated {len(chunks)} chunks")

    # ─── Embed & prepare rows ───
    prev_chunk = None
    chunk_rows = []

    for idx, chunk in enumerate(chunks):
        embedding_text = build_embedding_text(
            {"text": chunk},
            prev_chunk,
        )

        embedding = get_embedding(embedding_text)

        chunk_rows.append({
            "meeting_id": meeting_id,
            "chunk_index": idx,
            "chunk_text": chunk,
            "embedding": embedding,
            "source": "slack",
        })

        prev_chunk = {"text": chunk}

    # ─── Insert ───
    if chunk_rows:
        supabase.insert_chunks(chunk_rows)
        print(f"[SLACK INGEST] Inserted {len(chunk_rows)} chunks.")

    print("[SLACK INGEST] Done.")


# ───────────────────────────────────────
# CLI Runner
# ───────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m scripts.ingest_slack_to_supabase <channel_id>")
        sys.exit(1)

    ingest_slack_history(channel_id=sys.argv[1])
