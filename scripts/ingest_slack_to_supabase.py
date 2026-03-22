"""
Ingest Slack channel history into Supabase as embedded chunks.
Groups messages by day — one chunk per calendar date.

Usage:
    from scripts.ingest_slack_to_supabase import ingest_slack_history
    ingest_slack_history(channel_id="C0123456789")
"""

import os
from dotenv import load_dotenv

from services.slack_history_service import SlackHistoryService
from services.supabase_service import SupabaseService
from services.embedding_api import get_embedding

load_dotenv()


def ingest_slack_history(channel_id: str, limit: int = 500):
    """
    Fetch Slack channel messages → group by day → embed → store in Supabase.
    One chunk per calendar day (e.g. all March 22 messages = one chunk).
    """

    print(f"\n[SLACK INGEST] Fetching history for channel: {channel_id}")

    slack_service = SlackHistoryService()
    supabase = SupabaseService()

    # ── Fetch all messages ──
    messages = slack_service.fetch_channel_history(
        channel_id=channel_id,
        limit=limit,
    )

    if not messages:
        print("[SLACK INGEST] No messages found. Skipping.")
        return

    print(f"[SLACK INGEST] Fetched {len(messages)} messages")

    # ── Group into daily chunks ──
    daily_chunks = slack_service.messages_to_daily_chunks(messages)

    if not daily_chunks:
        print("[SLACK INGEST] No daily chunks generated. Skipping.")
        return

    print(f"[SLACK INGEST] {len(daily_chunks)} daily chunks (one per day)")

    # ── Ensure user exists ──
    user = supabase.get_user_by_username(channel_id)
    if not user:
        user = supabase.create_user(channel_id)

    user_uuid = user["id"]

    # ── Ensure meeting record for this Slack channel ──
    meeting_name = f"slack_{channel_id}"
    meeting = supabase.get_meeting_by_name(meeting_name)

    if meeting:
        meeting_id = meeting["id"]
        print("[SLACK INGEST] Slack meeting record exists. Re-ingesting...")
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

    # ── Upsert raw transcript (full text blob for reference) ──
    full_text = "\n\n".join(c["text"] for c in daily_chunks)
    supabase.upsert_transcript(meeting_id, full_text)

    # ── Embed each daily chunk and insert ──
    chunk_rows = []

    for idx, chunk in enumerate(daily_chunks):
        embedding_text = chunk["text"]
        embedding = get_embedding(embedding_text)

        chunk_rows.append({
            "meeting_id": meeting_id,
            "chunk_index": idx,
            "chunk_text": chunk["text"],
            "embedding": embedding,
            "source": "slack",
        })

        print(f"[SLACK INGEST] Embedded day {chunk['date']} (chunk {idx})")

    if chunk_rows:
        supabase.insert_chunks(chunk_rows)
        print(f"[SLACK INGEST] Inserted {len(chunk_rows)} daily chunks.")

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