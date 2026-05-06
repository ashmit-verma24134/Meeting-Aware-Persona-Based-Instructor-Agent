"""
Ingest Slack channel history into Supabase as embedded chunks.
Groups messages by day — one chunk per calendar date.

Goal chunk is updated HERE (on /sync_slack) — NOT on meeting ingest.
This is the correct place because:
- Slack reflects real-time team activity and decisions
- Meeting ingest chunks are mid-deletion during ingest, making goal reads unsafe
- Goal should reflect what the team is actually doing day-to-day

Usage:
    from scripts.ingest_slack_to_supabase import ingest_slack_history
    ingest_slack_history(channel_id="C0123456789")
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from services.slack_history_service import SlackHistoryService
from services.supabase_service import SupabaseService
from services.embedding_api import get_embedding
from scripts.ingest_to_supabase import update_dynamic_project_goal

load_dotenv()

# ── In-process dedup guard to prevent double goal update if called twice fast ──
_SYNC_IN_PROGRESS = set()


def ingest_slack_history(channel_id: str, limit: int = 5000):
    """
    Fetch Slack channel messages → group by day → embed → store in Supabase.
    One chunk per calendar day (e.g. all March 22 messages = one chunk).
    Updates the dynamic project goal after ingestion.
    """

    # ── Dedup guard: prevent double execution for same channel ──
    if channel_id in _SYNC_IN_PROGRESS:
        print(f"[SLACK INGEST] Sync already in progress for {channel_id}, skipping.")
        return
    _SYNC_IN_PROGRESS.add(channel_id)

    try:
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

        # ── Group into small chunks ──
        small_chunks = slack_service.messages_to_small_chunks(messages, max_words=300)

        if not small_chunks:
            print("[SLACK INGEST] No chunks generated. Skipping.")
            return

        print(f"[SLACK INGEST] {len(small_chunks)} small chunks generated")

        # ── SIRF RECENT 60 CHUNKS — timeout fix ──
        if len(small_chunks) > 60:
            print(f"[SLACK INGEST] Trimming to last 60 chunks (was {len(small_chunks)})")
            small_chunks = small_chunks[-60:]

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
        full_text = "\n\n".join(c["text"] for c in small_chunks)
        supabase.upsert_transcript(meeting_id, full_text)

        # ── Embed each chunk in parallel ──
        chunk_rows = [None] * len(small_chunks)

        def embed_one(args):
            idx, chunk = args
            embedding = get_embedding(chunk["text"])
            print(f"[SLACK INGEST] Embedded chunk {idx} for date {chunk.get('date', 'unknown')}")
            return idx, {
                "meeting_id": meeting_id,
                "chunk_index": idx,
                "chunk_text": chunk["text"],
                "embedding": embedding,
                "source": "slack",
            }

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(embed_one, (i, c)) for i, c in enumerate(small_chunks)]
            for future in as_completed(futures):
                idx, row = future.result()
                chunk_rows[idx] = row

        # ── Batch insert every 20 chunks ──
        BATCH_SIZE = 20
        for i in range(0, len(chunk_rows), BATCH_SIZE):
            batch = chunk_rows[i:i + BATCH_SIZE]
            supabase.insert_chunks(batch)
            print(f"[SLACK INGEST] Inserted batch {i//BATCH_SIZE + 1} ({len(batch)} chunks)")

        print(f"[SLACK INGEST] Inserted {len(chunk_rows)} slack chunks total.")

        # ── Update goal AFTER all chunks are safely inserted ──
        print("[SLACK INGEST] Updating dynamic project goal from Slack activity...")
        recent_slack_text = "\n\n".join(c["text"] for c in small_chunks[-10:])
        update_dynamic_project_goal(supabase, user_uuid, recent_slack_text)

        print("[SLACK INGEST] Done.")

    finally:
        # Always release the lock, even if something crashed
        _SYNC_IN_PROGRESS.discard(channel_id)


# ───────────────────────────────────────
# CLI Runner
# ───────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m scripts.ingest_slack_to_supabase <channel_id>")
        sys.exit(1)

    ingest_slack_history(channel_id=sys.argv[1])