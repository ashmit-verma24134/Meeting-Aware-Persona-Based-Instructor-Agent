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
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

import requests as _requests

from services.slack_history_service import SlackHistoryService
from services.supabase_service import SupabaseService
from services.embedding_api import get_embedding
from scripts.ingest_to_supabase import update_dynamic_project_goal, MAX_CHUNK_WORDS

load_dotenv()

# ── In-process dedup guard to prevent double goal update if called twice fast ──
_SYNC_IN_PROGRESS = set()

_GDOC_ID_PATTERN = re.compile(r'https://docs\.google\.com/document/d/([a-zA-Z0-9_-]+)')

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")


def _ingest_google_docs_from_messages(
    messages: list[dict],
    channel_id: str,
    supabase: SupabaseService,
    user_uuid: str,
) -> int:
    """
    Scan Slack messages for Google Doc URLs, fetch each (public export),
    chunk-embed-store with source="gdoc".  Returns total chunks inserted.
    """
    seen_ids: set[str] = set()
    total = 0

    for msg in messages:
        for doc_id in _GDOC_ID_PATTERN.findall(msg.get("text", "")):
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
            try:
                resp = _requests.get(export_url, allow_redirects=True, timeout=30)

                if resp.status_code == 403:
                    print(f"[GDOC INGEST] Doc {doc_id} is private — skipping. Make it 'Anyone with link can view' to ingest it.")
                    continue

                resp.raise_for_status()
                doc_text = resp.text.strip()

                if not doc_text:
                    print(f"[GDOC INGEST] Doc {doc_id} returned empty content, skipping.")
                    continue

                print(f"[GDOC INGEST] Fetched doc {doc_id} ({len(doc_text)} chars)")

                meeting_name = f"gdoc_{doc_id}_{channel_id}"
                existing = supabase.get_meeting_by_name(meeting_name)
                if existing:
                    meeting_id = existing["id"]
                    supabase.delete_chunks_by_meeting(meeting_id)
                else:
                    m = supabase.create_meeting({
                        "meeting_name": meeting_name,
                        "user_id": user_uuid,
                        "channel_id": channel_id,
                        "run_id": meeting_name,
                        "status": "ingested",
                    })
                    meeting_id = m["id"]

                supabase.upsert_transcript(meeting_id, doc_text)

                words = doc_text.split()
                chunk_rows = []
                for start in range(0, len(words), MAX_CHUNK_WORDS):
                    chunk_text = " ".join(words[start:start + MAX_CHUNK_WORDS])
                    if not chunk_text.strip():
                        continue
                    emb = get_embedding(chunk_text)
                    chunk_rows.append({
                        "meeting_id": meeting_id,
                        "chunk_index": len(chunk_rows),
                        "chunk_text": chunk_text,
                        "embedding": emb,
                        "source": "gdoc",
                        "topic": " ".join(chunk_text.split()[:6]),
                    })

                for i in range(0, len(chunk_rows), 20):
                    supabase.insert_chunks(chunk_rows[i:i + 20])

                total += len(chunk_rows)
                print(f"[GDOC INGEST] Inserted {len(chunk_rows)} chunks for {meeting_name}")

            except Exception as e:
                print(f"[GDOC INGEST] Failed for doc {doc_id}: {e}")

    return total


def _ingest_pdfs_from_messages(
    messages: list[dict],
    channel_id: str,
) -> int:
    """
    Scan Slack messages for PDF file attachments (captured during history fetch),
    download each via bot token, and ingest. Skips files already ingested.
    Returns total chunks inserted.
    """
    import tempfile
    from scripts.ingest_pdf import ingest_pdf

    if not SLACK_BOT_TOKEN:
        print("[SLACK PDF INGEST] SLACK_BOT_TOKEN not set — skipping PDF ingest.")
        return 0

    seen_ids: set[str] = set()
    total = 0

    for msg in messages:
        for pdf_file in msg.get("pdf_files", []):
            file_id = pdf_file.get("file_id")
            download_url = pdf_file.get("download_url")
            if not file_id or not download_url or file_id in seen_ids:
                continue
            seen_ids.add(file_id)

            try:
                headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
                resp = _requests.get(download_url, headers=headers, timeout=60)
                resp.raise_for_status()

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(resp.content)
                tmp.close()

                count, meeting_name = ingest_pdf(tmp.name, channel_id, source_id=file_id)

                import os as _os
                _os.unlink(tmp.name)

                total += count
                print(f"[SLACK PDF INGEST] {pdf_file.get('name')} → {meeting_name} ({count} chunks)")

            except Exception as e:
                print(f"[SLACK PDF INGEST] Failed for file {file_id}: {e}")

    return total


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

        # ── Ingest PDFs attached in historical Slack messages ──
        pdf_count = _ingest_pdfs_from_messages(messages, channel_id)
        if pdf_count:
            print(f"[SLACK INGEST] Ingested {pdf_count} PDF chunks from history.")

        # ── Detect + ingest Google Docs linked in Slack messages ──
        gdoc_count = _ingest_google_docs_from_messages(messages, channel_id, supabase, user_uuid)
        if gdoc_count:
            print(f"[SLACK INGEST] Ingested {gdoc_count} Google Doc chunks.")

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