import os
from datetime import datetime, timezone
from collections import defaultdict
from slack_sdk import WebClient
from dotenv import load_dotenv

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")


class SlackHistoryService:
    """Fetch and structure Slack channel message history."""

    def __init__(self):
        if not SLACK_BOT_TOKEN:
            raise RuntimeError("SLACK_BOT_TOKEN not set")
        self.client = WebClient(token=SLACK_BOT_TOKEN)

    # ─────────────────────────────────────
    # Fetch channel messages
    # ─────────────────────────────────────

    def fetch_channel_history(
        self,
        channel_id: str,
        limit: int = 500,
    ) -> list[dict]:
        """
        Fetch up to `limit` messages from a Slack channel.
        Returns list of {text, user, timestamp} dicts.
        Stores ALL messages — user + bot.
        Only skips structural system events.
        """
        messages = []
        cursor = None

        while len(messages) < limit:
            batch_size = min(200, limit - len(messages))

            response = self.client.conversations_history(
                channel=channel_id,
                limit=batch_size,
                cursor=cursor,
            )

            for msg in response.get("messages", []):

                # Skip ONLY system/structural events
                if msg.get("subtype") in [
                    "channel_join",
                    "channel_leave",
                    "channel_topic",
                    "channel_purpose",
                    "message_changed",
                    "message_deleted",
                    "thread_broadcast",
                ]:
                    continue

                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                # Clean Slack mention formatting
                import re
                text = re.sub(r'<@[^>]+>', '', text).strip()
                text = re.sub(r'<!channel>', '@channel', text).strip()
                text = re.sub(r'<!here>', '@here', text).strip()
                if not text:
                    continue

                # Label speaker clearly
                if msg.get("bot_id"):
                    speaker = msg.get("username") or "MMA AGENT"
                else:
                    user_id = msg.get("user", "unknown")
                    try:
                        user_info = self.client.users_info(user=user_id)
                        speaker = (
                            user_info["user"]["profile"].get("display_name")
                            or user_info["user"]["profile"].get("real_name")
                            or user_id
                        )
                    except Exception:
                        speaker = user_id

                # Collect PDF file attachments for later ingestion
                pdf_files = [
                    {
                        "file_id": f.get("id"),
                        "name": f.get("name", "file.pdf"),
                        "download_url": f.get("url_private_download") or f.get("url_private"),
                    }
                    for f in msg.get("files", [])
                    if f.get("filetype") == "pdf" and (
                        f.get("url_private_download") or f.get("url_private")
                    )
                ]

                messages.append({
                    "text": text,
                    "user": speaker,
                    "timestamp": msg.get("ts", ""),
                    "pdf_files": pdf_files,
                })

            # Pagination
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        # Return in chronological order (oldest first)
        messages.reverse()
        return messages

    # ─────────────────────────────────────
    # Group messages by day → one chunk per day
    # ─────────────────────────────────────

    def messages_to_daily_chunks(self, messages: list[dict]) -> list[dict]:
        """
        Group messages by calendar date (UTC).
        Returns list of {date, text} dicts — one per day.
        Each day's chunk = all messages from that day joined together.
        """
        daily = defaultdict(list)

        for msg in messages:
            ts = msg.get("timestamp", "")
            try:
                dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                date_key = dt.strftime("%Y-%m-%d")  # e.g. "2026-03-22"
            except Exception:
                date_key = "unknown"

            user = msg.get("user", "unknown")
            text = msg.get("text", "")
            daily[date_key].append(f"[{user}]: {text}")

        # Sort dates chronologically
        sorted_dates = sorted(daily.keys())

        chunks = []
        for date in sorted_dates:
            chunk_text = f"--- {date} ---\n" + "\n\n".join(daily[date])
            chunks.append({
                "date": date,
                "text": chunk_text,
            })

        return chunks

    # ─────────────────────────────────────
    # Build text pieces split by word limit
    # ─────────────────────────────────────
    def messages_to_small_chunks(self, messages: list[dict], max_words: int = 300) -> list[dict]:
        """
        Group messages by calendar date (UTC), then split into smaller chunks (under max_words).
        Each chunk starts with the date header to preserve date filtering.
        """
        daily = defaultdict(list)

        for msg in messages:
            ts = msg.get("timestamp", "")
            try:
                dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                date_key = dt.strftime("%Y-%m-%d")
            except Exception:
                date_key = "unknown"

            user = msg.get("user", "unknown")
            text = msg.get("text", "")
            daily[date_key].append(f"[{user}]: {text}")

        chunks = []
        for date in sorted(daily.keys()):
            current_messages = []
            current_words = 0

            for msg_line in daily[date]:
                words_in_msg = len(msg_line.split())

                # Single message larger than max_words — flush & split it directly
                if words_in_msg > max_words:
                    if current_messages:
                        chunk_text = f"--- {date} ---\n" + "\n\n".join(current_messages)
                        chunks.append({"date": date, "text": chunk_text})
                        current_messages = []
                        current_words = 0
                    msg_words = msg_line.split()
                    for start in range(0, len(msg_words), max_words):
                        sub = " ".join(msg_words[start:start + max_words])
                        chunks.append({"date": date, "text": f"--- {date} ---\n{sub}"})
                    continue

                if current_words + words_in_msg > max_words and current_messages:
                    # Finalize current chunk
                    chunk_text = f"--- {date} ---\n" + "\n\n".join(current_messages)
                    chunks.append({
                        "date": date,
                        "text": chunk_text,
                    })
                    current_messages = []
                    current_words = 0

                current_messages.append(msg_line)
                current_words += words_in_msg

            # Append the remainder
            if current_messages:
                chunk_text = f"--- {date} ---\n" + "\n\n".join(current_messages)
                chunks.append({
                    "date": date,
                    "text": chunk_text,
                })

        return chunks

    # ─────────────────────────────────────
    # Build full text blob (kept for compatibility)
    # ─────────────────────────────────────

    def messages_to_text(self, messages: list[dict]) -> str:
        """
        Convert list of Slack messages into a single text blob.
        """
        lines = []
        for msg in messages:
            user = msg.get("user", "unknown")
            text = msg.get("text", "")
            lines.append(f"[{user}]: {text}")

        return "\n\n".join(lines)