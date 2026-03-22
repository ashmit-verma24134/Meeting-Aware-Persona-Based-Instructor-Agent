import os
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
        Filters out bot messages and system join/leave messages.
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
                # Skip bot messages
                if msg.get("bot_id") or msg.get("subtype") in [
                    "bot_message",
                    "channel_join",
                    "channel_leave",
                    "channel_topic",
                    "channel_purpose",
                ]:
                    continue

                text = (msg.get("text") or "").strip()
                if not text:
                    continue

                messages.append({
                    "text": text,
                    "user": msg.get("user", "unknown"),
                    "timestamp": msg.get("ts", ""),
                })

            # Pagination
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        # Return in chronological order (oldest first)
        messages.reverse()
        return messages

    # ─────────────────────────────────────
    # Build text blob from messages
    # ─────────────────────────────────────

    def messages_to_text(self, messages: list[dict]) -> str:
        """
        Convert list of Slack messages into a single text blob
        suitable for chunking and embedding.
        """
        lines = []
        for msg in messages:
            user = msg.get("user", "unknown")
            text = msg.get("text", "")
            lines.append(f"[{user}]: {text}")

        return "\n\n".join(lines)
