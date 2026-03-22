import os
from slack_sdk import WebClient
from dotenv import load_dotenv

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_BOT_NAME = os.getenv("SLACK_BOT_NAME", "MMA AGENT")  # add this to .env or keep default


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
        Includes bot's own replies so full conversation is stored.
        Filters out only system events and other bots.
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

                # Always skip system events
                if msg.get("subtype") in [
                    "channel_join",
                    "channel_leave",
                    "channel_topic",
                    "channel_purpose",
                    "message_changed",
                    "message_deleted",
                ]:
                    continue

                # Allow OUR bot's messages through
                # Block only other/external bots
                if msg.get("bot_id"):
                    bot_name = (msg.get("username") or "").strip()
                    if bot_name != SLACK_BOT_NAME:
                        continue  # skip other bots, keep ours

                text = (msg.get("text") or "").strip()
                if not text:
                    continue

                # Label user vs bot for context clarity
                if msg.get("bot_id"):
                    speaker = f"BOT({SLACK_BOT_NAME})"
                else:
                    speaker = msg.get("user", "unknown")

                messages.append({
                    "text": text,
                    "user": speaker,
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