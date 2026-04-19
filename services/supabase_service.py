import os
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()


class SupabaseService:

    # =========================================================
    # INIT
    # =========================================================

    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not url or not key:
            raise ValueError("Supabase credentials missing in .env")

        self.client: Client = create_client(url, key)

    # =========================================================
    # USERS
    # =========================================================

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        response = (
            self.client
            .table("users")
            .select("*")
            .eq("username", username)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    def create_user(self, username: str) -> Dict:
        response = (
            self.client
            .table("users")
            .insert({"username": username})
            .execute()
        )

        if not response.data:
            raise Exception("Failed to create user")

        return response.data[0]

    def get_meeting_by_run_id(self, run_id: str):
        result = (
            self.client
            .table("meetings")
            .select("*")
            .eq("run_id", run_id)
            .limit(1)
            .execute()
        )

        return result.data[0] if result.data else None

    def get_latest_meeting_by_user(self, user_id):
        result = (
            self.client.table("meetings")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        if result.data:
            return result.data[0]
        return None

    def get_recent_meetings(self, user_id: str, limit: int = 2) -> List[Dict]:
        result = (
            self.client.table("meetings")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit + 5)
            .execute()
        )
        meetings = []
        if result.data:
            for m in result.data:
                if not m["meeting_name"].startswith("slack_") and not m["meeting_name"].startswith("goal_"):
                    meetings.append(m)
                if len(meetings) == limit:
                    break
        return meetings

    def get_meeting_by_name(self, meeting_name: str) -> Optional[Dict]:
        response = (
            self.client
            .table("meetings")
            .select("*")
            .eq("meeting_name", meeting_name)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None

    # =========================================================
    # TRANSCRIPTS
    # =========================================================

    def transcript_exists(self, meeting_id: str) -> bool:
        response = (
            self.client
            .table("transcripts")
            .select("id")
            .eq("meeting_id", meeting_id)
            .limit(1)
            .execute()
        )
        return bool(response.data)

    def upsert_transcript(self, meeting_id: str, raw_text: str) -> None:
        self.client.table("transcripts").upsert({
            "meeting_id": meeting_id,
            "raw_text": raw_text
        }, on_conflict="meeting_id").execute()

    # =========================================================
    # CHUNKS
    # =========================================================

    def chunks_exist(self, meeting_id: str) -> bool:
        response = (
            self.client
            .table("chunks")
            .select("id")
            .eq("meeting_id", meeting_id)
            .limit(1)
            .execute()
        )
        return bool(response.data)

    def delete_chunks_by_meeting(self, meeting_id: str) -> None:
        self.client.table("chunks") \
            .delete() \
            .eq("meeting_id", meeting_id) \
            .execute()

    def insert_chunks(self, chunks: List[Dict]) -> None:
        if not chunks:
            return

        response = (
            self.client
            .table("chunks")
            .upsert(chunks, on_conflict="meeting_id,chunk_index")
            .execute()
        )

        if not response.data:
            raise Exception("Chunk insert failed")

    def get_adjacent_chunks(self, meeting_id: str, chunk_indices: List[int], window: int = 2) -> List[Dict]:
        """Fetch adjacent chunks (+/- window) for a given set of matched chunk indices."""
        if not chunk_indices:
            return []

        all_needed = set()
        for idx in chunk_indices:
            for j in range(idx - window, idx + window + 1):
                if j >= 0:
                    all_needed.add(j)

        response = (
            self.client
            .table("chunks")
            .select("*")
            .eq("meeting_id", meeting_id)
            .in_("chunk_index", list(all_needed))
            .execute()
        )
        clean_chunks = []
        for row in (response.data or []):
            if isinstance(row.get("chunk_index"), int) and isinstance(row.get("chunk_text"), str):
                clean_chunks.append({
                    "meeting_id": row["meeting_id"],
                    "chunk_index": row["chunk_index"],
                    "text": row["chunk_text"],
                    "similarity": 0.0,
                    "source": row.get("source", "transcript"),
                })
        return clean_chunks

    # =========================================================
    # VECTOR SEARCH
    # =========================================================

    def match_chunks_by_user(
        self,
        query_embedding: List[float],
        user_id: str,
        match_count: int = 5
    ) -> List[Dict[str, Any]]:

        response = self.client.rpc(
            "match_chunks_by_user",
            {
                "query_embedding": query_embedding,
                "match_count": match_count,
                "filter_user_id": user_id
            }
        ).execute()

        return response.data or []

    # =========================================================
    # BM25 KEYWORD SEARCH
    # =========================================================

    def match_chunks_bm25(
        self,
        query_text: str,
        user_id: str,
        match_count: int = 20
    ) -> List[Dict[str, Any]]:
        try:
            response = self.client.rpc(
                "match_chunks_bm25",
                {
                    "query_text": query_text,
                    "filter_user_id": user_id,
                    "match_count": match_count
                }
            ).execute()
            return response.data or []
        except Exception as e:
            print(f"[BM25] Search failed: {e}")
            return []

    # =========================================================
    # SESSIONS
    # =========================================================

    def create_session(self, session_id: str, user_id: str):
        self.client.table("sessions").insert({
            "session_id": session_id,
            "user_id": user_id
        }).execute()

    def end_session(self, session_id: str):
        self.client.table("sessions") \
            .update({"ended_at": "now()"}) \
            .eq("session_id", session_id) \
            .execute()

    def create_session_if_not_exists(self, session_id: str, user_id: str):
        existing = (
            self.client
            .table("sessions")
            .select("id")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        )

        if existing.data:
            return

        self.client.table("sessions").insert({
            "session_id": session_id,
            "user_id": user_id
        }).execute()

    # =========================================================
    # CHAT MEMORY
    # =========================================================

    def create_session_for_user(self, user_id):
        from uuid import uuid4
        session_id = str(uuid4())
        self.create_session(session_id, user_id)
        return session_id

    def create_meeting(self, data: dict):
        return self.client.table("meetings").insert(data).execute().data[0]

    def get_latest_meeting_by_channel(self, channel_id: str):
        result = (
            self.client
            .table("meetings")
            .select("*")
            .eq("channel_id", channel_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        return result.data[0] if result.data else None

    def update_meeting_status(self, run_id: str, status: str):
        return (
            self.client
            .table("meetings")
            .update({"status": status})
            .eq("run_id", run_id)
            .execute()
        )

    def get_slack_ingestion_status(self, channel_id: str) -> bool:
        """Check if Slack history has already been ingested for a channel."""
        meeting = self.get_meeting_by_name(f"slack_{channel_id}")
        if not meeting:
            return False
        return self.chunks_exist(meeting["id"])

    def save_chat_turn(
        self,
        session_id: str,
        user_id: str,
        question: str,
        answer: str,
        source: str,
        meeting_id: Optional[str] = None,
        method: Optional[str] = None,
        time_scope: Optional[str] = None,
    ):
        self.create_session_if_not_exists(session_id, user_id)

        payload = {
            "session_id": session_id,
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "source": source,
            "meeting_id": meeting_id,
            "method": method,
            "time_scope": time_scope,
        }

        try:
            self.client.table("chat_turns").insert(payload).execute()
            print("Chat turn saved.")
        except Exception as e:
            print(f"[SUPABASE CHAT SAVE ERROR] {e}")

    def get_recent_chat_turns(
        self,
        session_id: str,
        limit: int = 20
    ) -> List[Dict]:
        """
        Returns list of {"question": str, "answer": str} dicts in chronological order.
        FIX: Previously returned List[str] which broke query_understanding_agent and
        chat_answer_node context building.
        """
        response = (
            self.client.table("chat_turns")
            .select("question, answer")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )

        rows = response.data or []

        return [
            {
                "question": r.get("question", ""),
                "answer": r.get("answer", "")
            }
            for r in rows
        ]

    def get_recent_chat_turns_by_user(
        self,
        user_id: str,
        limit: int = 20
    ) -> List[str]:
        """
        Returns flat list of strings for the heartbeat context.
        Format: ["User: q", "AI: a", ...]
        """
        response = (
            self.client.table("chat_turns")
            .select("question, answer")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        rows = response.data or []
        rows = list(reversed(rows))  # Back to chronological order

        lines = []
        for r in rows:
            if r.get("question"):
                lines.append(f"User: {r['question']}")
            if r.get("answer"):
                lines.append(f"AI: {r['answer']}")

        return lines

    def append_live_chat_to_slack_chunk(self, user_id: str, question: str, answer: str):
        from services.embedding_api import get_embedding
        from datetime import datetime

        # Use channel_id-based meeting name (channel_id is stored as username in users table)
        user_resp = self.client.table("users").select("username").eq("id", user_id).limit(1).execute()
        channel_id = user_resp.data[0]["username"] if user_resp.data else user_id

        meeting_name = f"slack_{channel_id}"
        meeting = self.get_meeting_by_name(meeting_name)
        if not meeting:
            insert_resp = self.client.table("meetings").insert({
                "meeting_name": meeting_name,
                "user_id": user_id,
                "status": "completed"
            }).execute()
            if not insert_resp.data:
                return
            meeting = insert_resp.data[0]

        meeting_id = meeting["id"]
        today_date = datetime.now().strftime("%Y-%m-%d")
        new_text_to_append = f"[User]: {question}\n[MMA AGENT]: {answer}\n"

        recent_chunk_resp = self.client.table("chunks") \
            .select("*") \
            .eq("meeting_id", meeting_id) \
            .eq("source", "slack") \
            .order("chunk_index", desc=True) \
            .limit(1) \
            .execute()

        recent_chunk = recent_chunk_resp.data[0] if recent_chunk_resp.data else None

        needs_new_chunk = True

        if recent_chunk and recent_chunk.get("chunk_text"):
            text = recent_chunk["chunk_text"]
            if f"--- {today_date} ---" in text:
                current_words = len(text.split())
                new_words = len(new_text_to_append.split())
                if current_words + new_words <= 300:
                    needs_new_chunk = False
                    updated_text = text + "\n" + new_text_to_append

                    try:
                        new_embedding = get_embedding(updated_text)
                        self.client.table("chunks").update({
                            "chunk_text": updated_text,
                            "embedding": new_embedding
                        }).eq("id", recent_chunk["id"]).execute()
                        print("[AUTO-EMBED] Appended live chat to existing chunk.")
                    except Exception as e:
                        print(f"[AUTO-EMBED] Failed to update: {e}")
                    return

        if needs_new_chunk:
            next_idx = 0
            if recent_chunk:
                next_idx = recent_chunk.get("chunk_index", -1) + 1

            new_chunk_text = f"--- {today_date} ---\n\n{new_text_to_append}"
            try:
                new_embedding = get_embedding(new_chunk_text)
                self.client.table("chunks").insert({
                    "meeting_id": meeting_id,
                    "chunk_index": next_idx,
                    "chunk_text": new_chunk_text,
                    "embedding": new_embedding,
                    "source": "slack"
                }).execute()
                print("[AUTO-EMBED] Created new live chat chunk.")
            except Exception as e:
                print(f"[AUTO-EMBED] Failed to insert new chunk: {e}")

    # =========================================================
    # EVENT DEDUPLICATION (Serverless-safe)
    # Requires table: CREATE TABLE slack_events (event_id TEXT PRIMARY KEY, created_at TIMESTAMPTZ DEFAULT NOW());
    # =========================================================

    def is_event_processed(self, event_id: str) -> bool:
        """
        Check + mark an event as processed atomically.
        Returns True if already processed (duplicate), False if new (should process).
        Uses INSERT with ON CONFLICT to make it atomic and serverless-safe.
        Falls back gracefully if table doesn't exist.
        """
        try:
            self.client.table("slack_events").insert(
                {"event_id": event_id}
            ).execute()
            return False  # Successfully inserted → new event, process it
        except Exception as e:
            err_str = str(e)
            if "duplicate" in err_str.lower() or "unique" in err_str.lower() or "23505" in err_str:
                return True  # Duplicate key → already processed
            # Table doesn't exist or other error → allow processing (fail open)
            print(f"[EVENT DEDUP] Fallback (table missing?): {e}")
            return False


# =========================================================
# GLOBAL SUPABASE SINGLETON
# =========================================================

_SUPABASE_INSTANCE = None

def get_supabase_client() -> SupabaseService:
    global _SUPABASE_INSTANCE

    if _SUPABASE_INSTANCE is None:
        _SUPABASE_INSTANCE = SupabaseService()

    return _SUPABASE_INSTANCE