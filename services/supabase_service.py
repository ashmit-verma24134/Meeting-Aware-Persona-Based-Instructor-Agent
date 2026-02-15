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

    # =========================================================
    # MEETINGS
    # =========================================================

    def create_meeting(self, meeting_name: str, user_id: str) -> Dict:
        response = (
            self.client
            .table("meetings")
            .insert({
                "meeting_name": meeting_name,
                "user_id": user_id
            })
            .execute()
        )

        if not response.data:
            raise Exception("Failed to create meeting")

        return response.data[0]

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
        }).execute()

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
            .insert(chunks)
            .execute()
        )

        if not response.data:
            raise Exception("Chunk insert failed")

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
    # CHAT MEMORY (PRODUCTION REPLACEMENT)
    # =========================================================
    def create_session_for_user(self, user_id):
        from uuid import uuid4
        session_id = str(uuid4())
        self.create_session(session_id, user_id)
        return session_id

    def save_chat_turn(
        self,
        session_id: str,
        user_id: str,
        question: str,
        answer: str,
        source: str,              # "chat" | "system"
        meeting_id: Optional[str] = None,
        method: Optional[str] = None,
        time_scope: Optional[str] = None,
    ):

        # Ensure session exists
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
    ) -> List[str]:

        response = (
            self.client.table("chat_turns")  
            .select("question, answer")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )

        rows = response.data or []

        lines = []
        for r in rows:
            if r.get("question"):
                lines.append(f"User: {r['question']}")
            if r.get("answer"):
                lines.append(f"AI: {r['answer']}")

        return lines
# =========================================================
# GLOBAL SUPABASE SINGLETON
# =========================================================

_SUPABASE_INSTANCE = None

def get_supabase_client() -> SupabaseService:
    global _SUPABASE_INSTANCE

    if _SUPABASE_INSTANCE is None:
        _SUPABASE_INSTANCE = SupabaseService()

    return _SUPABASE_INSTANCE
