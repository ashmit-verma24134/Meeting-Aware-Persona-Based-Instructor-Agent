import json
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict


class SessionMemory:
    """
    GOOGLE-ALIGNED SHORT-TERM SESSION MEMORY

    PRINCIPLES (per Google docs):
    - Short-term memory = conversational context, NOT evidence
    - Memory dominance only for follow-ups / clarification
    - No intent inference, no domain inference
    - Stateless-safe: memory can be externalized
    """

    def __init__(
        self,
        storage_path: str = "data/session_history.json",
        ttl_minutes: int = 120,
        max_turns_per_session: int = 50,
    ):
        self.storage_path = storage_path
        self.ttl = timedelta(minutes=ttl_minutes)
        self.max_turns = max_turns_per_session
        self.sessions: Dict[str, List[Dict]] = self._load_from_disk()

    # -------------------------------------------------
    # Disk I/O
    # -------------------------------------------------

    def _load_from_disk(self) -> Dict[str, List[Dict]]:
        if not os.path.exists(self.storage_path):
            return {}

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            print("⚠️ session_history.json corrupted. Resetting.")

        return {}

    def _save_to_disk(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.sessions, f, indent=2)

    # -------------------------------------------------
    # Write API
    # -------------------------------------------------

    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        *,
        source: str,  # "chat" | "retrieval" | "summary"
        method: Optional[str] = None,
        standalone_query: Optional[str] = None,
        time_scope: Optional[str] = None,
        meeting_index: Optional[int] = None,
        meeting_indices: Optional[List[int]] = None,
    ):
        """
        Adds a conversational turn.

        RULES:
        - source="chat" → conversational context only
        - retrieval/summary → NOT reused as chat evidence
        """

        now = datetime.now(timezone.utc)

        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append(
            {
                "question": question,
                "standalone_query": standalone_query,
                "answer": answer,
                "source": source,
                "meeting_index": meeting_index,
                "meeting_indices": meeting_indices,
                "time_scope": time_scope,
                "method": method,
                "timestamp": now.isoformat(),
            }
        )

        # Enforce windowing (Google short-term memory guidance)
        self.sessions[session_id] = self.sessions[session_id][-self.max_turns :]

        self._save_to_disk()

    # -------------------------------------------------
    # Read APIs (SAFE)
    # -------------------------------------------------

    def _is_fresh(self, ts: str) -> bool:
        try:
            t = datetime.fromisoformat(ts)
            return datetime.now(timezone.utc) - t <= self.ttl
        except Exception:
            return False

    def get_recent_context(self, session_id: str, k: int = 4) -> List[Dict]:
        """
        Recent conversational turns (fresh only).
        Used for pronouns / follow-ups.
        """
        if session_id not in self.sessions:
            return []

        fresh = [
            t
            for t in self.sessions[session_id]
            if self._is_fresh(t.get("timestamp", ""))
        ]
        return fresh[-k:]

    # def get_full_session(self, session_id: str) -> List[Dict]:
    #     if session_id not in self.sessions:
    #         return []
    #     return list(self.sessions[session_id])

    # -------------------------------------------------
    # 🔥 CHAT DOMINANCE CHECK (GOOGLE-CORRECT)
    # -------------------------------------------------

    def chat_can_answer(self, question: str, recent_context: List[Dict]) -> bool:
        """
        Determines whether CHAT_ONLY is allowed.

        HARD RULES:
        - Question is short or referential
        - Refers to something in recent chat
        - No new entities introduced
        """

        if not recent_context:
            return False

        q_tokens = set(re.findall(r"\w+", question.lower()))
        if not q_tokens:
            return False

        # Very short follow-ups ("explain more", "what about that")
        if len(q_tokens) <= 4:
            return True

        context_text = " ".join(
            f"{t.get('question','')} {t.get('answer','')}"
            for t in recent_context
            if t.get("source") == "chat"
        ).lower()

        overlap = q_tokens & set(re.findall(r"\w+", context_text))

        # Require meaningful overlap
        return len(overlap) >= 2

    # -------------------------------------------------
    # Chat retriever (used ONLY in CHAT_ONLY path)
    # -------------------------------------------------

    def retrieve_chat_chunks(
        self,
        session_id: str,
        query: str,
        k: int = 3,
    ) -> List[str]:
        """
        Lightweight chat retriever.

        GUARANTEES:
        - source == "chat" only
        - NOT factual grounding
        """

        if session_id not in self.sessions:
            return []

        query_tokens = set(re.findall(r"\w+", query.lower()))
        scored = []

        for turn in self.sessions[session_id]:
            if turn.get("source") != "chat":
                continue
            if not self._is_fresh(turn.get("timestamp", "")):
                continue

            text = f"User: {turn.get('question','')}\nAI: {turn.get('answer','')}"
            tokens = set(re.findall(r"\w+", text.lower()))

            overlap = len(query_tokens & tokens)
            if overlap > 0:
                scored.append((overlap, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:k]]

    # -------------------------------------------------
    # Maintenance
    # -------------------------------------------------

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id] = []
            self._save_to_disk()


# -------------------------------------------------
# Global singleton (production-safe)
# -------------------------------------------------
session_memory = SessionMemory()
