from typing import List
from memory.session_memory import session_memory


def get_chat_texts(session_id: str, k: int = 20) -> List[str]:
    """
    Returns last k CHAT-only turns as plain text.

    PURPOSE:
    - Used ONLY for chat embeddings (chat_recall_node)
    - NO transcript data
    - NO summaries
    - GOOGLE-aligned short-term memory

    FORMAT:
    User: ...
    AI: ...
    """

    turns = session_memory.get_recent_context(
        session_id=session_id,
        k=k
    )

    texts: List[str] = []

    for t in turns:
        if t.get("source") == "chat":
            q = t.get("question", "").strip()
            a = t.get("answer", "").strip()

            if q or a:
                texts.append(
                    f"User: {q}\nAI: {a}"
                )

    return texts
