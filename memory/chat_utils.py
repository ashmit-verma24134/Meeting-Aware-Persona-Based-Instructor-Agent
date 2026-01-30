from typing import List
from memory.session_memory import session_memory


def get_chat_texts(session_id: str, k: int = 20) -> List[str]:
    """
    Returns the last k conversation turns as plain text,
    in chronological order, for LLM context.

    Includes BOTH chat + system answers, because once shown
    to the user, they are conversational ground truth.
    """

    turns = session_memory.get_recent_context(
        session_id=session_id,
        k=k
    )

    texts: List[str] = []

    for t in turns:
        q = (t.get("question") or "").strip()
        a = (t.get("answer") or "").strip()

        if q:
            texts.append(f"User: {q}")
        if a:
            texts.append(f"AI: {a}")

    return texts
