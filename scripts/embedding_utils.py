def last_n_tokens(text: str, n: int = 30) -> str:
    """
    Returns the last `n` tokens (words) from text.
    Safe for empty or short text.
    """
    if not text:
        return ""

    tokens = text.split()
    return " ".join(tokens[-n:])


def build_embedding_text(chunk: dict, prev_chunk: dict | None = None, overlap_tokens: int = 30) -> str:
    """
    Builds embedding text with overlap from the previous chunk.
    Prevents cross-meeting context bleed.
    """

    # First chunk OR safety fallback
    if prev_chunk is None:
        return chunk["text"].strip()

    # 🚨 Safety check: do NOT mix meetings
    if (
        prev_chunk.get("meeting_name") != chunk.get("meeting_name")
        or prev_chunk.get("meeting_index") != chunk.get("meeting_index")
    ):
        return chunk["text"].strip()

    overlap = last_n_tokens(prev_chunk.get("text", ""), overlap_tokens)

    if overlap:
        return f"{overlap} {chunk['text']}".strip()

    return chunk["text"].strip()
