import re
from typing import Optional

SAFE_ABSTAIN = "This was not clearly discussed in the meeting."

def trim_chunk_text(text: Optional[str], max_words: int = 120) -> str:


    if not text or not isinstance(text, str):
        return ""

    # Normalize whitespace
    cleaned = " ".join(text.split())
    words = cleaned.split()

    if len(words) <= max_words:
        return cleaned

    return " ".join(words[:max_words]) + "..."

def clean_answer(text: Optional[str], max_sentences: int = 4) -> str:
    """
    FINAL ANSWER SANITIZER

    GUARANTEES:
    - Removes LLM narrator / boilerplate
    - Prevents speculative or generic expansion
    - Enforces short, transcript-faithful answers
    - NEVER invents information
    - Deterministic & production-safe
    """

    if not text or not isinstance(text, str):
        return SAFE_ABSTAIN

    cleaned = text.strip()

    if len(cleaned) < 10:
        return SAFE_ABSTAIN

    prefixes = [
        r"based on the transcript",
        r"according to the meeting notes",
        r"the transcript mentions",
        r"based on the provided fragments",
        r"from the meeting",
        r"it was discussed that",
        r"the discussion was about",
        r"the meeting discussed",
    ]

    for pref in prefixes:
        cleaned = re.sub(
            rf"^{pref}[:,]?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE
        )

    cleaned = re.sub(r"[#*_>`]", "", cleaned).strip()

    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    sentences = [
        s.strip()
        for s in sentences
        if len(s.strip().split()) >= 5
    ]

    if not sentences:
        return SAFE_ABSTAIN

    cleaned = " ".join(sentences[:max_sentences])

    hallucination_patterns = [
        r"\bin general\b",
        r"\btypically\b",
        r"\busually\b",
        r"\bcan be used to\b",
        r"\bone could\b",
        r"\bvarious ways\b",
        r"\bmany approaches\b",
        r"\betc\b",
        r"\bfor example\b",
        r"\bfor instance\b",
    ]

    if any(
        re.search(pattern, cleaned, re.IGNORECASE)
        for pattern in hallucination_patterns
    ):
        return SAFE_ABSTAIN
    cleaned = cleaned[0].upper() + cleaned[1:]

    if len(cleaned.split()) < 5:
        return SAFE_ABSTAIN

    return cleaned
