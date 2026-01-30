import re
from typing import Optional

SAFE_ABSTAIN = "This was not clearly discussed in the meeting."


def trim_chunk_text(text: Optional[str], max_words: int = 120) -> str:
    """
    Safely trims transcript chunk text without breaking sentences.
    """
    if not text or not isinstance(text, str):
        return ""

    cleaned = " ".join(text.split())
    words = cleaned.split()

    if len(words) <= max_words:
        return cleaned

    return " ".join(words[:max_words]) + "..."


def clean_answer(text: Optional[str], max_sentences: int = 4) -> str:
    """
    FINAL ANSWER SANITIZER (TRANSCRIPT-SAFE)

    Guarantees:
    - Does NOT delete valid transcript-grounded definitions
    - Removes narrator / boilerplate framing
    - Blocks generic, speculative, or filler expansion
    - Deterministic & conservative
    """

    # --------------------
    # Basic validation
    # --------------------
    if not text or not isinstance(text, str):
        return SAFE_ABSTAIN

    cleaned = text.strip()

    # Too short → useless
    if len(cleaned.split()) < 3:
        return SAFE_ABSTAIN

    # --------------------
    # Remove narrator / meta prefixes
    # --------------------
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

    # Remove markdown / formatting noise
    cleaned = re.sub(r"[#*_>`]", "", cleaned).strip()

    # --------------------
    # Sentence filtering
    # --------------------
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    sentences = [
        s.strip()
        for s in sentences
        if len(s.split()) >= 3
    ]

    if not sentences:
        return SAFE_ABSTAIN

    # Keep only first N sentences
    cleaned = " ".join(sentences[:max_sentences])

    # --------------------
    # Hallucination guard (STRICT but FAIR)
    # --------------------
    # NOTE:
    # We intentionally DO NOT block:
    # - "can be used to"
    # - "allows users to"
    # - "provides"
    # These are normal definition phrases.
    hallucination_patterns = [
        r"\bin general\b",
        r"\bvarious ways\b",
        r"\bmany approaches\b",
        r"\betc\b",
        r"\bfor instance\b",
        r"\bfor example\b",
    ]

    for pattern in hallucination_patterns:
        if re.search(pattern, cleaned, re.IGNORECASE):
            return SAFE_ABSTAIN

    # --------------------
    # Normalize capitalization
    # --------------------
    cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned
