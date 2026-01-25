import re
from typing import Optional

# -------------------------------------------------
# GLOBAL SAFE FALLBACK
# -------------------------------------------------
SAFE_ABSTAIN = "This was not clearly discussed in the meeting."


# -------------------------------------------------
# CHUNK TRIMMER (🔥 REQUIRED BY RETRIEVAL + QA)
# -------------------------------------------------
def trim_chunk_text(text: Optional[str], max_words: int = 120) -> str:
    """
    Token-safe chunk trimming.

    PURPOSE:
    - Reduce token usage
    - Preserve semantic core
    - Deterministic (no summarization)

    Used by:
    - chunk_answer_node
    - generate_answer_with_llm
    """

    if not text or not isinstance(text, str):
        return ""

    # Normalize whitespace
    cleaned = " ".join(text.split())
    words = cleaned.split()

    if len(words) <= max_words:
        return cleaned

    return " ".join(words[:max_words]) + "..."


# -------------------------------------------------
# FINAL ANSWER SANITIZER (🔥 LAST LINE OF DEFENSE)
# -------------------------------------------------
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

    # -------------------------------------------------
    # 0️⃣ Hard safety
    # -------------------------------------------------
    if not text or not isinstance(text, str):
        return SAFE_ABSTAIN

    cleaned = text.strip()

    if len(cleaned) < 10:
        return SAFE_ABSTAIN

    # -------------------------------------------------
    # 1️⃣ Remove narrator / AI boilerplate
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 2️⃣ Remove markdown / formatting noise
    # -------------------------------------------------
    cleaned = re.sub(r"[#*_>`]", "", cleaned).strip()

    # -------------------------------------------------
    # 3️⃣ Sentence limiting (ANTI-RAMBLE)
    # -------------------------------------------------
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    sentences = [
        s.strip()
        for s in sentences
        if len(s.strip().split()) >= 5
    ]

    if not sentences:
        return SAFE_ABSTAIN

    cleaned = " ".join(sentences[:max_sentences])

    # -------------------------------------------------
    # 4️⃣ Generic hallucination guard (STRICT)
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 5️⃣ Capitalize safely
    # -------------------------------------------------
    cleaned = cleaned[0].upper() + cleaned[1:]

    # -------------------------------------------------
    # 6️⃣ Final sanity check
    # -------------------------------------------------
    if len(cleaned.split()) < 5:
        return SAFE_ABSTAIN

    return cleaned
