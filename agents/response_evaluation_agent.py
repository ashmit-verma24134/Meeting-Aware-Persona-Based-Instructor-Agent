SAFE_ABSTAIN = "This was not clearly discussed in the meeting."


def evaluate_answer(answer: str, question: str) -> bool:

    if not answer:
        return False

    answer = answer.strip()

    # Rule 1: Safe abstention is OK
    if answer == SAFE_ABSTAIN:
        return True

    # Rule 2: Sentence length check (1–3 sentences)
    sentence_count = answer.count(".")
    if sentence_count == 0 or sentence_count > 3:
        return False

    # Rule 3: Relevance to meeting context
    relevance_keywords = [
        "meeting",
        "instructor",
        "discussed",
        "asked",
        "explained",
        "project",
        "presentation",
        "session"
    ]

    lower_answer = answer.lower()
    if not any(keyword in lower_answer for keyword in relevance_keywords):
        return False

    return True
