from agents.dominance_utils import select_dominant_meeting
from agents.decision_types import Decision


def decide_source_node(state):

    state["path"].append("source_decider")
    print("\n DEBUG decide_source_node: ENTER")

    chunks = state.get("retrieved_chunks")

    print(
        f" DEBUG: incoming chunks = "
        f"{len(chunks) if isinstance(chunks, list) else 'INVALID'}"
    )

    #  HARD RULE: transcript evidence overrides chat
    if not isinstance(chunks, list) or not chunks:
        print(" DEBUG: NO_EVIDENCE (empty or invalid chunks)")
        state["decision"] = Decision.NO_EVIDENCE
        state["retrieved_chunks"] = []
        state["meeting_indices"] = []
        return state

    # Force retrieval mode once evidence exists
    state["decision"] = Decision.RETRIEVAL_ONLY

    meeting_ids = [
        c.get("meeting_index")
        for c in chunks
        if isinstance(c, dict) and isinstance(c.get("meeting_index"), int)
    ]

    print(f" DEBUG: meeting_ids distribution = {meeting_ids}")

    if not meeting_ids:
        print(" DEBUG: NO_EVIDENCE (no valid meeting_index)")
        state["decision"] = Decision.NO_EVIDENCE
        state["retrieved_chunks"] = []
        state["meeting_indices"] = []
        return state

    question_intent = state.get("question_intent", "factual")

    #  FACTUAL QUESTIONS → keep all evidence
    if question_intent == "factual":
        print(
            f" DEBUG: factual intent → "
            f"passing ALL {len(chunks)} chunks forward"
        )
        state["retrieved_chunks"] = chunks
        state["meeting_indices"] = sorted(set(meeting_ids))
        return state

    #  META / DISCUSSION → find dominant meeting
    dominant_meeting = select_dominant_meeting(chunks)
    print(f" DEBUG: dominant_meeting = {dominant_meeting}")

    if dominant_meeting is None:
        print(
            " DEBUG: dominance UNCLEAR → "
            "passing ALL chunks forward"
        )
        state["retrieved_chunks"] = chunks
        state["meeting_indices"] = sorted(set(meeting_ids))
        return state

    filtered_chunks = [
        c for c in chunks
        if c.get("meeting_index") == dominant_meeting
    ]

    print(
        f" DEBUG: filtered_chunks = {len(filtered_chunks)} "
        f"for meeting {dominant_meeting}"
    )

    if not filtered_chunks:
        print(" DEBUG: NO_DOMINANT_EVIDENCE")
        state["decision"] = Decision.NO_DOMINANT_EVIDENCE
        state["retrieved_chunks"] = []
        state["meeting_indices"] = []
        return state

    state["retrieved_chunks"] = filtered_chunks
    state["meeting_indices"] = [dominant_meeting]

    print(
        f" DEBUG decide_source_node: EXIT with "
        f"{len(filtered_chunks)} chunks from meeting {dominant_meeting}"
    )

    return state
