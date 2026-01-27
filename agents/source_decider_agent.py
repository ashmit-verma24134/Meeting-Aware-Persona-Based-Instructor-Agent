from agents.dominance_utils import select_dominant_meeting
from agents.decision_types import Decision

from agents.dominance_utils import select_dominant_meeting
from agents.decision_types import Decision


def decide_source_node(state):

    state["path"].append("source_decider")

    print("\n DEBUG decide_source_node: ENTER")


    if state.get("decision") == Decision.CHAT_ONLY:
        print(" DEBUG: decision already CHAT_ONLY → returning")
        return state


    chunks = state.get("retrieved_chunks")
    print(
        f" DEBUG: incoming chunks = "
        f"{len(chunks) if isinstance(chunks, list) else 'INVALID'}"
    )

    if not isinstance(chunks, list) or not chunks:
        print(" DEBUG: NO_EVIDENCE (empty or invalid chunks)")
        state["decision"] = Decision.NO_EVIDENCE
        state["retrieved_chunks"] = []
        state["meeting_indices"] = []
        return state


    meeting_ids = [
        c.get("meeting_index")
        for c in chunks
        if isinstance(c.get("meeting_index"), int)
    ]

    print(f" DEBUG: meeting_ids distribution = {meeting_ids}")

    if not meeting_ids:
        print(" DEBUG: NO_EVIDENCE (no valid meeting_index)")
        state["decision"] = Decision.NO_EVIDENCE
        state["retrieved_chunks"] = []
        state["meeting_indices"] = []
        return state


    question_intent = state.get("question_intent")

    if question_intent == "factual":
        print(
            f" DEBUG: factual intent detected → "
            f"passing ALL {len(chunks)} chunks forward"
        )
        state["decision"] = Decision.RETRIEVAL_ONLY
        state["retrieved_chunks"] = chunks
        state["meeting_indices"] = sorted(set(meeting_ids))
        return state


    dominant_meeting = select_dominant_meeting(chunks)
    print(f"DEBUG: dominant_meeting = {dominant_meeting}")

    # If dominance unclear → keep all evidence
    if dominant_meeting is None:
        print(
            f" DEBUG: dominance UNCLEAR → "
            f"passing ALL {len(chunks)} chunks forward"
        )
        state["decision"] = Decision.RETRIEVAL_ONLY
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
        print(" DEBUG: NO_DOMINANT_EVIDENCE (filter removed everything)")
        state["decision"] = Decision.NO_DOMINANT_EVIDENCE
        state["retrieved_chunks"] = []
        state["meeting_indices"] = []
        return state


    state["retrieved_chunks"] = filtered_chunks
    state["meeting_indices"] = [dominant_meeting]
    state["decision"] = Decision.RETRIEVAL_ONLY

    print(
        f" DEBUG decide_source_node: EXIT with "
        f"{len(filtered_chunks)} chunks from meeting {dominant_meeting}"
    )

    return state
