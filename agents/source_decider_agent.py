from agents.dominance_utils import select_dominant_meeting
from agents.decision_types import Decision


def decide_source_node(state):

    state.setdefault("path", [])
    state["path"].append("source_decider")

    print("\n DEBUG decide_source_node: ENTER")

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

    # Once transcript evidence exists → retrieval mode
    state["decision"] = Decision.RETRIEVAL_ONLY


    meeting_ids = [
        c.get("meeting_id")
        for c in chunks
        if isinstance(c, dict) and isinstance(c.get("meeting_id"), str)
    ]

    print(f" DEBUG: meeting_ids distribution = {meeting_ids}")

    if not meeting_ids:
        print(" DEBUG: NO_EVIDENCE (no valid meeting_id)")
        state["decision"] = Decision.NO_EVIDENCE
        state["retrieved_chunks"] = []
        state["meeting_indices"] = []
        return state

    question_intent = state.get("question_intent", "factual")
  
    if question_intent == "factual":
        print(
            f" DEBUG: factual intent → KEEPING ALL "
            f"{len(chunks)} chunks"
        )

        state["retrieved_chunks"] = chunks
        state["meeting_indices"] = list(dict.fromkeys(meeting_ids))  # preserve order
        return state


    dominant_meeting = select_dominant_meeting(chunks)
    print(f" DEBUG: dominant_meeting = {dominant_meeting}")

    # If dominance unclear → keep everything
    if dominant_meeting is None:
        print(" DEBUG: dominance unclear → KEEPING ALL chunks")

        state["retrieved_chunks"] = chunks
        state["meeting_indices"] = list(dict.fromkeys(meeting_ids))
        return state


    filtered_chunks = [
        c for c in chunks
        if c.get("meeting_id") == dominant_meeting
    ]

    print(
        f" DEBUG: filtered_chunks = {len(filtered_chunks)} "
        f"for meeting {dominant_meeting}"
    )

    if not filtered_chunks:
        print(" DEBUG: NO_DOMINANT_EVIDENCE after filtering")

        state["decision"] = Decision.NO_DOMINANT_EVIDENCE
        state["retrieved_chunks"] = []
        state["meeting_indices"] = []
        return state

    state["retrieved_chunks"] = filtered_chunks
    state["meeting_indices"] = [dominant_meeting]

    print(
        f" DEBUG decide_source_node: EXIT with "
        f"{len(filtered_chunks)} chunks"
    )

    return state
