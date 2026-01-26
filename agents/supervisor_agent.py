from graphs.meeting_graph import meeting_graph

SAFE_ABSTAIN = "This was not clearly discussed in the meeting."


def supervisor(user_id: str, session_id: str, question: str):
    """
    GOOGLE-ALIGNED SUPERVISOR (DUMB ORCHESTRATOR)

    RULES:
    - NO chat inspection
    - NO decision logic
    - NO retrieval logic
    - Graph is the brain
    """

    question = (question or "").strip()

    # -------------------------------------------------
    # 1. MINIMAL INITIAL STATE
    # -------------------------------------------------
    state = {
        "user_id": user_id,
        "session_id": session_id,
        "question": question,

        # Graph will fill everything else
        "decision": None,
        "standalone_query": question,

        "retrieved_chunks": [],
        "candidate_answer": None,
        "final_answer": None,
        "confidence": None,
        "context_extended": False,
        "method": "",
        "path": ["SUPERVISOR"],
    }

    print(f" SUPERVISOR → question='{question}'")

    # -------------------------------------------------
    # 2. EXECUTE GRAPH (ALL INTELLIGENCE INSIDE)
    # -------------------------------------------------
    final_state = meeting_graph.invoke(state)

    # -------------------------------------------------
    # 3. RETURN USER RESPONSE ONLY
    # -------------------------------------------------
    return {
        "answer": final_state.get("final_answer") or SAFE_ABSTAIN,
        "method": final_state.get("method"),
        "context_extended": final_state.get("context_extended", False),
        "confidence": final_state.get("confidence"),
    }
