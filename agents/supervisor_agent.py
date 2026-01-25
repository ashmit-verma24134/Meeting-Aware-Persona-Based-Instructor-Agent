from graphs.meeting_graph import meeting_graph
from memory.session_memory import session_memory
from agents.query_understanding_agent import understand_query
from agents.decision_types import Decision

SAFE_ABSTAIN = "This was not clearly discussed in the meeting."


def supervisor(user_id: str, session_id: str, question: str):
    """
    GOOGLE-ALIGNED COORDINATOR AGENT

    RULES:
    - Uses enums only (no string decisions)
    - No intent inference
    - No domain inference
    - No evidence inspection
    - Delegates ALL logic to the graph
    """

    question = (question or "").strip()

    # -------------------------------------------------
    # 1. SHORT-TERM MEMORY (HARD DOMINANCE)
    # -------------------------------------------------
    recent_context = session_memory.get_recent_context(
        session_id=session_id,
        k=6
    )

    if session_memory.chat_can_answer(question, recent_context):
        decision = Decision.CHAT_ONLY
        standalone_query = question
    else:
        analysis = understand_query(
            question=question,
            recent_history=recent_context,
            user_id=user_id
        )

        standalone_query = analysis.get("standalone_query") or question
        decision = (
            Decision.IGNORE
            if analysis.get("ignore")
            else Decision.RETRIEVAL_ONLY
        )

    # -------------------------------------------------
    # 2. INITIAL GRAPH STATE (MINIMAL + CLEAN)
    # -------------------------------------------------
    state = {
        "user_id": user_id,
        "session_id": session_id,
        "question": question,
        "standalone_query": standalone_query,
        "decision": decision,

        # Graph-managed fields
        "retrieved_chunks": [],
        "candidate_answer": None,
        "final_answer": None,
        "context_extended": False,
        "method": "",
        "path": ["SUPERVISOR"],
    }

    print(
        " COORDINATOR → "
        f"decision={decision} | "
        f"query='{standalone_query}'"
    )

    # -------------------------------------------------
    # 3. EXECUTE GRAPH
    # -------------------------------------------------
    final_state = meeting_graph.invoke(state)

    # -------------------------------------------------
    # 4. RETURN RESPONSE ONLY
    # -------------------------------------------------
    return {
        "answer": final_state.get("final_answer") or SAFE_ABSTAIN,
        "method": final_state.get("method"),
        "context_extended": final_state.get("context_extended", False),
        "standalone_query": final_state.get("standalone_query"),
    }
