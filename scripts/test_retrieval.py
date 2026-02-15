import sys
import os
from uuid import uuid4

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphs.meeting_graph import meeting_graph
from services.supabase_service import get_supabase_client


# --------------------------------------------------
# Compile and store full transcript for a session
# --------------------------------------------------

def compile_and_store_full_transcript(session_id):
    supabase = get_supabase_client()

    response = (
        supabase.client
        .table("chat_turns")
        .select("question, answer")
        .eq("session_id", session_id)
        .order("id", desc=False)
        .execute()
    )

    rows = response.data or []

    if not rows:
        return

    transcript_parts = []

    for row in rows:
        q = row.get("question", "")
        a = row.get("answer", "")
        transcript_parts.append(f"User: {q}\nAI: {a}")

    full_transcript = "\n\n".join(transcript_parts)

    (
        supabase.client
        .table("sessions")
        .update({"full_transcript": full_transcript})
        .eq("session_id", session_id)
        .execute()
    )


# --------------------------------------------------
# Main CLI
# --------------------------------------------------

def main():
    username = input("Enter username: ").strip()

    supabase = get_supabase_client()

    # Resolve user
    user = supabase.get_user_by_username(username)

    if not user:
        user = supabase.create_user(username)

    user_id = user["id"]

    # Create session
    session_id = str(uuid4())
    supabase.create_session(session_id, user_id)

    print("Session started.")
    print("Type 'exit' to end.\n")

    # Chat loop
    while True:
        question = input("Ask: ").strip()

        if not question:
            continue

        if question.lower() == "exit":
            compile_and_store_full_transcript(session_id)
            supabase.end_session(session_id)
            print("Session ended.")
            break

        initial_state = {
            "user_id": user_id,
            "session_id": session_id,
            "question": question,
            "decision": None,
            "standalone_query": question,
            "confidence": None,
            "temporal_constraint": None,
            "domain_constraint": None,
            "retrieved_chunks": [],
            "meeting_indices": None,
            "_all_meeting_indices": None,
            "question_intent": None,
            "time_scope": None,
            "candidate_answer": None,
            "final_answer": None,
            "method": "",
            "context_extended": False,
            "path": [],
        }

        result = meeting_graph.invoke(initial_state)

        print("\nAnswer:")
        print(result.get("final_answer"))
        print()


if __name__ == "__main__":
    main()
