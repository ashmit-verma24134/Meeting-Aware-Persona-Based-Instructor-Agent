from datetime import datetime
from agents.supervisor_agent import supervisor
from memory.session_memory import session_memory
from memory.session_persistence import save_session

SAFE_ABSTAIN_MSG = "This was not clearly discussed in the meeting."


def normalize_result(result):
    """
    Ensures CLI ALWAYS works with a dict.
    """
    if isinstance(result, dict):
        return result

    return {
        "answer": SAFE_ABSTAIN_MSG,
        "method": "SAFE_ABSTAIN",
        "context_extended": False
    }


def main():
    print("\nMeeting-QA Supervisor")

    user_id = input("Enter user_id: ").strip()
    meeting_name = "auto_session"

    session_start_time = datetime.now()
    safe_timestamp = session_start_time.strftime("%Y-%m-%d_%H-%M-%S")
    session_id = f"{user_id}_{meeting_name}_{safe_timestamp}"

    print("\nSession started. Type 'exit' to end.\n")

    while True:
        question = input("Ask: ").strip()

        if not question:
            continue

        if question.lower() == "exit":
            break

        result = None
        raw_result = supervisor(user_id, session_id, question)
        result = normalize_result(raw_result)


        print("\nANSWER:")
        print(result.get("answer", SAFE_ABSTAIN_MSG))

        print("\nCONTEXT EXTENDED:")
        print("Yes" if result.get("context_extended", False) else "No")

        print("\nMETHOD USED:")
        print(result.get("method", "unknown"))

        print("\n" + "-" * 40 + "\n")

    session_end_time = datetime.now()

    conversation = []
    history = session_memory.get_recent_context(session_id, k=10_000)

    for turn in history:
        conversation.append({
            "question": turn.get("question"),
            "answer": turn.get("answer"),
            "timestamp": turn.get("timestamp")
        })

    session_data = {
        "user_id": user_id,
        "meeting_name": meeting_name,
        "session_start_time": session_start_time.isoformat(),
        "session_end_time": session_end_time.isoformat(),
        "conversation": conversation
    }

    save_session(session_id, session_data)
    print("\nSession ended and saved successfully.")


if __name__ == "__main__":
    main()
