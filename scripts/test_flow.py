from datetime import datetime
from agents.supervisor_agent import supervisor

USER_ID = "ashmit"

QUESTIONS = [
    # Normal QA
    "Why was manual chunking introduced?",

    # Follow-up rewrite
    "Explain that in simple terms.",

    # Normal QA (should not rewrite)
    "What did sir say about frame extraction?",

    # Weak-answer → backward context
    "Why did he mention efficiency issues?",

    # Summary (latest meeting)
    "What were the goals of the meeting?",

    # Rewrite summary
    "Explain that.",

    # Verification test (example vs decision)
    "Did we decide to deploy next week?",

    # Out-of-scope
    "Who is the Prime Minister of India?"
]


def main():
    session_start = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_id = f"{USER_ID}_test_{session_start}"

    print("\n==============================")
    print("LANGGRAPH FULL SYSTEM TEST")
    print("Session:", session_id)
    print("==============================\n")

    for i, q in enumerate(QUESTIONS, start=1):
        print(f"\n🟦 Q{i}: {q}")
        print("-" * 60)

        result = supervisor(USER_ID, session_id, q)

        print("ANSWER:")
        print(result["answer"])

        print("\nMETHOD USED:", result.get("method"))
        print("CONTEXT EXTENDED:", result.get("context_extended", False))

        if result.get("path"):
            print("PATH:", " → ".join(result["path"]))

        print("-" * 60)

    print("\nTEST COMPLETE")


if __name__ == "__main__":
    main()
