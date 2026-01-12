
from agents.supervisor_agent import supervisor

def main():
    print("\nMeeting-QA Supervisor")

    user_id = input("Enter user_id: ").strip()
    question = input("Enter question: ").strip()

    result = supervisor(user_id, question)

    print("\nFINAL ANSWER:")
    print(result["answer"])

    print("\nEVIDENCE USED:")
    if not result["evidence"]:
        print("None")
    else:
        for chunk in result["evidence"]:
            print(f"- {chunk['meeting_name']} | chunk_id {chunk['chunk_id']}")
            preview = chunk["text"].splitlines()[0]
            print(f"  Text preview: {preview}")


if __name__ == "__main__":
    main()
