
from scripts.generate_answer import retrieve_chunks, generate_answer_with_llm
from agents.query_understanding_agent import understand_query, is_follow_up
from memory.session_memory import SessionMemory

session_memory = SessionMemory()


def clean_answer(text: str, max_sentences=3) -> str:

    #output cleaning

    if not text:
        return ""

    text = text.replace("*", "").replace("#", "").strip()
    sentences = [s.strip() for s in text.split(".") if s.strip()]

    sentences = sentences[:max_sentences]
    final = ". ".join(sentences)

    if final and not final.endswith("."):
        final += "."

    return final

def evaluate_answer(answer: str) -> bool:

    if not answer:
        return False

    if len(answer.split()) < 6:
        return False

    return True


def supervisor(user_id: str, session_id: str, question: str):

    print("\n[Supervisor] Understanding query...")
    intent = understand_query(question)

    if intent != "meeting_content":
        return {
            "answer": "This was not clearly discussed in the meeting.",
            "evidence": []
        }



#FOLLOW-UP HANDLING 

    is_followup = is_follow_up(question)

    retrieval_question = question
    generation_question = question

    if is_followup:
        recent = session_memory.get_recent_context(session_id, k=1)
        last_turn = recent[0] if recent else None

        if last_turn:
            generation_question = (
                "Explain the following in simpler terms:\n"
                f"{last_turn['answer']}"
            )


    print("[Supervisor] Retrieving transcript chunks...")
    chunks = retrieve_chunks(user_id, question)

    if not chunks:
        return {
            "answer": "This was not clearly discussed in the meeting.",
            "evidence": []
        }

    print("[Supervisor] Generating answer...")
    raw_answer = generate_answer_with_llm(generation_question, chunks)

    print("[Supervisor] Evaluating answer...")
    if not evaluate_answer(raw_answer):
        return {
            "answer": "This was not clearly discussed in the meeting.",
            "evidence": []
        }

    final_answer = clean_answer(raw_answer)

    #Store last one

    session_memory.add_turn(
        session_id=session_id,
        question=question,
        answer=final_answer
    )

    return {
        "answer": final_answer,
        "evidence": chunks
    }


#Entry point

def main():
    user_id = input("Enter user_id: ").strip()
    session_id = input("Enter session_id: ").strip()
    question = input("Enter question: ").strip()

    result = supervisor(user_id, session_id, question)

    print("\nFINAL ANSWER:")
    print(result["answer"])

    print("\nEVIDENCE USED:")
    for chunk in result["evidence"]:
        print(f"- {chunk['meeting_name']} | chunk_id {chunk['chunk_id']}")
        print("  Text preview:", chunk["text"].splitlines()[0])


if __name__ == "__main__":
    main()
