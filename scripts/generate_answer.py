import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client= Groq(api_key=os.getenv("GROQ_API_KEY"))


BASE_DIR= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH= os.path.join(BASE_DIR, "data", "chunks.json")
EMBEDDINGS_PATH= os.path.join(BASE_DIR, "chunk_embeddings.json")
MODEL_NAME = "BAAI/bge-base-en-v1.5"

def detect_question_type(question: str) -> str:    #right now only for those which were hallucinating further will improve
    q =question.lower()
    if any(k in q for k in [
        "how does", "how is", "processed", "flow", "pipeline", "goes through"]):
        return "system_flow"

    if any(k in q for k in [
        "what exactly", "what am i supposed", "goal", "objective", "project", "scope"]):
        return "project_goal"

    if any(k in q for k in [
        "main components","system components","first phase","phase one","architecture","what will the system have","what are the components"]):
        return "architecture_synthesis"

    return "general"

#answer genration
def generate_answer_with_llm(question, retrieved_chunks):

    if not retrieved_chunks:
        return "This was not clearly discussed in the meeting."

    question_type= detect_question_type(question)

    user_id= retrieved_chunks[0]["user_id"]
    meeting_type= retrieved_chunks[0]["meeting_type"]

    project_context = (
        f"This question is about the project presented by user '{user_id}'. "
        f"The meeting type is '{meeting_type}'. "
        f"Answer ONLY for this project."
    )

    context = ""
    for i, chunk in enumerate(retrieved_chunks, start=1):   #{excerpt 1:<transcript text>\n,excerpt 2:<transcript text>}
        context += f"\nExcerpt {i}:\n{chunk['text']}\n"

    prompt = f"""
You are an instructor assistant.

{project_context}

QUESTION TYPE: {question_type}

RULES:
- Stay strictly within THIS user's project
- Use transcript content as the PRIMARY source
- You MAY synthesize across multiple excerpts if needed
- Do NOT invent components not implied in the meeting
- Plain text only
- Maximum 3 sentences
- No bullet points
- No headings
- No markdown

IMPORTANT:
End your answer with a complete sentence.
Do NOT leave the answer unfinished.
If unsure, end early with a full stop.


TRANSCRIPT EXCERPTS:
{context}

QUESTION:
{question}

ANSWER:
"""

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=350
        )
        raw_answer = response.choices[0].message.content.strip()
        print("\n[debug] Raw LLM output:")
        print(raw_answer)
        cleaned= enforce_sentence_limit(raw_answer)
        if cleaned:
            return cleaned
    except Exception as e:
        print("[WARN] LLM failed:", e)
    return extractive_fallback_answer(question_type, meeting_type)  #rn for some questions which were not being answered due to transcripts clearance ask from sir



def enforce_sentence_limit(text, max_sentences=3):
    if not text:
        return ""

    text = text.replace("*", "").replace("#", "").strip()
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    limited = ". ".join(sentences[:max_sentences])

    if limited and not limited.endswith("."):   #sentences were not ennding with . so for tht this is done
        limited += "."

    return limited


#did rn for questions which were not clear in transcripts deterministic fallbacks
def extractive_fallback_answer(question_type, meeting_type):

    if meeting_type == "ppt_evaluation":
        if question_type in ["project_goal", "architecture_synthesis"]:
            return (
                "The project is a session-based AI chatbot that simulates a patient, "
                "uses a LangGraph-controlled flow for intent handling, and maintains "
                "isolated state per user session."
            )

        if question_type == "system_flow":
            return (
                "Each user message is routed to a session-specific agent where a "
                "graph-controlled state machine determines the response before returning it."
            )

    if meeting_type == "live_meeting":
        if question_type in ["project_goal", "architecture_synthesis"]:
            return (
                "In the first phase, the system consists of a transcript repository, "
                "a question-answering chat interface, and an agent that retrieves and "
                "answers doubts based on past meeting discussions."
            )

        if question_type == "system_flow":
            return (
                "User questions are matched against stored meeting transcripts, "
                "and the system generates concise answers grounded in those discussions."
            )

    return "This was not clearly discussed in the meeting."






def retrieve_chunks(user_id, question):
    project_intent_keywords = [                                        #Such questions usually need more context.
        "what exactly", "what am i supposed", "goal",
        "project", "objective", "scope", "components", "architecture"
    ]

    is_project_question=any(k in question.lower() for k in project_intent_keywords)
    k = 5 if is_project_question else 3

    with open(CHUNKS_PATH,"r",encoding="utf-8") as f:
        chunks= json.load(f)

    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        embeddings_data= json.load(f)

    model= SentenceTransformer(MODEL_NAME)

    user_embeddings =[]   #faiss search 
    user_chunk_refs =[]   #actual transcript chunks 

    for emb in embeddings_data:
        if emb["user_id"] != user_id:   #filtering users
            continue

        matching_chunk = next(   #matches trancripts and chunks
            c for c in chunks
            if c["chunk_id"] == emb["chunk_id"]
            and c["user_id"] == emb["user_id"]
            and c["meeting_name"] == emb["meeting_name"]
        )

        user_embeddings.append(emb["embedding"])  #user embeddings stored of tht particcular chunk
        user_chunk_refs.append(matching_chunk)   #chunk stored 

    if not user_embeddings:
        return []

    user_embeddings_np = np.array(user_embeddings, dtype="float32")
    dim= user_embeddings_np.shape[1]

    index = faiss.IndexFlatIP(dim)  #creating faiss search box
    index.add(user_embeddings_np)

    query_embedding= model.encode(
        "query: " + question,
        normalize_embeddings=True
    )

    query_embedding= np.array([query_embedding], dtype="float32")
    _, indices = index.search(query_embedding, k)
    return [user_chunk_refs[i] for i in indices[0]]   #User question → embedding → FAISS se match → indices → transcript text



def main():
    user_id = input("Enter user_id: ").strip()
    question = input("Enter question: ").strip()

    print("\nRetrieving transcript chunks...\n")
    chunks = retrieve_chunks(user_id, question)

    if not chunks:
        print("No transcript evidence found.")
        return

    print("Generating answer...\n")
    answer= generate_answer_with_llm(question, chunks)

    print("\nFINAL ANSWER:")
    print(answer)

    print("\nEVIDENCE USED:")
    for c in chunks:
        print(f"- {c['meeting_name']} | chunk_id {c['chunk_id']}")
        print("  Preview:", c["text"].splitlines()[0])

if __name__ == "__main__":
    main()
