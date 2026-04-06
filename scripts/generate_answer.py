import os
import json
import re
from services.embedding_api import get_embedding
from groq import Groq
from typing import Union, Dict, List
from dotenv import load_dotenv


_FAISS_INDEX = {}
_VECTOR_CHUNKS = {}

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "chunks.json")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "chunk_embeddings.json")

SAFE_ABSTAIN = "This was not clearly discussed in the meeting."


def trim_chunk_text(text: str, max_words: int = 120) -> str:
    if not text:
        return ""
    cleaned_text = " ".join(text.split())
    words = cleaned_text.split()
    if len(words) <= max_words:
        return cleaned_text
    return " ".join(words[:max_words]) + "..."


def generate_answer_with_llm(
    question: str,
    retrieved_chunks: list,
    chat_context: str = ""
) -> str:

    if not retrieved_chunks:
        return SAFE_ABSTAIN

    # Cap to top 5 chunks to reduce noise reaching the LLM
    retrieved_chunks = retrieved_chunks[:5]

    context = "\n\n".join(
        f"[Meeting {c.get('meeting_id', c.get('meeting_index'))} | "
        f"Chunk {c.get('chunk_index')}]: {c.get('text', '')}"
        for c in retrieved_chunks
        if isinstance(c, dict) and isinstance(c.get("text"), str)
    )

    if not context.strip():
        return SAFE_ABSTAIN

    prompt = f"""
You are a helpful project assistant for students working on capstone projects.

RULES:
- PRIORITIZE the provided context (transcripts, Slack history, project docs).
- If the context contains a direct answer to the question, answer from it.
- If the question asks what happened, what was discussed, or what was done on a specific date, describe ALL activity visible in the context — including commands run, pipeline starts, questions asked, bot responses, and any other actions. Do NOT refuse just because there is no formal "discussion".
- If the question asks which model/tool was used for something, collect ALL models/tools mentioned across ALL chunks that are directly related to that task and list them together in one answer. Do NOT pick just one and ignore the rest.
- If multiple chunks discuss the SAME topic or task, combine their information into one complete answer.
- Do NOT draw connections between chunks that are clearly about different unrelated topics.
- If a chunk describes something visual (e.g. "menu bar", "screen", "window visible") and the question is NOT about visuals or screen content, ignore that chunk entirely.
- If the question is about a person and their name appears in the context as a speaker or subject with relevant facts, answer from the context directly.

CRITICAL — CHUNK CONTAINS THE QUESTION BEING ASKED, NOT AN ANSWER:
- If the context only shows that someone ASKED the question (e.g. "[Ashmit Verma]: who is abdul kalam") but does NOT provide an actual answer to it, that chunk does NOT count as context. Treat it as if no relevant context exists and apply the rules below.

WHEN CONTEXT DOES NOT CONTAIN THE ANSWER:
- If the question is general knowledge (a well-known person, historical figure, scientific concept, geography, technology), answer it from your own knowledge concisely in 1–2 sentences.
- If the question is NOT general knowledge and the context does not contain the answer, respond with exactly: "The answer is not available in the provided context."
- Do NOT fabricate or infer answers from unrelated chunks.

OTHER RULES:
- Paraphrase facts; do NOT copy text verbatim.
- EXCEPTION: For specific IDs, codes, tokens, URLs, numbers — always quote the exact value directly.
- Answer in 1–3 concise sentences.
- Do NOT include bare timestamps like [00:15:30] in your answer unless you also know the calendar date.

EXAMPLES:
Q: Who was presenting in the meeting?
A: Gautam Shroff was presenting in the meeting.

Q: What was visible on the screen?
A: A Finder window with TypeScript files and a Google Meet picture-in-picture window were visible on the screen.

Q: What happened to Manmohan Singh?
A: Manmohan Singh, a member of parliament, passed away on 19/03/2026 at 5:49 pm.

Q: What pipeline was used to process the video?
A: The smart_keyframes_and_classify pipeline was used to process the meeting video.

Q: Was the bot working well?
A: Yes, Ashmit Verma mentioned that the bot was working really well.

Q: What models were used for text extraction?
A: The BG model code, along with the Blip and Clip models, were used for text extraction from frames.

Q: Who is Abdul Kalam?
Context only contains: "[Ashmit Verma]: who is abdul kalam" (someone asking, no answer present)
A: Dr. A.P.J. Abdul Kalam was an Indian aerospace scientist and the 11th President of India, widely known as the Missile Man of India.

Q: What is the capital of France?
Context contains no relevant information.
A: The capital of France is Paris.

Q: What did the team decide about database migrations?
Context contains no relevant information.
A: The answer is not available in the provided context.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )

        answer = response.choices[0].message.content.strip()

        if not answer or SAFE_ABSTAIN.lower() in answer.lower():
            return SAFE_ABSTAIN

        # Map the new abstention phrase back to SAFE_ABSTAIN constant
        if "not available in the provided context" in answer.lower():
            return SAFE_ABSTAIN

        return answer

    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return SAFE_ABSTAIN


def enforce_sentence_limit(text: str, max_sentences: int = 3) -> str:
    if not text:
        return ""

    clean_text = text.replace("##", "").replace("**", "").replace("__", "").strip()
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    limited = " ".join(sentences[:max_sentences])

    if limited and limited[-1] not in ".!?":
        limited += "."

    return limited


def retrieve_chunks(
    user_id: str,
    query_or_payload: Union[str, Dict],
) -> Dict[str, List[dict]]:

    if isinstance(query_or_payload, list):
        query_or_payload = {}

    if not isinstance(query_or_payload, (dict, str)):
        query_or_payload = {}

    if isinstance(query_or_payload, dict):
        query_text = (
            query_or_payload.get("standalone_query")
            or query_or_payload.get("question", "")
        )
        project_type = query_or_payload.get("project_type")
    else:
        query_text = str(query_or_payload)
        project_type = None

    print(f" SEARCHING (GLOBAL): '{query_text}' | project={project_type}")

    if not os.path.exists(CHUNKS_PATH) or not os.path.exists(EMBEDDINGS_PATH):
        return {"chunks": [], "_all_meeting_indices": []}

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    user_chunks = [
        c for c in all_chunks
        if isinstance(c, dict) and c.get("user_id") == user_id
    ]

    if not user_chunks:
        return {"chunks": [], "_all_meeting_indices": []}

    if project_type:
        user_chunks = [
            c for c in user_chunks
            if c.get("project_type") == project_type
        ]
        if not user_chunks:
            return {"chunks": [], "_all_meeting_indices": []}

    meeting_ids = sorted({
        c.get("meeting_index")
        for c in user_chunks
        if isinstance(c.get("meeting_index"), int)
    })

    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        embeddings_data = json.load(f)

    chunk_map = {c["chunk_id"]: c for c in user_chunks}

    vectors, vector_chunks = [], []
    for e in embeddings_data:
        if e["chunk_id"] in chunk_map:
            vectors.append(e["embedding"])
            vector_chunks.append(chunk_map[e["chunk_id"]])

    if not vectors:
        return {"chunks": [], "_all_meeting_indices": meeting_ids}

    global _FAISS_INDEX, _VECTOR_CHUNKS

    cache_key = f"{user_id}::{project_type or 'all'}"

    if cache_key not in _FAISS_INDEX:
        X = np.array(vectors, dtype="float32")
        faiss.normalize_L2(X)

        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)

        _FAISS_INDEX[cache_key] = index
        _VECTOR_CHUNKS[cache_key] = vector_chunks

    q_emb = get_embedding(query_text)
    q_emb = np.array([q_emb], dtype="float32")

    faiss.normalize_L2(q_emb)

    TOP_K = min(25, len(_VECTOR_CHUNKS[cache_key]))
    scores, ids = _FAISS_INDEX[cache_key].search(q_emb, TOP_K)

    SIM_THRESHOLD = 0.25

    filtered_chunks = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        if score < SIM_THRESHOLD:
            continue
        filtered_chunks.append(_VECTOR_CHUNKS[cache_key][idx])

    return {
        "chunks": filtered_chunks,
        "_all_meeting_indices": meeting_ids
    }