import os
import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from typing import Union, Dict, List
from dotenv import load_dotenv


_MODEL = None
_FAISS_INDEX = {}
_VECTOR_CHUNKS = {}

# SETUP
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "chunks.json")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "chunk_embeddings.json")

MODEL_NAME = "BAAI/bge-base-en-v1.5"
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

    context = "\n\n".join(
        f"[Meeting {c.get('meeting_id', c.get('meeting_index'))} | "
        f"Chunk {c.get('chunk_index')}]: {c.get('text', '')}"
        for c in retrieved_chunks
        if isinstance(c, dict) and isinstance(c.get("text"), str)
    )

    if not context.strip():
        return SAFE_ABSTAIN

    prompt = f"""
You are an evidence-bound assistant.

RULES (ABSOLUTE):
- Use ONLY the transcript evidence below.
- Do NOT use outside knowledge.
- Do NOT guess or assume.
- Paraphrase facts; do NOT copy transcript text verbatim.
- Answer in 1–3 concise sentences.
- If the transcript does NOT explicitly contain the answer,
  reply EXACTLY with:
  "{SAFE_ABSTAIN}"

Transcript evidence:
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

    global _MODEL, _FAISS_INDEX, _VECTOR_CHUNKS

    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)

    cache_key = f"{user_id}::{project_type or 'all'}"

    if cache_key not in _FAISS_INDEX:
        X = np.array(vectors, dtype="float32")
        faiss.normalize_L2(X)

        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)

        _FAISS_INDEX[cache_key] = index
        _VECTOR_CHUNKS[cache_key] = vector_chunks

    # IMPORTANT: must match ingestion format (no "query:" prefix unless used in ingestion)
    q_emb = _MODEL.encode(
        query_text,
        normalize_embeddings=True
    )
    q_emb = np.array([q_emb], dtype="float32")

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
