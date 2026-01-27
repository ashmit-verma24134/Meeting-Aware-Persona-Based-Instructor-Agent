from typing import TypedDict, List, Optional
import json
from agents.text_utils import clean_answer
from agents.source_decider_agent import decide_source_node
from agents.decision_types import Decision
from agents.text_utils import trim_chunk_text
from agents.decision_types import Decision
from memory.chat_utils import get_chat_texts
from memory.chat_vector_store import ChatVectorStore


import numpy as np

from langgraph.graph import StateGraph, END


from groq import Groq
from scripts.generate_answer import (
    retrieve_chunks, 
    generate_answer_with_llm, 
    CHUNKS_PATH 
)

client = Groq() 
class MeetingState(TypedDict):
    # Core identity
    user_id: str
    session_id: str
    question: str

    # Coordinator outputs
    decision: Decision
    standalone_query: str
    confidence: Optional[float]

    # Hard constraints (coordinator-controlled)
    temporal_constraint: Optional[str]     # "latest" | None
    domain_constraint: Optional[str]       # project isolation

    # Evidence tracking
    retrieved_chunks: List[dict]
    meeting_indices: Optional[List[int]]
    _all_meeting_indices: Optional[List[int]]

    # Reasoning outputs (post-retrieval only)
    question_intent: Optional[str]          # factual | meta
    time_scope: Optional[str]               # latest | global

    # Answer lifecycle
    candidate_answer: Optional[str]
    final_answer: Optional[str]
    method: str
    context_extended: bool

    path: List[str]



from sentence_transformers import SentenceTransformer

_ENTAILMENT_MODEL = None

def get_entailment_model():
    global _ENTAILMENT_MODEL
    if _ENTAILMENT_MODEL is None:
        _ENTAILMENT_MODEL = SentenceTransformer(
            "BAAI/bge-base-en-v1.5"
        )
    return _ENTAILMENT_MODEL



# QUERY UNDERSTANDING NODE
from agents.query_understanding_agent import understand_query
from memory.session_memory import session_memory

def query_understanding_node(state: MeetingState):


    state.setdefault("path", [])
    state["path"].append("query")


    recent = get_chat_texts(
        session_id=state["session_id"]
    )

    analysis = understand_query(
        state["question"],
        recent_history=recent,
        user_id=state["user_id"]
    )

    state["ignore"] = bool(analysis.get("ignore", False))
    state["standalone_query"] = analysis.get(
        "standalone_query", state["question"]
    )

    q = state["question"].lower()

    TEMPORAL_KEYS = [
        "last meeting",
        "latest meeting",
        "previous meeting",
        "most recent meeting",
        "last call",
        "last discussion",
    ]

    state["temporal_constraint"] = (
        "latest" if any(k in q for k in TEMPORAL_KEYS) else None
    )

    project_type = analysis.get("project_type")
    state["domain_constraint"] = (
        project_type
        if analysis.get("force_single_project") and project_type
        else None
    )

    state["candidate_answer"] = None
    state["final_answer"] = None
    state["retrieved_chunks"] = []
    state["context_extended"] = False
    state["method"] = ""
    state["meeting_indices"] = None
    state["_all_meeting_indices"] = None

    return state




def coordinator_node(state: MeetingState):


    state["path"].append("coordinator")

    if state.get("ignore"):
        state["decision"] = Decision.IGNORE
        return state

    decision = state.get("decision")

    if decision == Decision.CHAT_ONLY:
        return state

    if decision == Decision.RETRIEVAL_ONLY:
        return state

    state["decision"] = Decision.RETRIEVAL_ONLY
    state["method"] = "coordinator_fallback"
    return state


def meeting_summary_node(state: MeetingState):

    state["path"].append("meeting_summary")

    retrieved = state.get("retrieved_chunks", [])

    if not retrieved:
        state["final_answer"] = SAFE_ABSTAIN
        state["method"] = "summary_no_evidence"
        state["context_extended"] = False
        return state

    meeting_ids = [
        c["meeting_index"]
        for c in retrieved
        if isinstance(c.get("meeting_index"), int)
    ]

    if not meeting_ids:
        state["final_answer"] = SAFE_ABSTAIN
        state["method"] = "summary_no_meeting_index"
        state["context_extended"] = False
        return state

    latest_meeting = max(meeting_ids)

    latest_chunks = [
        c for c in retrieved
        if c.get("meeting_index") == latest_meeting
        and isinstance(c.get("text"), str)
    ]

    if not latest_chunks:
        state["final_answer"] = SAFE_ABSTAIN
        state["method"] = "summary_latest_empty"
        state["context_extended"] = False
        return state

    context = "\n\n".join(c["text"] for c in latest_chunks)[:12000]

    prompt = f"""
You are a professional executive assistant summarizing a meeting.

RULES:
- Use ONLY the provided transcript fragments.
- Do NOT infer missing information.
- Do NOT assume decisions unless explicitly stated.
- If information is unclear, omit it.

TASK:
Summarize the meeting with 3–5 bullet points covering:
• Goals
• Agenda
• Decisions
• Action items

TRANSCRIPT FRAGMENTS:
{context}

SUMMARY:
""".strip()

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )

        state["final_answer"] = response.choices[0].message.content.strip()
        state["method"] = "meeting_summary_latest"

    except Exception as e:
        print(f"Summary Error: {e}")
        state["final_answer"] = SAFE_ABSTAIN
        state["method"] = "summary_error"

    state["context_extended"] = False
    return state




from agents.text_utils import clean_answer, SAFE_ABSTAIN
from agents.decision_types import Decision
from memory.session_memory import session_memory


def pure_chat_node(state: MeetingState):

    state.setdefault("path", [])
    state["path"].append("pure_chat")

    # HARD ASSERT
    state["decision"] = Decision.CHAT_ONLY

    chat_chunks = session_memory.retrieve_chat_chunks(
        session_id=state["session_id"],
        query=state["question"],
        k=4
    )

    print("\n PURE CHAT DEBUG")
    print("Retrieved chat chunks:", len(chat_chunks))

    if not chat_chunks:
        print("No chat grounding → abstain")
        state["final_answer"] = SAFE_ABSTAIN
        state["method"] = "chat_no_memory"
        return state

    chat_context = "\n\n".join(chat_chunks)

    q = state["question"].lower()

    BRIEF_KEYS = {
        "brief", "summarize", "summary",
        "short", "in brief", "shortly"
    }

    wants_brief = any(k in q for k in BRIEF_KEYS)

    if wants_brief:
        task_instruction = (
            "Summarize the information in 1–2 concise lines."
        )
    else:
        task_instruction = (
            "Explain the information clearly in simple terms."
        )

    prompt = f"""
You are continuing an existing conversation.

STRICT RULES:
- Use ONLY the CHAT CONTEXT below.
- Do NOT add new facts or assumptions.
- Do NOT introduce new decisions.
- Do NOT reference meetings, transcripts, files, or retrieval.
- You MAY compress, summarize, or explain existing information.
- If the answer is not fully present in chat,
  reply EXACTLY with: "{SAFE_ABSTAIN}"

TASK:
{task_instruction}

CHAT CONTEXT:
{chat_context}

USER QUESTION:
"{state['question']}"

ANSWER:
""".strip()

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=120,
        )
        answer = response.choices[0].message.content.strip()
        if not answer:
            answer = SAFE_ABSTAIN

    except Exception as e:
        print("[PURE_CHAT_ERROR]", e)
        answer = SAFE_ABSTAIN

    state["final_answer"] = clean_answer(answer)
    state["method"] = "chat_only"
    state["context_extended"] = False

    # SAFETY CLEANUP
    state["retrieved_chunks"] = []
    state["meeting_indices"] = None

    return state


def retrieve_chunks_node(state: MeetingState):

    state["path"].append("retrieve_chunks")
   
    payload = {
        "standalone_query": state.get(
            "standalone_query",
            state["question"]
        )
    }

    if state.get("domain_constraint"):
        payload["project_type"] = state["domain_constraint"]

    result = retrieve_chunks(state["user_id"], payload)

    # FIX: retrieve_chunks returns LIST, not dict
    if isinstance(result, list):
        chunks = result
    elif isinstance(result, dict):
        chunks = result.get("chunks", [])
    else:
        chunks = []


    if not chunks:
        state["retrieved_chunks"] = []
        state["_all_meeting_indices"] = []
        state["meeting_indices"] = []
        return state

    if not isinstance(chunks, list):
        chunks = []

    clean_chunks = [
        c for c in chunks
        if isinstance(c, dict)
        and isinstance(c.get("meeting_index"), int)
        and isinstance(c.get("chunk_index"), int)
        and isinstance(c.get("text"), str)
    ]

    if not clean_chunks:
        state["retrieved_chunks"] = []
        state["_all_meeting_indices"] = []
        state["meeting_indices"] = []
        return state

    if state.get("temporal_constraint") == "latest":
        latest_meeting = max(c["meeting_index"] for c in clean_chunks)
        clean_chunks = [
            c for c in clean_chunks
            if c["meeting_index"] == latest_meeting
        ]

    clean_chunks = sorted(
        clean_chunks,
        key=lambda c: (c["meeting_index"], c["chunk_index"])
    )

    meeting_ids = sorted({
        c["meeting_index"] for c in clean_chunks
    })

    state["retrieved_chunks"] = clean_chunks
    state["_all_meeting_indices"] = meeting_ids
    state["meeting_indices"] = meeting_ids  #  FIX

    print(
        f" RETRIEVED {len(clean_chunks)} chunks | "
        f"meetings={meeting_ids}"
    )

    return state


def infer_intent_node(state: MeetingState):
    """
    Infer intent & time scope FROM EVIDENCE ONLY.
    Question text is used ONLY as a last-resort tie-breaker
    and MUST NOT introduce new intent.

    Option-3 compliant.
    """

    state["path"].append("infer_intent")

    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        state["question_intent"] = "factual"
        state["time_scope"] = "latest"
        return state

    meeting_ids = [
        c.get("meeting_index")
        for c in chunks
        if isinstance(c.get("meeting_index"), int)
    ]

    if not meeting_ids:
        state["question_intent"] = "factual"
        state["time_scope"] = "latest"
        return state

    latest_meeting = max(meeting_ids)


    latest_count = sum(1 for m in meeting_ids if m == latest_meeting)
    total = len(meeting_ids)
    latest_ratio = latest_count / total


    if latest_ratio >= 0.65:
        # Strong single-meeting dominance
        state["question_intent"] = "factual"
        state["time_scope"] = "latest"
        return state

    if latest_ratio <= 0.40:
        # Clearly spread across meetings
        state["question_intent"] = "meta"
        state["time_scope"] = "global"
        return state


    question = state.get(
        "standalone_query", state["question"]
    ).lower()

    META_HINTS = [
        "overall", "architecture", "design",
        "system", "approach", "workflow",
        "how does the project", "high level"
    ]

    if any(h in question for h in META_HINTS):
        state["question_intent"] = "meta"
        state["time_scope"] = "global"
    else:
        state["question_intent"] = "factual"
        state["time_scope"] = "latest"

    return state



def post_retrieve_router(state: MeetingState):


    q = state.get("question", "").lower()
    sq = state.get("standalone_query", "").lower()

    SUMMARY_KEYS = {
        "summary", "summarize", "overview",
        "highlights", "takeaways"
    }

    ACTION_KEYS = {
        "next step", "next steps",
        "steps decided", "decisions",
        "what was decided", "what were decided",
        "action item", "action items",
        "what to do next",
        "follow up", "follow-up",
        "things decided",
        "immediate steps",
        "plan decided"
    }

    DISCUSSION_KEYS = {
        "discussed about",
        "talked about",
        "two agent",
        "agent thing",
        "architecture",
        "approach",
        "design",
        "workflow"
    }

    def matches(keys, text):
        return any(k in text for k in keys)

    #  Explicit summary ONLY
    if matches(SUMMARY_KEYS, q) or matches(SUMMARY_KEYS, sq):
        return "meeting_summary"

    #  Decisions / steps / actions ( HIGH PRIORITY)
    if matches(ACTION_KEYS, q) or matches(ACTION_KEYS, sq):
        return "action_summary"

    # Conceptual / discussion questions
    if matches(DISCUSSION_KEYS, q) or matches(DISCUSSION_KEYS, sq):
        return "chunk_answer"

    #  Default → factual QA
    return "chunk_answer"



def action_summary_node(state: MeetingState):

    state["path"].append("action_summary")

    retrieved = state.get("retrieved_chunks", [])

    if not retrieved:
        state["final_answer"] = SAFE_ABSTAIN
        state["method"] = "action_no_evidence"
        return state

    # Ensure single meeting (latest already filtered upstream)
    meeting_ids = {
        c["meeting_index"]
        for c in retrieved
        if isinstance(c.get("meeting_index"), int)
    }

    if len(meeting_ids) != 1:
        state["final_answer"] = SAFE_ABSTAIN
        state["method"] = "action_mixed_meetings"
        return state

    context = "\n\n".join(
        c["text"] for c in retrieved if isinstance(c.get("text"), str)
    )[:12000]

    prompt = f"""
You are extracting ACTION ITEMS ONLY.

RULES:
- Use ONLY the transcript text
- List ONLY concrete next steps / tasks
- Do NOT include goals or explanations
- If no action items are explicit, say exactly:
"{SAFE_ABSTAIN}"

TRANSCRIPT:
{context}

ACTION ITEMS:
""".strip()

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )

        state["final_answer"] = response.choices[0].message.content.strip()
        state["method"] = "action_summary_latest"

    except Exception:
        state["final_answer"] = SAFE_ABSTAIN
        state["method"] = "action_error"

    state["context_extended"] = False
    return state





REFERENTIAL_WORDS = {"this", "that", "it", "those", "they"}

def is_referential_question(question: str) -> bool:
    tokens = question.lower().split()
    return any(t in REFERENTIAL_WORDS for t in tokens)


def has_explicit_antecedent(chunks: list) -> bool:
    """
    Checks whether the transcript explicitly resolves
    a referential term like 'this' or 'that'.
    """
    text = " ".join(
        c["text"].lower()
        for c in chunks
        if isinstance(c.get("text"), str)
    )

    EXPLICIT_PATTERNS = [
        "this refers to",
        "this means",
        "this is",
        "that refers to",
        "it refers to",
        "which refers to",
    ]

    return any(p in text for p in EXPLICIT_PATTERNS)



import re
import numpy as np

def generate_with_confidence(
    question: str,
    retrieved_chunks: list,
):
    if not retrieved_chunks:
        return SAFE_ABSTAIN, 0.0

    # Generate answer from evidence
    answer = generate_answer_with_llm(
        question=question,
        retrieved_chunks=retrieved_chunks
    )

    if not answer or answer.strip() == SAFE_ABSTAIN:
        return SAFE_ABSTAIN, 0.0

    # Build raw evidence text
    raw_evidence_text = " ".join(
        c.get("text", "").lower()
        for c in retrieved_chunks
        if isinstance(c.get("text"), str)
    )

    if not raw_evidence_text.strip():
        return SAFE_ABSTAIN, 0.0

    # Extract meaningful answer words
    def extract_keywords(text):
        return {
            w for w in re.findall(r"\b\w+\b", text.lower())
            if len(w) > 4
        }

    answer_words = extract_keywords(answer)

    lexical_overlap = sum(
        1 for w in answer_words
        if w in raw_evidence_text
    )

    #  PRIMARY ACCEPTANCE RULE
    if lexical_overlap >= 2:
        return answer, 0.25   # confidence is symbolic, not probabilistic

    # Debug only (safe)
    print("\n--- DEBUG generate_with_confidence ---")
    print("QUESTION:", question)
    print("ANSWER:", answer)
    print("ANSWER WORDS:", answer_words)
    print("LEXICAL OVERLAP:", lexical_overlap)
    print("------------------------------------\n")

    return SAFE_ABSTAIN, 0.0


    # Fallback: semantic entailment (existing logic)
    def split_sentences(text):
        return [
            s.strip()
            for s in re.split(r'(?<=[.!?])\s+', text)
            if len(s.split()) >= 4
        ]

    evidence_sentences = []
    for c in retrieved_chunks:
        if isinstance(c.get("text"), str):
            evidence_sentences.extend(split_sentences(c["text"]))

    if not evidence_sentences:
        return SAFE_ABSTAIN, 0.0

    model = get_entailment_model()

    answer_emb = model.encode(answer, normalize_embeddings=True)
    evidence_embs = model.encode(
        evidence_sentences,
        normalize_embeddings=True
    )

    sims = np.dot(evidence_embs, answer_emb)

    confidence = float(
        0.7 * np.max(sims) + 0.3 * np.mean(sims)
    )
    confidence = max(0.0, min(1.0, confidence))

    if confidence < 0.4:
        return SAFE_ABSTAIN, 0.0

    return answer, confidence


from memory.chat_utils import get_chat_texts
from memory.chat_vector_store import ChatVectorStore
import re


def chat_confidence(
    question: str,
    retrieved_chat_chunks: list[str],
) -> float:

    if not question or not retrieved_chat_chunks:
        return 0.0

    question = question.lower().strip()
    chat_text = " ".join(retrieved_chat_chunks).lower()

    def keywords(text: str):
        return {
            w for w in re.findall(r"\b[a-z]{4,}\b", text)
        }

    q_words = keywords(question)
    c_words = keywords(chat_text)

    if not q_words or not c_words:
        return 0.0

    overlap = q_words & c_words
    overlap_count = len(overlap)
    density = overlap_count / max(len(q_words), 1)

    # STRICT GATES
    if overlap_count >= 3 and density >= 0.4:
        return 0.6

    if overlap_count >= 2 and density >= 0.25:
        return 0.45

    if overlap_count == 1:
        return 0.25

    return 0.0


from agents.decision_types import Decision
from memory.chat_vector_store import chat_vector_store
from memory.chat_utils import get_chat_texts


def chat_recall_node(state: MeetingState):


    CHAT_CONFIDENCE_THRESHOLD = 0.55

    state.setdefault("path", [])
    state["path"].append("chat_recall")

    query = state.get("standalone_query", state["question"]).lower()


    FORCE_RETRIEVAL_KEYS = {
    "last meeting",
    "latest meeting",
    "next steps",
    "next step",
    "action items",
    "action item",
    "what was decided",
    "what were decided",
    "decisions",
    "follow up",
    "what to do next"
}

    if any(k in query for k in FORCE_RETRIEVAL_KEYS):
        print(" Forced RETRIEVAL (last meeting / actions policy)")
        state["decision"] = Decision.RETRIEVAL_ONLY
        state["confidence"] = 0.0
        state["method"] = "forced_latest_meeting_policy"
        return state

    REFERENTIAL_KEYS = {
        "summarize that",
        "summary of that",
        "explain that",
        "explain this",
        "in brief",
        "briefly",
        "short me",
        "summarize this"
    }

    if any(k in query for k in REFERENTIAL_KEYS):
        last_turn = get_chat_texts(
            session_id=state["session_id"],
            k=1
        )
        if last_turn:
            print(" Referential follow-up → CHAT_ONLY (last turn)")
            state["decision"] = Decision.CHAT_ONLY
            state["confidence"] = 0.9
            state["method"] = "referential_chat"
            return state


    chat_texts = get_chat_texts(
        session_id=state["session_id"],
        k=20
    )

    print("\n CHAT RECALL DEBUG")
    print("Query:", query)
    print("Chat texts count:", len(chat_texts))

    if not chat_texts:
        print(" No chat texts → RETRIEVAL")
        state["decision"] = Decision.RETRIEVAL_ONLY
        state["confidence"] = 0.0
        state["method"] = "chat_empty"
        return state


    print("\n CHAT VECTOR STORE DEBUG")
    print("Texts in vector store:", len(chat_vector_store.texts))
    print("FAISS empty:", chat_vector_store.is_empty())

    if chat_vector_store.is_empty():
        print(" FAISS empty → RETRIEVAL")
        state["decision"] = Decision.RETRIEVAL_ONLY
        state["confidence"] = 0.0
        state["method"] = "chat_vector_empty"
        return state


    results = chat_vector_store.search(
        query=query,
        k=3
    )

    print("\n CHAT FAISS RESULTS")
    for text, score in results:
        print(f"score={score:.3f} | {text[:120]}")

    if not results:
        print(" No semantic hit → RETRIEVAL")
        state["decision"] = Decision.RETRIEVAL_ONLY
        state["confidence"] = 0.0
        state["method"] = "chat_no_match"
        return state

    retrieved_texts = [t for (t, _) in results]


    conf = chat_confidence(
        question=query,
        retrieved_chat_chunks=retrieved_texts
    )

    print("\n CHAT CONFIDENCE DEBUG")
    print("confidence:", conf)

    state["confidence"] = conf


    if conf >= CHAT_CONFIDENCE_THRESHOLD:
        print(" High confidence → CHAT_ONLY")
        state["decision"] = Decision.CHAT_ONLY
        state["method"] = "chat_high_confidence"
    else:
        print(" Low confidence → RETRIEVAL")
        state["decision"] = Decision.RETRIEVAL_ONLY
        state["method"] = "chat_confidence_low"

    return state



from scripts.generate_answer import generate_answer_with_llm

from collections import deque

def chunk_answer_node(state: MeetingState):

    print("\n DEBUG: ENTERED chunk_answer_node")
    print("Retrieved chunks:", len(state.get("retrieved_chunks", [])))


    state["path"].append("chunk_answer")

    query = state.get("standalone_query", state["question"])
    retrieved = state.get("retrieved_chunks", [])

    print(f" INTELLIGENT QA: '{query}'")


    if not retrieved:
        state["candidate_answer"] = SAFE_ABSTAIN
        state["confidence"] = 0.0
        state["method"] = "no_evidence"
        return state


    retrieved = sorted(
        retrieved,
        key=lambda c: (c.get("meeting_index", 0), c.get("chunk_index", 0))
    )

    model = get_entailment_model()
    q_emb = model.encode(query, normalize_embeddings=True)

    texts = []
    aligned_chunks = []

    for c in retrieved:
        if isinstance(c.get("text"), str):
            texts.append(c["text"])   # NO TRIM FOR EMBEDDING
            aligned_chunks.append(c)

    if not texts:
        state["candidate_answer"] = SAFE_ABSTAIN
        state["confidence"] = 0.0
        state["method"] = "no_text_chunks"
        return state

    chunk_embs = model.encode(texts, normalize_embeddings=True)
 
    sims = np.dot(chunk_embs, q_emb)

    # Take top-K candidates (prevents missing declarative facts)
    TOP_K = min(5, len(sims))
    top_indices = np.argsort(sims)[-TOP_K:][::-1]

    candidate_chunks = [aligned_chunks[i] for i in top_indices]

    expanded_chunks = []
    seen = set()

    for idx in top_indices:
        base = aligned_chunks[idx]
        meeting = base.get("meeting_index")

        for j in (idx - 1, idx, idx + 1):
            if 0 <= j < len(aligned_chunks):
                ch = aligned_chunks[j]
                key = (ch.get("meeting_index"), ch.get("chunk_index"))
                if key not in seen and ch.get("meeting_index") == meeting:
                    expanded_chunks.append(ch)
                    seen.add(key)

    answer, confidence = generate_with_confidence(
        question=query,
        retrieved_chunks=expanded_chunks
    )

    if confidence <= 0.0 or not answer:
        state["candidate_answer"] = SAFE_ABSTAIN
        state["confidence"] = 0.0
        state["method"] = "not_in_transcript"
        return state

    state["candidate_answer"] = answer
    state["confidence"] = round(confidence, 3)
    state["method"] = "answer_entailment_verified"

    return state



    
def verification_node(state: MeetingState):


    state["path"].append("verify")

    raw_q = state["question"].lower()
    rewritten_q = state.get("standalone_query", "").lower()
    answer = (state.get("candidate_answer") or "").lower()

    CERTAINTY_KEYS = [
        "final", "finally decided", "confirmed", "approved",
        "mandatory", "must", "signed off", "fixed", "locked"
    ]

    NEXT_STEP_KEYS = [
        "next step", "next steps", "action item",
        "action items", "what to do next", "plan"
    ]

    def asks_for_certainty():
        return any(k in raw_q for k in CERTAINTY_KEYS) or \
               any(k in rewritten_q for k in CERTAINTY_KEYS)

    def asks_for_next_steps():
        return any(k in raw_q for k in NEXT_STEP_KEYS) or \
               any(k in rewritten_q for k in NEXT_STEP_KEYS)


    EXPLORATORY_PATTERNS = [
        "discussed", "suggested", "explored", "idea",
        "possible", "proposal", "considering", "might",
        "option", "could"
    ]

    CONFIRMATION_PATTERNS = [
        "decided", "finalized", "confirmed",
        "approved", "agreed", "will be implemented"
    ]

    has_exploratory = any(p in answer for p in EXPLORATORY_PATTERNS)
    has_confirmation = any(p in answer for p in CONFIRMATION_PATTERNS)

    # User wants certainty, answer is exploratory → BLOCK
    if asks_for_certainty() and has_exploratory:
        state["final_answer"] = (
            "The meeting explored this as an idea, but no final or "
            "confirmed decision was explicitly stated in the transcript."
        )
        state["method"] = "hard_decision_override"
        state["context_extended"] = False
        return state


    # Answer contains both exploratory + confirmation → CLARIFY
    if asks_for_certainty() and has_exploratory and has_confirmation:
        state["final_answer"] = (
            "The discussion included exploratory ideas, but the transcript "
            "does not clearly confirm this as a finalized or mandatory decision."
        )
        state["method"] = "mixed_certainty_override"
        state["context_extended"] = False
        return state

    # Exploratory answers are fine here
    if asks_for_next_steps():
        return state


    HYPOTHETICAL_PATTERNS = [
        "for example", "hypothetically",
        "imagine if", "let's say"
    ]

    if asks_for_certainty() and any(h in answer for h in HYPOTHETICAL_PATTERNS):
        state["final_answer"] = (
            "This was discussed as a hypothetical example, "
            "not as a confirmed instruction or decision."
        )
        state["method"] = "hypothetical_override"
        state["context_extended"] = False
        return state


    return state

# FINALIZE NODE (FINAL FIXED VERSION)

from agents.text_utils import clean_answer, SAFE_ABSTAIN
from agents.decision_types import Decision
from memory.session_memory import session_memory
from memory.chat_vector_store import chat_vector_store


def finalize_node(state: MeetingState):
    state.setdefault("path", [])
    state["path"].append("finalize")

    if state.get("final_answer"):
        answer = state["final_answer"]
    else:
        raw = state.get("candidate_answer")
        answer = clean_answer(raw) if raw else SAFE_ABSTAIN


    decision = state.get("decision")

    #  FINAL RULE:
    # Any user-visible answer is conversational memory
    if decision in (Decision.CHAT_ONLY, Decision.RETRIEVAL_ONLY):
        source = "chat"
    else:
        source = "system"

    meeting_index = None
    retrieved = state.get("retrieved_chunks")
    if (
        isinstance(retrieved, list)
        and retrieved
        and isinstance(retrieved[0], dict)
    ):
        meeting_index = retrieved[0].get("meeting_index")

    if answer != SAFE_ABSTAIN:
        session_memory.add_turn(
            session_id=state["session_id"],
            question=state["question"],
            answer=answer,
            source=source,
            meeting_index=meeting_index,
            method=state.get("method"),
            standalone_query=state.get("standalone_query"),
            time_scope=state.get("time_scope"),
            meeting_indices=state.get("meeting_indices"),
        )

        chat_text = f"User: {state['question']}\nAI: {answer}"
        chat_vector_store.add_texts([chat_text])

        print(" CHAT VECTOR STORE SIZE:", len(chat_vector_store.texts))

    state["final_answer"] = answer
    return state




graph = StateGraph(MeetingState)


graph.add_node("query", query_understanding_node)
graph.add_node("chat_recall", chat_recall_node)
graph.add_node("coordinator", coordinator_node)

graph.add_node("pure_chat", pure_chat_node)
graph.add_node("retrieve", retrieve_chunks_node)

graph.add_node("infer_intent", infer_intent_node)
graph.add_node("decide_source", decide_source_node)

graph.add_node("chunk_answer", chunk_answer_node)
graph.add_node("meeting_summary", meeting_summary_node)
graph.add_node("action_summary", action_summary_node)
graph.add_node("verify", verification_node)
graph.add_node("finalize", finalize_node)


graph.set_entry_point("query")


graph.add_edge("query", "chat_recall")
graph.add_edge("chat_recall", "coordinator")

graph.add_conditional_edges(
    "coordinator",
    lambda s: s["decision"],
    {
        Decision.CHAT_ONLY: "pure_chat",
        Decision.RETRIEVAL_ONLY: "retrieve",
        Decision.IGNORE: "finalize",
    }
)


graph.add_edge("retrieve", "infer_intent")
graph.add_edge("infer_intent", "decide_source")


graph.add_conditional_edges(
    "decide_source",
    post_retrieve_router,
    {
        "meeting_summary": "meeting_summary",
        "action_summary": "action_summary",
        "chunk_answer": "chunk_answer",
    }
)


graph.add_edge("pure_chat", "finalize")
graph.add_edge("meeting_summary", "finalize")
graph.add_edge("action_summary", "finalize")

graph.add_edge("chunk_answer", "verify")
graph.add_edge("verify", "finalize")

graph.add_edge("finalize", END)

meeting_graph = graph.compile()

