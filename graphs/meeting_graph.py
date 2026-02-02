from typing import TypedDict, List, Optional
import json
from agents.text_utils import clean_answer
from agents.source_decider_agent import decide_source_node
from agents.text_utils import trim_chunk_text
from agents.decision_types import Decision
from memory.chat_utils import get_chat_texts


import numpy as np

from langgraph.graph import StateGraph, END


from groq import Groq
from scripts.generate_answer import (
    retrieve_chunks, 
    generate_answer_with_llm, 
    CHUNKS_PATH 
)


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



# QUERY UNDERSTANDING NODE (UPDATED — CHAT AS CONTEXT)

from agents.query_understanding_agent import understand_query


def query_understanding_node(state: MeetingState):

    state.setdefault("path", [])
    state["path"].append("query")

    # Provide full recent chat context for reasoning (NO retrieval)
    recent_chat = get_chat_texts(
        session_id=state["session_id"],
        k=20
    )

    analysis = understand_query(
        state["question"],
        recent_history=recent_chat,
        user_id=state["user_id"]
    )

    # Basic flags
    state["ignore"] = bool(analysis.get("ignore", False))
    state["standalone_query"] = analysis.get(
        "standalone_query", state["question"]
    )

    q = state["question"].lower()

    # Temporal intent detection (lightweight, lexical)
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

    # Domain constraint (project isolation)
    project_type = analysis.get("project_type")
    state["domain_constraint"] = (
        project_type
        if analysis.get("force_single_project") and project_type
        else None
    )

    # Reset downstream state (important for multi-turn safety)
    state["candidate_answer"] = None
    state["final_answer"] = None
    state["retrieved_chunks"] = []
    state["meeting_indices"] = None
    state["_all_meeting_indices"] = None
    state["context_extended"] = False
    state["method"] = ""
    state["confidence"] = None
    state["decision"] = None

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
    print("\n========== generate_with_confidence ==========")
    print("QUESTION:", question)
    print("NUM CHUNKS:", len(retrieved_chunks))

    if not retrieved_chunks:
        print(" No retrieved chunks → ABSTAIN")
        return SAFE_ABSTAIN, 0.0

    # Generate answer strictly from evidence
    answer = generate_answer_with_llm(
        question=question,
        retrieved_chunks=retrieved_chunks
    )

    print("LLM RAW ANSWER:", repr(answer))

    if not answer or answer.strip() == SAFE_ABSTAIN:
        print(" LLM abstained → ABSTAIN")
        return SAFE_ABSTAIN, 0.0

    # Build raw evidence text
    raw_evidence_text = " ".join(
        c.get("text", "").lower()
        for c in retrieved_chunks
        if isinstance(c.get("text"), str)
    )

    print("EVIDENCE TEXT LENGTH:", len(raw_evidence_text))
    print("EVIDENCE PREVIEW (first 400 chars):")
    print(raw_evidence_text[:400])

    if not raw_evidence_text.strip():
        print("Empty evidence text → ABSTAIN")
        return SAFE_ABSTAIN, 0.0

    q_lower = question.lower()


    if (
        q_lower.startswith("what is")
        or q_lower.startswith("what does")
        or q_lower.startswith("define")
    ):
        tokens = re.findall(r"\b\w+\b", q_lower)
        print("Definition-style question detected")
        print("Question tokens:", tokens)

        if tokens:
            entity = tokens[-1]
            print("Extracted entity:", entity)
            print("Entity present in evidence?:", entity in raw_evidence_text)

            if entity in raw_evidence_text:
                print("Definition grounding PASSED")
                return answer, 0.3
            else:
                print("Definition grounding FAILED")


    def extract_keywords(text):
        return {
            w for w in re.findall(r"\b\w+\b", text.lower())
            if len(w) > 4
        }

    answer_words = extract_keywords(answer)
    print("Answer keywords:", answer_words)

    lexical_overlap = sum(
        1 for w in answer_words
        if w in raw_evidence_text
    )

    print("Lexical overlap count:", lexical_overlap)

    if lexical_overlap >= 1:
        return answer, 0.25


    return SAFE_ABSTAIN, 0.0




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

    # Take top-K semantic candidates
    TOP_K = min(5, len(sims))
    top_indices = np.argsort(sims)[-TOP_K:][::-1]

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

    # 🔒 HARD CAP — THIS IS THE FIX
    MAX_CONTEXT_CHUNKS = 4
    expanded_chunks = expanded_chunks[:MAX_CONTEXT_CHUNKS]

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

from agents.text_utils import clean_answer, SAFE_ABSTAIN
from agents.decision_types import Decision
from memory.chat_utils import get_chat_texts
from groq import Groq

client = Groq()


def chat_answer_node(state: MeetingState):
    """
    CHAT ANSWER NODE (CHAT-FIRST, LLM-DECIDES)

    CONTRACT:
    - Pass full chat context to LLM
    - LLM decides if it can answer from chat alone
    - If YES → answer from chat (normal, helpful answer)
    - If NO → fallback to retrieval
    """

    state.setdefault("path", [])
    state["path"].append("chat_answer")

    chat_lines = get_chat_texts(
        session_id=state["session_id"],
        k=20
    )

    print("\n CHAT ANSWER DEBUG")
    print("Chat lines:", len(chat_lines))

    # No chat → retrieval
    if not chat_lines:
        state["decision"] = Decision.RETRIEVAL_ONLY
        state["method"] = "chat_no_context"
        return state

    chat_context = "\n".join(chat_lines)

    # =================================================
    # 1️Can chat answer this question?
    # =================================================
    sufficiency_prompt = f"""
You are given a chat history and a user question.

Decide whether the question can be answered
using the chat conversation alone.

Rules:
- Use semantic understanding
- You may rely on earlier answers or summaries
- If chat is sufficient, reply YES
- If transcript knowledge is needed, reply NO

Reply with exactly one word:
YES or NO

CHAT:
{chat_context}

QUESTION:
{state["question"]}
""".strip()

    try:
        verdict_resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": sufficiency_prompt}],
            temperature=0.0,
            max_tokens=5,
        )

        verdict = verdict_resp.choices[0].message.content.strip().upper()
        verdict = verdict.split()[0]

        print("CHAT SUFFICIENCY:", verdict)

    except Exception as e:
        print("[CHAT SUFFICIENCY ERROR]", e)
        verdict = "NO"

    # Not sufficient → retrieval
    if verdict != "YES":
        state["decision"] = Decision.RETRIEVAL_ONLY
        state["method"] = "chat_insufficient"
        return state


    answer_prompt = f"""
    Answer the user's question directly.

    RULES:
    - Treat the chat as factual conversation history
    - Use ONLY information explicitly present in the chat
    - Do NOT mention the chat, AI, assistant, or messages
    - Do NOT say phrases like "based on the chat" or "the AI said"
    - Explain ONLY if the explanation already exists in the chat
    - If the question cannot be answered from the chat, reply EXACTLY:
    "{SAFE_ABSTAIN}"

    CHAT CONTEXT:
    {chat_context}

    USER QUESTION:
    {state["question"]}

    FINAL ANSWER:
    """.strip()


    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": answer_prompt}],
            temperature=0.2,
            max_tokens=200,
        )

        answer = clean_answer(
            response.choices[0].message.content.strip()
        )

        print("CHAT ANSWER:", repr(answer))

    except Exception as e:
        print("[CHAT ANSWER ERROR]", e)
        answer = SAFE_ABSTAIN

    # If LLM still can’t answer → fallback
    if not answer or answer == SAFE_ABSTAIN:
        state["decision"] = Decision.RETRIEVAL_ONLY
        state["method"] = "chat_answer_failed"
        return state


    state["final_answer"] = answer
    state["decision"] = Decision.CHAT_ONLY
    state["method"] = "chat_context_answer"
    state["context_extended"] = False

    # Cleanup
    state["retrieved_chunks"] = []
    state["meeting_indices"] = None

    print("Chat answer accepted (chat-first)")

    return state






from agents.text_utils import clean_answer, SAFE_ABSTAIN
from agents.decision_types import Decision
from memory.session_memory import session_memory


def finalize_node(state: MeetingState):
    state.setdefault("path", [])
    state["path"].append("finalize")

    decision = state.get("decision")
    method = state.get("method", "")


    if decision == Decision.CHAT_ONLY:
        answer = state.get("final_answer") or SAFE_ABSTAIN

        if answer != SAFE_ABSTAIN:
            session_memory.add_turn(
                session_id=state["session_id"],
                question=state["question"],
                answer=answer,
                source="chat",
                meeting_index=None,  # chat has NO meeting index
                method=method,
                standalone_query=state.get("standalone_query"),
                time_scope=None,
                meeting_indices=None,
            )

        state["final_answer"] = answer
        return state


    existing_final = state.get("final_answer")
    if existing_final == SAFE_ABSTAIN:
        existing_final = None

    if existing_final:
        answer = existing_final

    elif method == "answer_entailment_verified":
        answer = state.get("candidate_answer") or SAFE_ABSTAIN

    else:
        raw = state.get("candidate_answer")
        answer = clean_answer(raw) if raw else SAFE_ABSTAIN

    source = "system"


    meeting_index = None
    retrieved = state.get("retrieved_chunks")
    if (
        isinstance(retrieved, list)
        and retrieved
        and isinstance(retrieved[0], dict)
    ):
        meeting_index = retrieved[0].get("meeting_index")

    # --------------------
    # Persist conversation
    # --------------------
    if answer != SAFE_ABSTAIN:
        session_memory.add_turn(
            session_id=state["session_id"],
            question=state["question"],
            answer=answer,
            source=source,
            meeting_index=meeting_index,
            method=method,
            standalone_query=state.get("standalone_query"),
            time_scope=state.get("time_scope"),
            meeting_indices=state.get("meeting_indices"),
        )

    state["final_answer"] = answer
    return state





graph = StateGraph(MeetingState)

# --------------------
# Nodes
# --------------------
graph.add_node("query", query_understanding_node)

# REPLACE pure_chat WITH chat_answer
graph.add_node("chat_answer", chat_answer_node)

graph.add_node("retrieve", retrieve_chunks_node)
graph.add_node("infer_intent", infer_intent_node)
graph.add_node("decide_source", decide_source_node)

graph.add_node("chunk_answer", chunk_answer_node)
graph.add_node("meeting_summary", meeting_summary_node)
graph.add_node("action_summary", action_summary_node)
graph.add_node("verify", verification_node)
graph.add_node("finalize", finalize_node)


graph.set_entry_point("query")


graph.add_edge("query", "chat_answer")

graph.add_conditional_edges(
    "chat_answer",
    lambda s: (
        "finalize"
        if s.get("decision") == Decision.CHAT_ONLY
        else "retrieve"
    ),
    {
        "finalize": "finalize",
        "retrieve": "retrieve",
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


graph.add_edge("meeting_summary", "finalize")
graph.add_edge("action_summary", "finalize")

graph.add_edge("chunk_answer", "verify")
graph.add_edge("verify", "finalize")

graph.add_edge("finalize", END)

meeting_graph = graph.compile()

