from typing import TypedDict, List, Optional
import json
from agents.text_utils import clean_answer
from agents.source_decider_agent import decide_source_node
from agents.text_utils import trim_chunk_text
from agents.decision_types import Decision
from services.supabase_service import SupabaseService
import numpy as np
from langgraph.graph import StateGraph, END
from groq import Groq
from scripts.generate_answer import generate_answer_with_llm
from sentence_transformers import SentenceTransformer
SAFE_ABSTAIN = "This was not clearly discussed in the meeting."


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
    meeting_indices: Optional[List[str]]
    _all_meeting_indices: Optional[List[str]]

    # Reasoning outputs (post-retrieval only)
    question_intent: Optional[str]          # factual | meta
    time_scope: Optional[str]               # latest | global

    # Answer lifecycle
    candidate_answer: Optional[str]
    final_answer: Optional[str]
    method: str
    context_extended: bool

    path: List[str]




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

    supabase = get_supabase_client()

    # -----------------------------------
    # Fetch recent chat safely
    # -----------------------------------
    try:
        recent_chat_raw = supabase.get_recent_chat_turns(
            session_id=state["session_id"],
            limit=100
        )
    except Exception as e:
        print("Chat fetch failed:", e)
        recent_chat_raw = []

    # -----------------------------------
    # SANITIZE CHAT FORMAT
    # -----------------------------------
    recent_chat = []

    for item in recent_chat_raw:
        if isinstance(item, dict):
            recent_chat.append({
                "question": item.get("question", ""),
                "answer": item.get("answer", "")
            })
        elif isinstance(item, str):
            # fallback format
            recent_chat.append({
                "question": "",
                "answer": item
            })

    # -----------------------------------
    # Call understand_query safely
    # -----------------------------------
    try:
        analysis = understand_query(
            state["question"],
            recent_history=recent_chat,
            user_id=state["user_id"]
        )
    except Exception as e:
        print("understand_query failed:", e)
        analysis = {}

    # If understand_query returns invalid type
    if not isinstance(analysis, dict):
        analysis = {}

    # -----------------------------------
    # Extract analysis safely
    # -----------------------------------
    state["ignore"] = bool(analysis.get("ignore", False))

    state["standalone_query"] = analysis.get(
        "standalone_query",
        state["question"]
    )

    # -----------------------------------
    # Temporal detection
    # -----------------------------------
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

    # -----------------------------------
    # Domain constraint
    # -----------------------------------
    project_type = analysis.get("project_type")
    force_single_project = analysis.get("force_single_project", False)

    if force_single_project and project_type:
        state["domain_constraint"] = project_type
    else:
        state["domain_constraint"] = None

    # -----------------------------------
    # Reset downstream state
    # -----------------------------------
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

    # -----------------------------
    # Get first meeting (similarity-sorted already)
    # -----------------------------
    first_meeting_id = retrieved[0].get("meeting_id")

    if not first_meeting_id:
        state["final_answer"] = SAFE_ABSTAIN
        state["method"] = "summary_no_meeting_id"
        state["context_extended"] = False
        return state

    # -----------------------------
    # Keep only chunks from that meeting
    # -----------------------------
    meeting_chunks = [
        c for c in retrieved
        if c.get("meeting_id") == first_meeting_id
        and isinstance(c.get("text"), str)
    ]

    if not meeting_chunks:
        state["final_answer"] = SAFE_ABSTAIN
        state["method"] = "summary_empty_meeting"
        state["context_extended"] = False
        return state

    # Sort by chunk_index for proper transcript order
    meeting_chunks = sorted(
        meeting_chunks,
        key=lambda c: c.get("chunk_index", 0)
    )

    context = "\n\n".join(
        c["text"] for c in meeting_chunks
    )[:12000]

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
        state["method"] = "meeting_summary_uuid_safe"

    except Exception as e:
        print(f"Summary Error: {e}")
        state["final_answer"] = SAFE_ABSTAIN
        state["method"] = "summary_error"

    state["context_extended"] = False
    return state



# ==============================
# GLOBAL VECTOR MODEL + SUPABASE
# ==============================

_VECTOR_MODEL = None
_SUPABASE_CLIENT = None


def get_vector_model():
    global _VECTOR_MODEL
    if _VECTOR_MODEL is None:
        print("Loading vector model...")
        _VECTOR_MODEL = SentenceTransformer(
            "BAAI/bge-base-en-v1.5"
        )
        print("Vector model loaded.")
    return _VECTOR_MODEL


def get_supabase_client():
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is None:
        _SUPABASE_CLIENT = SupabaseService()
    return _SUPABASE_CLIENT




def retrieve_chunks_node(state: MeetingState):

    state["path"].append("retrieve_chunks")

    query_text = state.get(
        "standalone_query",
        state["question"]
    )

    model = get_vector_model()
    supabase = get_supabase_client()

    # -----------------------------------
    # Generate embedding safely
    # -----------------------------------
    try:
        query_embedding = model.encode(
            query_text,
            normalize_embeddings=True
        ).tolist()
    except Exception:
        state["retrieved_chunks"] = []
        state["_all_meeting_indices"] = []
        state["meeting_indices"] = []
        return state

    # -----------------------------------
    # Supabase RPC call (safe)
    # -----------------------------------
    try:
        results = supabase.match_chunks_by_user(
            query_embedding=query_embedding,
            user_id=state["user_id"],
            match_count=50
        )
    except Exception:
        results = []

    if not results:
        state["retrieved_chunks"] = []
        state["_all_meeting_indices"] = []
        state["meeting_indices"] = []
        return state

    # -----------------------------------
    # Convert Supabase rows → clean format
    # -----------------------------------
    clean_chunks = []

    for row in results:
        if (
            isinstance(row.get("meeting_id"), str)
            and isinstance(row.get("chunk_index"), int)
            and isinstance(row.get("chunk_text"), str)
        ):
            clean_chunks.append({
                "meeting_id": row["meeting_id"],
                "chunk_index": row["chunk_index"],
                "text": row["chunk_text"],
                "similarity": float(row.get("similarity", 0.0)),
            })

    if not clean_chunks:
        state["retrieved_chunks"] = []
        state["_all_meeting_indices"] = []
        state["meeting_indices"] = []
        return state

    # -----------------------------------
    # Sort by similarity FIRST
    # -----------------------------------
    clean_chunks = sorted(
        clean_chunks,
        key=lambda c: c["similarity"],
        reverse=True
    )

    # -----------------------------------
    # Temporal filtering
    # -----------------------------------
    if state.get("temporal_constraint") == "latest":
        first_meeting = clean_chunks[0]["meeting_id"]

        clean_chunks = [
            c for c in clean_chunks
            if c["meeting_id"] == first_meeting
        ]

    # -----------------------------------
    # Sort by transcript order
    # -----------------------------------
    clean_chunks = sorted(
        clean_chunks,
        key=lambda c: (c["meeting_id"], c["chunk_index"])
    )

    # -----------------------------------
    # Collect meeting IDs safely
    # -----------------------------------
    meeting_ids = []
    for c in clean_chunks:
        mid = c["meeting_id"]
        if mid not in meeting_ids:
            meeting_ids.append(mid)

    state["retrieved_chunks"] = clean_chunks
    state["_all_meeting_indices"] = meeting_ids
    state["meeting_indices"] = meeting_ids

    return state



def infer_intent_node(state: MeetingState):
    """
    Clean intent inference.
    - Meta ONLY if explicitly meta.
    - Never infer meta based on number of meetings.
    - Safe for numeric / factual queries.
    """

    state.setdefault("path", [])
    state["path"].append("infer_intent")

    question = state.get("standalone_query", state["question"]).lower()
    chunks = state.get("retrieved_chunks", [])

    # ---------------------------------
    # Default fallback
    # ---------------------------------
    state["question_intent"] = "factual"
    state["time_scope"] = "latest"

    if not chunks:
        return state

    meeting_ids = [
        c.get("meeting_id")
        for c in chunks
        if isinstance(c.get("meeting_id"), str)
    ]

    if not meeting_ids:
        return state

    unique_meetings = list(dict.fromkeys(meeting_ids))


    META_HINTS = [
        "overall",
        "architecture",
        "design",
        "workflow",
        "approach",
        "system",
        "how does the system",
        "how did the system",
        "high level",
        "in general",
        "big picture",
    ]

    if any(hint in question for hint in META_HINTS):
        state["question_intent"] = "meta"
        state["time_scope"] = "global"
        return state


    state["question_intent"] = "factual"

    if len(unique_meetings) == 1:
        state["time_scope"] = "latest"
    else:
        state["time_scope"] = "mixed"

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
        c["meeting_id"]
        for c in retrieved
        if isinstance(c.get("meeting_id"), str)
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

    # No evidence
    if not retrieved_chunks:
        print("No retrieved chunks → ABSTAIN")
        return SAFE_ABSTAIN, 0.0

    # Generate answer strictly from provided chunks
    answer = generate_answer_with_llm(
        question=question,
        retrieved_chunks=retrieved_chunks
    )

    print("LLM RAW ANSWER:", repr(answer))

    # LLM abstained
    if not answer or answer.strip() == SAFE_ABSTAIN:
        print("LLM abstained → ABSTAIN")
        return SAFE_ABSTAIN, 0.0

    # Light safety check: ensure some overlap with evidence
    raw_evidence_text = " ".join(
        c.get("text", "").lower()
        for c in retrieved_chunks
        if isinstance(c.get("text"), str)
    )

    if not raw_evidence_text.strip():
        print("Empty evidence text → ABSTAIN")
        return SAFE_ABSTAIN, 0.0

    # Very light grounding check (not aggressive)
    answer_tokens = [
        w for w in re.findall(r"\b\w+\b", answer.lower())
        if len(w) > 4
    ]

    overlap = sum(
        1 for w in answer_tokens
        if w in raw_evidence_text
    )

    print("Light lexical overlap:", overlap)

    if overlap >= 1:
        return answer, 0.6

    # If LLM answered but overlap is weak,
    # still allow but with lower confidence.
    return answer, 0.4





from scripts.generate_answer import generate_answer_with_llm
from collections import deque

def chunk_answer_node(state: MeetingState):

    print("\nDEBUG: ENTERED chunk_answer_node")
    print("Retrieved chunks:", len(state.get("retrieved_chunks", [])))

    state["path"].append("chunk_answer")

    query = state.get("standalone_query", state["question"])
    retrieved = state.get("retrieved_chunks", [])

    if not retrieved:
        state["candidate_answer"] = SAFE_ABSTAIN
        state["confidence"] = 0.0
        state["method"] = "no_evidence"
        return state

    # ---------------------------------------------------
    #GLOBAL TOP-K (NO MEETING DOMINANCE)
    # ---------------------------------------------------

    # Sort all retrieved chunks by similarity
    sorted_chunks = sorted(
        retrieved,
        key=lambda c: c.get("similarity", 0.0),
        reverse=True
    )

    MAX_CONTEXT_CHUNKS = 12
    selected_chunks = sorted_chunks[:MAX_CONTEXT_CHUNKS]

    print(f"\nChunks passed to LLM: {len(selected_chunks)}")

    # ---------------------------------------------------
    # Generate answer
    # ---------------------------------------------------
    answer, confidence = generate_with_confidence(
        question=query,
        retrieved_chunks=selected_chunks
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

def chat_answer_node(state: MeetingState):
    """
    CHAT-FIRST NODE (Production Hardened)

    Behaviour:
    - If chat is sufficient → ALWAYS answer from chat
    - No fallback after YES
    - Yes/No answers must include explanation
    - Semantic comparison (case/wording safe)
    """

    state.setdefault("path", [])
    state["path"].append("chat_answer")

    supabase = get_supabase_client()

    print("\n================= CHAT NODE START =================")

    # --------------------------------------------------
    # Fetch chat history
    # --------------------------------------------------
    try:
        chat_lines = supabase.get_recent_chat_turns(
            session_id=state["session_id"],
            limit=50
        )
    except Exception as e:
        print("Chat fetch failed:", e)
        chat_lines = []

    print("Session:", state["session_id"])
    print("Question:", state["question"])
    print("Chat lines:", len(chat_lines))

    if not chat_lines:
        print(" No chat history → Retrieval")
        state["decision"] = Decision.RETRIEVAL_ONLY
        state["method"] = "chat_no_context"
        return state

    # --------------------------------------------------
    # Build structured chat context
    # --------------------------------------------------
    chat_context_parts = []

    for c in chat_lines:
        if isinstance(c, dict):
            q = c.get("question", "")
            a = c.get("answer", "")
            chat_context_parts.append(f"User: {q}\nAI: {a}")
        elif isinstance(c, str):
            chat_context_parts.append(c)

    chat_context = "\n\n".join(chat_context_parts).strip()

    print("\n========== CHAT CONTEXT SENT TO SUFFICIENCY ==========")
    print(chat_context)
    print("======================================================\n")


    if not chat_context:
        print(" Empty chat context → Retrieval")
        state["decision"] = Decision.RETRIEVAL_ONLY
        state["method"] = "chat_empty_context"
        return state

    # --------------------------------------------------
    # Semantic Sufficiency Check (Anti-Sabji Mode)
    # --------------------------------------------------
    assistant_only = "\n\n".join(
        part for part in chat_context_parts
        if part.startswith("AI:")
    )

    sufficiency_prompt = f"""
You are checking whether the QUESTION can be answered
using information already provided in earlier ASSISTANT responses.

IF you can answer Reply with YES 
IF you can not answer(means answer is not in context) reply with NO

ASSISTANT MESSAGES:
{assistant_only}

QUESTION:
{state["question"]}
""".strip()


    try:
        verdict_resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": sufficiency_prompt}],
            temperature=0.0,
            max_tokens=3,
        )

        raw_verdict = verdict_resp.choices[0].message.content.strip()
        verdict = raw_verdict.upper().split()[0]

        print("Sufficiency RAW:", raw_verdict)
        print("Sufficiency PARSED:", verdict)

    except Exception as e:
        print("Sufficiency error:", e)
        verdict = "NO"

    if verdict != "YES":
        print(" Retrieval path")
        state["decision"] = Decision.RETRIEVAL_ONLY
        state["method"] = "chat_insufficient"
        return state

    print("Chat sufficient → Generating answer")

    # --------------------------------------------------
    # Answer Generation (Explained Yes/No Mode)
    # --------------------------------------------------
    answer_prompt = f"""
Answer the QUESTION using ONLY the CHAT below.

Rules:
- If answer is yes/no, start with:
  "Yes," or "No,"
- Always explain briefly in one clear sentence.
- Do NOT just say Yes or No alone.
- Be concise.
- Do NOT mention chat history.

CHAT:
{chat_context}

QUESTION:
{state["question"]}

FINAL ANSWER:
""".strip()

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": answer_prompt}],
            temperature=0.0,
            max_tokens=200,
        )

        answer = response.choices[0].message.content.strip()

        print("Chat Answer RAW:", repr(answer))

    except Exception as e:
        print("Answer generation error:", e)
        answer = SAFE_ABSTAIN

    if not answer:
        answer = SAFE_ABSTAIN

    # --------------------------------------------------
    # Finalize (No fallback allowed)
    # --------------------------------------------------
    state["final_answer"] = answer
    state["decision"] = Decision.CHAT_ONLY
    state["method"] = "chat_context_answer_final"
    state["context_extended"] = False
    state["retrieved_chunks"] = []
    state["meeting_indices"] = None

    print("================= CHAT NODE END =================\n")

    return state






from agents.text_utils import clean_answer, SAFE_ABSTAIN
from agents.decision_types import Decision
from services.supabase_service import get_supabase_client


from services.supabase_service import get_supabase_client


def finalize_node(state: MeetingState):
    state.setdefault("path", [])
    state["path"].append("finalize")

    supabase = get_supabase_client()

    decision = state.get("decision")
    method = state.get("method", "")

    # -----------------------------------
    # CHAT-ONLY ANSWER
    # -----------------------------------
    if decision == Decision.CHAT_ONLY:
        answer = state.get("final_answer") or SAFE_ABSTAIN

        if answer != SAFE_ABSTAIN:
            supabase.save_chat_turn(
                session_id=state["session_id"],
                user_id=state["user_id"],
                question=state["question"],
                answer=answer,
                source="chat",
                meeting_id=None,
                method=method,
                time_scope=None,
            )

        state["final_answer"] = answer
        return state

    # -----------------------------------
    # RETRIEVAL / SYSTEM ANSWER
    # -----------------------------------
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

    # Extract meeting_id (UUID-safe)
    meeting_id = None
    retrieved = state.get("retrieved_chunks")

    if (
        isinstance(retrieved, list)
        and retrieved
        and isinstance(retrieved[0], dict)
    ):
        meeting_id = retrieved[0].get("meeting_id")

    # -----------------------------------
    # Save to Supabase
    # -----------------------------------
    if answer != SAFE_ABSTAIN:
        supabase.save_chat_turn(
            session_id=state["session_id"],
            user_id=state["user_id"],
            question=state["question"],
            answer=answer,
            source=source,
            meeting_id=meeting_id,
            method=method,
            time_scope=state.get("time_scope"),
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

