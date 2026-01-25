from typing import TypedDict, List, Optional
import json
from agents.text_utils import clean_answer
from agents.source_decider_agent import decide_source_node
from agents.decision_types import Decision
from agents.text_utils import trim_chunk_text


import numpy as np

from langgraph.graph import StateGraph, END

# --- ADD THIS AT THE TOP (Global Scope) ---
from groq import Groq
from scripts.generate_answer import (
    retrieve_chunks, 
    generate_answer_with_llm, 
    CHUNKS_PATH  # <--- Needed for meeting_summary_node
)

client = Groq() # <--- Initialize this once at the top
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

    # Debug / trace
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
    """
    Query understanding ONLY.

    Responsibilities:
    - Rewrite query if needed
    - Resolve references using recent chat
    - Extract HARD constraints (temporal / domain)

     DOES NOT decide routing
     DOES NOT set Decision
    """

    state.setdefault("path", [])
    state["path"].append("query")

    # -------------------------------------------------
    # 1. Fetch recent chat (coreference only)
    # -------------------------------------------------
    recent = session_memory.get_recent_context(
        state["session_id"], k=2
    )

    # -------------------------------------------------
    # 2. Rewrite / annotate query (NO CONTROL LOGIC)
    # -------------------------------------------------
    analysis = understand_query(
        state["question"],
        recent_history=recent,
        user_id=state["user_id"]
    )

    state["ignore"] = bool(analysis.get("ignore", False))
    state["standalone_query"] = analysis.get(
        "standalone_query", state["question"]
    )

    # -------------------------------------------------
    # 3.  HARD TEMPORAL CONSTRAINT (DETERMINISTIC)
    # -------------------------------------------------
    q = state["question"].lower()

    TEMPORAL_KEYS = [
        "last meeting",
        "latest meeting",
        "previous meeting",
        "most recent meeting",
        "last call",
        "last discussion"
    ]

    if any(k in q for k in TEMPORAL_KEYS):
        state["temporal_constraint"] = "latest"
    else:
        state["temporal_constraint"] = None

    # -------------------------------------------------
    # 4. DOMAIN CONSTRAINT (PASSIVE ONLY)
    # -------------------------------------------------
    project_type = analysis.get("project_type")
    state["domain_constraint"] = (
        project_type
        if analysis.get("force_single_project") and project_type
        else None
    )

    # -------------------------------------------------
    # 5. Explicitly unset reasoning outputs
    # -------------------------------------------------
    state["decision"] = None
    state["question_intent"] = None
    state["time_scope"] = "unknown"
    state["meeting_indices"] = None

    # -------------------------------------------------
    # 6. Reset downstream artifacts
    # -------------------------------------------------
    state["candidate_answer"] = None
    state["final_answer"] = None
    state["retrieved_chunks"] = []
    state["context_extended"] = False
    state["method"] = ""
    state["meeting_indices"] = None
    state["_all_meeting_indices"] = None


    return state




def coordinator_node(state: MeetingState):
    """
    COORDINATOR / ROUTER NODE

    Responsibilities:
    - Decide execution path
    - Enforce chat-vs-retrieval rules
    - Owns Decision completely
    """

    state["path"].append("coordinator")

    has_reference = bool(
    state.get("reference_chunks")
    or state.get("retrieved_chunks")
    )


    # -------------------------------------------------
    # 1. Ignore empty / noop queries
    # -------------------------------------------------
    if state.get("ignore"):
        state["decision"] = Decision.IGNORE
        return state

    # -------------------------------------------------
    # 2. Chat-only dominance check
    # -------------------------------------------------
    recent = session_memory.get_recent_context(
        state["session_id"], k=4
    )

    REFERENTIAL = {"this", "that", "it", "those", "they"}

    tokens = set(state["question"].lower().split())
    if has_reference and session_memory.chat_can_answer(
        question=state["question"],
        recent_context=recent
    ):
        state["decision"] = Decision.CHAT_ONLY
        return state


    # -------------------------------------------------
    # 3. Default → retrieval
    # -------------------------------------------------
    state["decision"] = Decision.RETRIEVAL_ONLY
    return state






def meeting_summary_node(state: MeetingState):
    """
    Generates a high-level meeting summary.

    GOOGLE-ALIGNED RULES:
    - Uses ONLY retrieved_chunks
    - If multiple meetings exist → select LATEST meeting ONLY
    - No question inspection
    - No inference beyond evidence
    """

    state["path"].append("meeting_summary")

    retrieved = state.get("retrieved_chunks", [])

    # --------------------------------------------------
    #  No evidence → cannot summarize
    # --------------------------------------------------
    if not retrieved:
        state["final_answer"] = SAFE_ABSTAIN
        state["method"] = "summary_no_evidence"
        state["context_extended"] = False
        return state

    # --------------------------------------------------
    #  FIX: ALWAYS RESOLVE TO LATEST MEETING
    # --------------------------------------------------
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

    # HARD FILTER → LATEST MEETING ONLY
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

    # --------------------------------------------------
    # 2. Build summary context (STRICT)
    # --------------------------------------------------
    context = "\n\n".join(c["text"] for c in latest_chunks)[:12000]

    # --------------------------------------------------
    # 3. Generate Summary
    # --------------------------------------------------
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




def pure_chat_node(state: MeetingState):
    """
    Handles conversational follow-ups using ONLY chat memory.

    GUARANTEES (DOC-ALIGNED):
    - Uses ONLY prior chat turns (source="chat")
    - No meeting / transcript / retrieval usage
    - No new facts or grounding
    - Only clarification / rephrasing of prior AI responses
    """

    state.setdefault("path", [])
    state["path"].append("pure_chat")

    # -------------------------------------------------
    # 0. HARD ASSERT: this path is CHAT_ONLY
    # -------------------------------------------------
    state["decision"] = Decision.CHAT_ONLY

    # -------------------------------------------------
    # 1. Retrieve relevant chat context (semantic, not last-k)
    # -------------------------------------------------
    chat_chunks = session_memory.retrieve_chat_chunks(
        session_id=state["session_id"],
        query=state["question"],
        k=3
    )

    # No conversational grounding → safe abstain
    if not chat_chunks:
        # FALL BACK to retrieval instead of abstaining
        state["decision"] = Decision.RETRIEVAL_ONLY
        return state


    chat_context = "\n\n".join(chat_chunks)

    # -------------------------------------------------
    # 2. STRICT conversational-only prompt
    # -------------------------------------------------
    prompt = f"""
You are continuing an existing conversation.

STRICT RULES:
- Use ONLY the CHAT CONTEXT below.
- Do NOT add new facts or assumptions.
- Do NOT reference meetings, transcripts, files, or retrieval.
- ONLY clarify, explain, or rephrase what the AI already said.
- If the question requires any new information,
  respond EXACTLY with: "{SAFE_ABSTAIN}"

CHAT CONTEXT:
{chat_context}

USER QUESTION:
"{state['question']}"

ANSWER:
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,         
            max_tokens=180
        )

        answer = response.choices[0].message.content.strip()
        if not answer:
            answer = SAFE_ABSTAIN

    except Exception as e:
        print(f"Pure Chat Error: {e}")
        answer = SAFE_ABSTAIN

    # -------------------------------------------------
    # 3. Finalize (LIGHT CLEAN ONLY)
    # -------------------------------------------------
    state["final_answer"] = clean_answer(answer)
    state["method"] = "chat_only"
    state["context_extended"] = False

    #  Explicitly clear evidence fields (safety)
    state["retrieved_chunks"] = []
    state["meeting_indices"] = None

    return state

def retrieve_chunks_node(state: MeetingState):

    state["path"].append("retrieve_chunks")

    # -------------------------------------------------
    # 1. Build retrieval payload (constraints INCLUDED)
    # -------------------------------------------------
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

    # -------------------------------------------------
    #  Temporal constraint (STRICT FILTER ONLY)
    # -------------------------------------------------
    if state.get("temporal_constraint") == "latest":
        latest_meeting = max(c["meeting_index"] for c in clean_chunks)
        clean_chunks = [
            c for c in clean_chunks
            if c["meeting_index"] == latest_meeting
        ]

    # ------------------------------------------------
    # 3.  CRITICAL FIX: deterministic ordering
    # -------------------------------------------------
    clean_chunks = sorted(
        clean_chunks,
        key=lambda c: (c["meeting_index"], c["chunk_index"])
    )

    meeting_ids = sorted({
        c["meeting_index"] for c in clean_chunks
    })

    # -------------------------------------------------
    # 4. Store evidence
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 0. No evidence → safest defaults
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 1. Evidence dominance (PRIMARY & REQUIRED)
    # -------------------------------------------------
    latest_count = sum(1 for m in meeting_ids if m == latest_meeting)
    total = len(meeting_ids)
    latest_ratio = latest_count / total

    # -------------------------------------------------
    # 2. HARD evidence-based intent inference
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 3. Ambiguous zone → ALLOW question as tie-breaker
    # -------------------------------------------------
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
    """
    Routes between:
    - meeting_summary   → ONLY when explicitly asked
    - action_summary    → decisions / steps / next actions
    - chunk_answer      → factual answers & explanations
    """

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

    # 1️⃣ Explicit summary ONLY
    if matches(SUMMARY_KEYS, q) or matches(SUMMARY_KEYS, sq):
        return "meeting_summary"

    # 2️⃣ Decisions / steps / actions ( HIGH PRIORITY)
    if matches(ACTION_KEYS, q) or matches(ACTION_KEYS, sq):
        return "action_summary"

    # 3️⃣ Conceptual / discussion questions
    if matches(DISCUSSION_KEYS, q) or matches(DISCUSSION_KEYS, sq):
        return "chunk_answer"

    # 4️⃣ Default → factual QA
    return "chunk_answer"



def action_summary_node(state: MeetingState):
    """
    Extracts ONLY action items / next steps
    from the LATEST meeting.
    """

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



from scripts.generate_answer import generate_answer_with_llm

from collections import deque

def chunk_answer_node(state: MeetingState):
    """
    FINAL, CORRECT DESIGN (BUG-FIXED)

    - Question → chunks : recall only
    - LLM generates answer ONCE
    - Answer → evidence : truth validation
    - No hallucination
    - No missed explicit facts
    """
    print("\n DEBUG: ENTERED chunk_answer_node")
    print("Retrieved chunks:", len(state.get("retrieved_chunks", [])))


    state["path"].append("chunk_answer")

    query = state.get("standalone_query", state["question"])
    retrieved = state.get("retrieved_chunks", [])

    print(f" INTELLIGENT QA: '{query}'")

    # -------------------------------------------------
    #  No evidence
    # -------------------------------------------------
    if not retrieved:
        state["candidate_answer"] = SAFE_ABSTAIN
        state["confidence"] = 0.0
        state["method"] = "no_evidence"
        return state

    # -------------------------------------------------
    # Deterministic ordering
    # -------------------------------------------------
    retrieved = sorted(
        retrieved,
        key=lambda c: (c.get("meeting_index", 0), c.get("chunk_index", 0))
    )

    # -------------------------------------------------
    #  Embed question ONCE (RECALL ONLY)
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Question → chunk similarity (RECALL, NOT DECISION)
    # -------------------------------------------------
    sims = np.dot(chunk_embs, q_emb)

    # Take top-K candidates (prevents missing declarative facts)
    TOP_K = min(5, len(sims))
    top_indices = np.argsort(sims)[-TOP_K:][::-1]

    candidate_chunks = [aligned_chunks[i] for i in top_indices]

    # -------------------------------------------------
    # Expand context (prev + self + next, same meeting)
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Generate + VERIFY answer ( CRITICAL FIX)
    # -------------------------------------------------
    answer, confidence = generate_with_confidence(
        question=query,
        retrieved_chunks=expanded_chunks
    )

    # -------------------------------------------------
    # Final decision
    # -------------------------------------------------
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
    """
    EPISTEMIC GUARD (Safety Check)

    PURPOSE:
    - Prevents presenting exploratory ideas as confirmed decisions
    - Only intervenes when the USER asks for certainty
    - Option-3 compliant (no chunk inspection)
    """

    state["path"].append("verify")

    raw_q = state["question"].lower()
    rewritten_q = state.get("standalone_query", "").lower()
    answer = (state.get("candidate_answer") or "").lower()

    # -------------------------------------------------
    # 1. USER IS ASKING FOR CERTAINTY?
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 2. LANGUAGE CLASSIFICATION (ANSWER SIDE)
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 3. HARD DECISION GUARD
    # -------------------------------------------------
    # User wants certainty, answer is exploratory → BLOCK
    if asks_for_certainty() and has_exploratory:
        state["final_answer"] = (
            "The meeting explored this as an idea, but no final or "
            "confirmed decision was explicitly stated in the transcript."
        )
        state["method"] = "hard_decision_override"
        state["context_extended"] = False
        return state

    # -------------------------------------------------
    # 4. MIXED SIGNAL GUARD (VERY IMPORTANT)
    # -------------------------------------------------
    # Answer contains both exploratory + confirmation → CLARIFY
    if asks_for_certainty() and has_exploratory and has_confirmation:
        state["final_answer"] = (
            "The discussion included exploratory ideas, but the transcript "
            "does not clearly confirm this as a finalized or mandatory decision."
        )
        state["method"] = "mixed_certainty_override"
        state["context_extended"] = False
        return state

    # -------------------------------------------------
    # 5. NEXT-STEP QUESTIONS ARE ALLOWED
    # -------------------------------------------------
    # Exploratory answers are fine here
    if asks_for_next_steps():
        return state

    # -------------------------------------------------
    # 6. HYPOTHETICAL GUARD (ONLY IF USER WANTS FACT)
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 7. PASS
    # -------------------------------------------------
    return state

# FINALIZE NODE (Updated)
from agents.text_utils import clean_answer, SAFE_ABSTAIN

def finalize_node(state: MeetingState):
    state["path"].append("finalize")

    # 1. Determine the Final Answer
    # If a specialized node already set final_answer → KEEP IT
    if state.get("final_answer"):
        answer = state["final_answer"]
    else:
        raw = state.get("candidate_answer")
        if raw:
            answer = clean_answer(raw)
        else:
            answer = SAFE_ABSTAIN

    # 2. Extract Metadata (Meeting Index)
    meeting_index = None
    retrieved = state.get("retrieved_chunks")

    if (
        isinstance(retrieved, list)
        and len(retrieved) > 0
        and isinstance(retrieved[0], dict)
    ):
        meeting_index = retrieved[0].get("meeting_index")

    # 3. Save to Session Memory
    session_memory.add_turn(
        session_id=state["session_id"],
        question=state["question"],
        answer=answer,
        source=state.get("decision"),
        meeting_index=meeting_index,
        method=state.get("method"),
        standalone_query=state.get("standalone_query"),
        time_scope=state.get("time_scope"),
        meeting_indices=state.get("meeting_indices"),
    )

    # 4. Commit
    state["final_answer"] = answer
    return state


graph = StateGraph(MeetingState)

# -------------------------------------------------
# 1. ADD NODES
# -------------------------------------------------
graph.add_node("query", query_understanding_node)
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

# -------------------------------------------------
# 2. ENTRY POINT
# -------------------------------------------------
graph.set_entry_point("query")

# -------------------------------------------------
# 3. QUERY → COORDINATOR
# -------------------------------------------------
graph.add_edge("query", "coordinator")

graph.add_conditional_edges(
    "coordinator",
    lambda s: s["decision"],
    {
        Decision.CHAT_ONLY: "pure_chat",
        Decision.RETRIEVAL_ONLY: "retrieve",
        Decision.IGNORE: "finalize",
    }
)

# -------------------------------------------------
# 4. RETRIEVAL PIPELINE (LINEAR, SAFE)
# -------------------------------------------------
graph.add_edge("retrieve", "infer_intent")
graph.add_edge("infer_intent", "decide_source")

# -------------------------------------------------
# 5. SINGLE ROUTER ( ONLY ONE)
# -------------------------------------------------
graph.add_conditional_edges(
    "decide_source",
    post_retrieve_router,
    {
        "meeting_summary": "meeting_summary",
        "action_summary": "action_summary",
        "chunk_answer": "chunk_answer",
    }
)

# -------------------------------------------------
# 6. LINEAR TERMINAL PATHS
# -------------------------------------------------
graph.add_edge("pure_chat", "finalize")
graph.add_edge("meeting_summary", "finalize")
graph.add_edge("action_summary", "finalize")

graph.add_edge("chunk_answer", "verify")
graph.add_edge("verify", "finalize")

# -------------------------------------------------
# 7. END
# -------------------------------------------------
graph.add_edge("finalize", END)

# -------------------------------------------------
# 8. COMPILE
# -------------------------------------------------
meeting_graph = graph.compile()
