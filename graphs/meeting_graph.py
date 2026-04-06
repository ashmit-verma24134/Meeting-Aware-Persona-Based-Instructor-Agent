from typing import TypedDict, List, Optional
import json
from agents.text_utils import clean_answer
from agents.source_decider_agent import decide_source_node
from agents.text_utils import trim_chunk_text
from agents.decision_types import Decision
from services.supabase_service import SupabaseService
from langgraph.graph import StateGraph, END
from groq import Groq
from scripts.generate_answer import generate_answer_with_llm
SAFE_ABSTAIN = "This was not clearly discussed in the meeting."
from services.embedding_api import get_embedding

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

    # ── Keep ONLY transcript chunks for summary ──
    # Slack chunks contain Q&A history which pollutes the summary
    transcript_chunks = [
        c for c in retrieved
        if c.get("source", "transcript") == "transcript"
    ]

    # Fallback to all chunks if no transcript chunks found
    if not transcript_chunks:
        transcript_chunks = retrieved

    first_meeting_id = transcript_chunks[0].get("meeting_id")

    if not first_meeting_id:
        state["final_answer"] = SAFE_ABSTAIN
        state["method"] = "summary_no_meeting_id"
        state["context_extended"] = False
        return state

    meeting_chunks = [
        c for c in transcript_chunks
        if c.get("meeting_id") == first_meeting_id
        and isinstance(c.get("text"), str)
    ]

    if not meeting_chunks:
        state["final_answer"] = SAFE_ABSTAIN
        state["method"] = "summary_empty_meeting"
        state["context_extended"] = False
        return state

    meeting_chunks = sorted(
        meeting_chunks,
        key=lambda c: c.get("chunk_index", 0)
    )

    context = "\n\n".join(
        c["text"] for c in meeting_chunks
    )[:12000]

    prompt = f"""
You are a professional project assistant summarizing a meeting.

RULES:
- Use ONLY the provided transcript fragments.
- Do NOT assume decisions unless explicitly stated.
- If information is unclear, omit it.
- Do NOT include bare timestamps like [00:15:30] in your answer unless you also know the calendar date.
- Do NOT include Slack messages, run IDs, or chat history in the summary.

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

_SUPABASE_CLIENT = None




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

    supabase = get_supabase_client()

    # -----------------------------------
    # QUERY EXPANSION
    # -----------------------------------
    SYNONYMS = {
        "course": "course class subject lecture CSE",
        "bot": "bot agent MMA assistant",
        "screen": "screen display visible monitor showed",
        "meeting": "meeting call session discussion",
        "model": "model tool library framework",
        "extraction": "extraction parsing reading OCR",
        "recording": "recording video demo presentation",
        "pipeline": "pipeline process workflow system",
        "agenda": "agenda plan goals topics",
        "summary": "summary overview highlights recap",
        "chatbot": "chatbot bot assistant patient chat",
        "demo": "demo demonstration presentation showcase",
        "file": "file folder directory path",
        "run": "run execute start pipeline process",
    }

    expanded_query = query_text
    q_lower = query_text.lower()
    for keyword, expansion in SYNONYMS.items():
        if keyword in q_lower:
            expanded_query = expanded_query + " " + expansion

    print(f"[QUERY EXPAND] Original: '{query_text}' → Expanded: '{expanded_query[:80]}...'")

    try:
        query_embedding = get_embedding(expanded_query)
    except Exception:
        state["retrieved_chunks"] = []
        state["_all_meeting_indices"] = []
        state["meeting_indices"] = []
        return state

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
                "source": row.get("source", "transcript"),
            })

    if not clean_chunks:
        state["retrieved_chunks"] = []
        state["_all_meeting_indices"] = []
        state["meeting_indices"] = []
        return state

    # -----------------------------------
    # BM25 HYBRID SEARCH
    # Run keyword search in parallel with vector search
    # Vector catches semantic matches, BM25 catches exact keywords
    # Chunks found by both get a score boost
    # Chunks only found by BM25 get added to the pool
    # -----------------------------------
    try:
        bm25_results = supabase.match_chunks_bm25(
            query_text=query_text,
            user_id=state["user_id"],
            match_count=20
        )

        if bm25_results:
            existing_ids = {
                c["meeting_id"] + str(c["chunk_index"])
                for c in clean_chunks
            }

            for row in bm25_results:
                if (
                    isinstance(row.get("meeting_id"), str)
                    and isinstance(row.get("chunk_index"), int)
                    and isinstance(row.get("chunk_text"), str)
                ):
                    chunk_key = row["meeting_id"] + str(row["chunk_index"])

                    if chunk_key not in existing_ids:
                        # Found by BM25 but missed by vector — add it
                        clean_chunks.append({
                            "meeting_id": row["meeting_id"],
                            "chunk_index": row["chunk_index"],
                            "text": row["chunk_text"],
                            "similarity": float(row.get("bm25_rank", 0.1)),
                            "source": row.get("source", "transcript"),
                        })
                        existing_ids.add(chunk_key)
                    else:
                        # Found by both — boost its similarity score
                        for c in clean_chunks:
                            if (
                                c["meeting_id"] == row["meeting_id"]
                                and c["chunk_index"] == row["chunk_index"]
                            ):
                                c["similarity"] += float(row.get("bm25_rank", 0.0)) * 0.3
                                break

            print(f"[BM25] Hybrid merge done, total chunks: {len(clean_chunks)}")

    except Exception as e:
        print(f"[BM25] Hybrid search failed: {e}")

    # -----------------------------------
    # Sort by similarity FIRST
    # -----------------------------------
    clean_chunks = sorted(
        clean_chunks,
        key=lambda c: c["similarity"],
        reverse=True
    )

    import re as _re

    q_text = (state.get("question", "") + " " + state.get("standalone_query", "")).lower()

    # -----------------------------------
    # SOURCE-AWARE ROUTING
    # -----------------------------------
    SLACK_KEYS = {
        "slack", "channel", "discuss", "discussed", "chat",
        "said", "typed", "message", "conversation", "talked"
    }
    TRANSCRIPT_KEYS = {
        "screen", "meeting", "transcript", "visible", "recording",
        "speaker", "demo", "agenda", "summary", "presentation",
        "keyframe", "pipeline", "video"
    }

    q_lower_words = set(q_text.split())
    force_slack = any(k in q_lower_words for k in SLACK_KEYS)
    force_transcript = any(k in q_lower_words for k in TRANSCRIPT_KEYS)

    if force_slack and not force_transcript:
        slack_only = [c for c in clean_chunks if c.get("source") == "slack"]
        if slack_only:
            clean_chunks = slack_only
            print("[SOURCE ROUTE] Forced to slack only")

    elif force_transcript and not force_slack:
        transcript_only = [c for c in clean_chunks if c.get("source") == "transcript"]
        if transcript_only:
            clean_chunks = transcript_only
            print("[SOURCE ROUTE] Forced to transcript only")

    # -----------------------------------
    # RE-RANKING
    # -----------------------------------
    q_words = set(_re.findall(r'\b\w{3,}\b', q_text))

    for chunk in clean_chunks:
        chunk_words = set(_re.findall(r'\b\w{3,}\b', chunk.get("text", "").lower()))
        keyword_overlap = len(q_words & chunk_words)
        chunk["rerank_score"] = chunk["similarity"] + (0.05 * keyword_overlap)

    clean_chunks = sorted(
        clean_chunks,
        key=lambda c: c.get("rerank_score", c["similarity"]),
        reverse=True
    )
    print(f"[RERANK] Re-ranked {len(clean_chunks)} chunks by keyword overlap")

    # -----------------------------------
    # DATE + SLACK KEYWORD LOGIC
    # -----------------------------------
    detected_date = None

    m = _re.search(r'(\d{4})[-/](\d{2})[-/](\d{2})', q_text)
    if m:
        detected_date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    if not detected_date:
        m = _re.search(r'(\d{1,2})\s*[-/\s]\s*(\d{1,2})\s*[-/\s]\s*(\d{2,4})', q_text)
        if m:
            d, mo, y = m.group(1), m.group(2), m.group(3)
            if len(y) == 2:
                y = "20" + y
            detected_date = f"{y}-{mo.zfill(2)}-{d.zfill(2)}"

    if not detected_date:
        MONTHS = {
            "january": "01", "february": "02", "march": "03", "april": "04",
            "may": "05", "june": "06", "july": "07", "august": "08",
            "september": "09", "october": "10", "november": "11", "december": "12",
            "jan": "01", "feb": "02", "mar": "03", "apr": "04",
            "jun": "06", "jul": "07", "aug": "08", "sep": "09",
            "oct": "10", "nov": "11", "dec": "12"
        }
        for month_name, month_num in MONTHS.items():
            m = _re.search(
                rf'(\d{{1,2}})\s*(?:st|nd|rd|th)?\s*{month_name}|{month_name}\s*(\d{{1,2}})',
                q_text
            )
            if m:
                day = m.group(1) or m.group(2)
                detected_date = f"2026-{month_num}-{day.zfill(2)}"
                break

    if detected_date:
        print(f"[DATE FILTER] Detected date: {detected_date}")
        date_chunks = [
            c for c in clean_chunks
            if c.get("source") == "slack"
            and detected_date in c.get("text", "")
        ]
        if date_chunks:
            print(f"[DATE FILTER] Found {len(date_chunks)} slack chunks for {detected_date}")
            clean_chunks = date_chunks
        else:
            print(f"[DATE FILTER] No slack chunks for {detected_date}, keeping all")

    else:
        boosted = False

        for chunk in clean_chunks:
            if chunk.get("source") == "slack":
                chunk_words = set(_re.findall(r'\b\w{3,}\b', chunk.get("text", "").lower()))
                hits = len(q_words & chunk_words)
                if hits >= 2:
                    chunk["similarity"] += 0.15 * hits
                    boosted = True

        if boosted:
            clean_chunks = sorted(clean_chunks, key=lambda c: c["similarity"], reverse=True)

            for i, chunk in enumerate(clean_chunks):
                if chunk.get("source") == "slack":
                    lines = chunk.get("text", "").split("\n")
                    relevant = []
                    seen = set()
                    for j, line in enumerate(lines):
                        line_words = set(_re.findall(r'\b\w{3,}\b', line.lower()))
                        if len(q_words & line_words) >= 1:
                            start = max(0, j - 2)
                            end = min(len(lines), j + 4)
                            for k in range(start, end):
                                if k not in seen:
                                    relevant.append(lines[k])
                                    seen.add(k)
                    if relevant:
                        clean_chunks[i] = {**chunk, "text": "\n".join(relevant)}

            print(f"[SLACK BOOST] Applied keyword boost + windowing to slack chunks")

    # -----------------------------------
    # Temporal filtering (latest meeting)
    # -----------------------------------
    if state.get("temporal_constraint") == "latest":
        try:
            latest_meeting = supabase.get_latest_meeting_by_user(
                user_id=state["user_id"]
            )
            if latest_meeting:
                latest_id = latest_meeting["id"]
                clean_chunks = [
                    c for c in clean_chunks
                    if c["meeting_id"] == latest_id
                ]
        except Exception as e:
            print("Latest meeting fetch failed:", e)

    # -----------------------------------
    # Sort by transcript order
    # -----------------------------------
    clean_chunks = sorted(
        clean_chunks,
        key=lambda c: (c["meeting_id"], c["chunk_index"])
    )

    meeting_ids = []
    for c in clean_chunks:
        mid = c["meeting_id"]
        if mid not in meeting_ids:
            meeting_ids.append(mid)

    state["retrieved_chunks"] = clean_chunks
    state["_all_meeting_indices"] = meeting_ids
    state["meeting_indices"] = meeting_ids

    source_counts = {}
    for c in clean_chunks:
        src = c.get("source", "transcript")
        source_counts[src] = source_counts.get(src, 0) + 1
    print(f"[RETRIEVE] Sources: {source_counts}")

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
        "summary", "summarize","summarise" ,  "overview",
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
You are extracting ACTION ITEMS from the context.

RULES:
- Use primarily the transcript text
- List concrete next steps / tasks
- You may include brief context for clarity
- If no action items are found, say exactly:
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

def generate_with_confidence(
    question: str,
    retrieved_chunks: list,
):
   
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
        try:
            gen_prompt = f"""You are a helpful project assistant.
Answer the question concisely in 1-2 sentences using your general knowledge.
If you are unsure, say so.

Question: {query}

Answer:"""
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": gen_prompt}],
                temperature=0.0,
                max_tokens=150,
            )
            general_answer = response.choices[0].message.content.strip()
            if general_answer:
                state["candidate_answer"] = general_answer
                state["confidence"] = 0.5
                state["method"] = "general_knowledge"
                return state
        except Exception as e:
            print(f"General knowledge fallback failed: {e}")

        state["candidate_answer"] = SAFE_ABSTAIN
        state["confidence"] = 0.0
        state["method"] = "no_evidence"
        return state

    # -----------------------------------
    # CONFIDENCE THRESHOLD TUNING
    # Different question types need different thresholds
    # Factual (who/what/when) → stricter → less hallucination
    # General (how/why/explain) → looser → more answers
    # Date questions → very loose → date filter already did the work
    # -----------------------------------
    q_lower = query.lower()

    FACTUAL_KEYS = {"who", "what", "when", "where", "which"}
    GENERAL_KEYS = {"how", "why", "explain", "describe", "tell"}
    DATE_KEYS = {"march", "january", "february", "april", "may", "june",
                 "july", "august", "september", "october", "november", "december",
                 "2026", "2025", "date", "day"}

    q_words_set = set(q_lower.split())

    if q_words_set & DATE_KEYS:
        sim_threshold = 0.10
        print("[THRESHOLD] Date question → 0.10")
    elif q_words_set & FACTUAL_KEYS:
        sim_threshold = 0.25
        print("[THRESHOLD] Factual question → 0.25")
    elif q_words_set & GENERAL_KEYS:
        sim_threshold = 0.18
        print("[THRESHOLD] General question → 0.18")
    else:
        sim_threshold = 0.20
        print("[THRESHOLD] Default → 0.20")

    # Filter chunks below threshold
    filtered = [c for c in retrieved if c.get("similarity", 0.0) >= sim_threshold]
    if not filtered:
        # Fallback: keep top 1 regardless of threshold
        filtered = retrieved[:1]
        print("[THRESHOLD] All below threshold, keeping top 1")

    # -----------------------------------
    # TOP-K — max 3 to avoid cross-chunk hallucination
    # -----------------------------------
    sorted_chunks = sorted(
        filtered,
        key=lambda c: c.get("rerank_score", c.get("similarity", 0.0)),
        reverse=True
    )

    q_words_set = set(q_lower.split())
    MAX_CONTEXT_CHUNKS = 8 if q_words_set & DATE_KEYS else 3
    selected_chunks = sorted_chunks[:MAX_CONTEXT_CHUNKS]

    print(f"\nChunks passed to LLM: {len(selected_chunks)}")

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

    # User wants certainty, answer is exploratory → ADD DISCLAIMER instead of blocking
    if asks_for_certainty() and has_exploratory:
        state["candidate_answer"] = (
            state.get("candidate_answer", "") + 
            "\n\nNote: The meeting explored this as an idea, but no final "
            "or confirmed decision was explicitly stated in the transcript."
        )
        state["method"] = "answer_with_certainty_disclaimer"
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
    - Chat history is summarized before sufficiency check (fewer tokens, better context)
    """

    state.setdefault("path", [])
    state["path"].append("chat_answer")

    supabase = get_supabase_client()

    print("\nCHAT NODE STARTS ")

    # --------------------------------------------------
    # FORCE RETRIEVAL for summary/overview questions
    # These must NEVER be answered from chat memory
    # --------------------------------------------------
    FORCE_RETRIEVAL_KEYS = [
        "summarize", "summary", "overview", "highlights", "takeaways",
        "what happened", "what was discussed", "what did we discuss",
        "summarise", "give me a summary", "latest meeting",
    ]
    q_lower = state["question"].lower()
    if any(k in q_lower for k in FORCE_RETRIEVAL_KEYS):
        print("Force retrieval — summary/overview question")
        state["decision"] = Decision.RETRIEVAL_ONLY
        state["method"] = "force_retrieval_summary"
        return state

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

    chat_context_raw = "\n\n".join(chat_context_parts).strip()

    if not chat_context_raw:
        print(" Empty chat context → Retrieval")
        state["decision"] = Decision.RETRIEVAL_ONLY
        state["method"] = "chat_empty_context"
        return state

    # --------------------------------------------------
    # CHAT MEMORY SUMMARIZATION
    # If history is long (>10 turns), compress it into a
    # concise summary before passing to sufficiency check.
    # Benefits:
    # - Fewer tokens sent to LLM
    # - Removes old irrelevant turns
    # - Keeps only key facts from the conversation
    # --------------------------------------------------
    if len(chat_lines) > 10:
        try:
            summarize_prompt = f"""
You are summarizing a conversation between a User and an AI assistant.

TASK:
- Extract ONLY the key facts and answers that were established in this conversation.
- Write as a compact bullet list (max 8 bullets).
- Each bullet = one fact/answer established.
- Do NOT include greetings, failed answers, or "I couldn't find" responses.
- Keep it under 200 words total.

CONVERSATION:
{chat_context_raw}

KEY FACTS ESTABLISHED:
""".strip()

            summary_resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": summarize_prompt}],
                temperature=0.0,
                max_tokens=250,
            )

            chat_summary = summary_resp.choices[0].message.content.strip()
            print(f"[CHAT SUMMARY] Compressed {len(chat_lines)} turns → summary")
            print(f"[CHAT SUMMARY] {chat_summary[:100]}...")

            # Use summary as the context for sufficiency check
            chat_context = chat_summary
            assistant_only = chat_summary  # summary already contains AI facts

        except Exception as e:
            print(f"[CHAT SUMMARY] Failed: {e} — using raw context")
            chat_context = chat_context_raw
            assistant_only = "\n\n".join(
                part for part in chat_context_parts
                if part.startswith("AI:")
            )
    else:
        # Short history — use raw context directly
        chat_context = chat_context_raw
        assistant_only = "\n\n".join(
            part for part in chat_context_parts
            if part.startswith("AI:")
        )

    print("\n...CHAT CONTEXT SENT TO SUFFICIENCY... ")
    print(chat_context[:300])
    print("\n..........................\n")

    # --------------------------------------------------
    # Semantic Sufficiency Check
    # --------------------------------------------------
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


def llm_verify_node(state: MeetingState):

    state.setdefault("path", [])
    state["path"].append("llm_verify")

    candidate = state.get("candidate_answer", "")
    method = state.get("method", "")

    if method == "general_knowledge":
        return state

    if not candidate or candidate.strip() == SAFE_ABSTAIN:
        return state

    retrieved = state.get("retrieved_chunks", [])
    if not retrieved:
        return state

    # Skip verify ONLY for date/activity questions
    # Date filter already narrowed chunks upstream, verify would incorrectly kill valid answers
    import re as _re
    q_lower = state.get("question", "").lower()
    DATE_SIGNALS = {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "2025", "2026", "what happened", "what was done", "what did"
    }
    if any(sig in q_lower for sig in DATE_SIGNALS):
        print("[LLM VERIFY] Skipping — date/activity question")
        return state

    context = "\n\n".join(
        c.get("text", "") for c in retrieved[:3]
        if isinstance(c.get("text"), str)
    )

    verify_prompt = f"""
Does the ANSWER directly come from or is clearly supported by the CONTEXT?
Reply YES if supported. Reply NO if it contains information NOT in the context.
Reply with only YES or NO.

CONTEXT:
{context}

ANSWER:
{candidate}

QUESTION:
{state.get("question", "")}
""".strip()

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": verify_prompt}],
            temperature=0.0,
            max_tokens=3,
        )
        raw = resp.choices[0].message.content.strip().upper()
        verdict = raw.split()[0] if raw else "NO"
        print(f"[LLM VERIFY] Verdict: {verdict}")
        if verdict == "NO":
            state["candidate_answer"] = SAFE_ABSTAIN
            state["method"] = "llm_verify_failed"
    except Exception as e:
        print(f"[LLM VERIFY] Error: {e}")

    return state



graph = StateGraph(MeetingState)

graph.add_node("query", query_understanding_node)
graph.add_node("chat_answer", chat_answer_node)
graph.add_node("retrieve", retrieve_chunks_node)
graph.add_node("infer_intent", infer_intent_node)
graph.add_node("decide_source", decide_source_node)
graph.add_node("chunk_answer", chunk_answer_node)
graph.add_node("meeting_summary", meeting_summary_node)
graph.add_node("action_summary", action_summary_node)
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
graph.add_edge("chunk_answer", "finalize")  # verify removed

graph.add_edge("finalize", END)

meeting_graph = graph.compile()