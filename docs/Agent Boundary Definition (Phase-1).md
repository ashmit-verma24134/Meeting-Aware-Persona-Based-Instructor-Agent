AGENT BOUNDARIES — PHASE 1 (TRANSCRIPT-DRIVEN)

1. Query Understanding Agent
   - Input: user question (text)
   - Responsibility:
     • Understand what the user is asking
     • Decide whether the question refers to meeting content or is out-of-scope
   - Implementation:
     • Simple rule-based intent detection (Phase 1)
   - Status: Implemented (basic)

2. Transcript Retrieval Agent
   - Input: user_id, question
   - Responsibility:
     • Retrieve relevant transcript chunks
     • Enforce strict user isolation
     • Support multiple meetings per user
   - Implementation:
     • BGE embeddings + FAISS similarity search
   - Status: Implemented (final, frozen)

3. Answer Generation Agent
   - Input: user question + retrieved transcript chunks
   - Responsibility:
     • Generate a concise (1–3 sentence) answer
     • Use ONLY transcript excerpts
     • Avoid summarization and hallucination
   - Implementation:
     • LLM with a locked prompt
   - Status: Implemented (Phase-1 compliant)

4. Response Evaluation Agent
   - Input: generated answer
   - Responsibility:
     • Check answer length and relevance
     • Ensure no summary dump
     • Allow abstention if information is unclear
   - Implementation:
     • Simple rule-based validation (Phase 1)
   - Status: Implemented (basic)

5. Supervisor Agent
   - Input: user_id, question
   - Responsibility:
     • Orchestrate the overall flow
     • Call Query Understanding → Retrieval → Answer → Evaluation
     • Return final answer + evidence to the user
   - Implementation:
     • Control logic (no LLM)
   - Status: To be implemented (Day-5 Task-2)
