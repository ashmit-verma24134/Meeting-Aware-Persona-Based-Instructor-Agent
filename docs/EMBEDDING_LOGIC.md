EMBEDDING INPUT LOGIC (FINAL – DO NOT CHANGE)

Stored Text:
- stored_text = chunk.text
- No overlap
- Used only for display and retrieval output

Embedding Text:
If the chunk is the first chunk of a meeting:
    embedding_text = chunk.text
Else:
    embedding_text = last_30_tokens(previous_chunk.text) + chunk.text

Rules:
- Overlap is NOT stored
- Overlap is ONLY used for embedding input
- One chunk produces exactly one embedding
- No chunk merging
- No summarization
- No artificial speaker injection

This logic is frozen after Day 3.
If transcripts change → regenerate embeddings.
If transcripts do not change → embeddings MUST NOT change.



Session memory stores only recent question–answer pairs to support follow-up queries. Transcript embeddings remain the sole authoritative knowledge source for answer generation.