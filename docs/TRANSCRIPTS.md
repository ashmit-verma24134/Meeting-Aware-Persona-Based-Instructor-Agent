chunks.json is the Authoritative Transcript Store.

Rule:
- If chunks.json changes → embeddings MUST be regenerated
- If chunks.json does not change → embeddings MUST NOT be regenerated
