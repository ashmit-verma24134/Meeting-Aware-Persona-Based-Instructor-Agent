import json   # ✅ REQUIRED

with open("data/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open("data/chunk_embeddings.json", "r", encoding="utf-8") as f:
    embeds = json.load(f)

chunk_ids = {c["chunk_id"] for c in chunks}
embed_ids = {e["chunk_id"] for e in embeds}

missing = chunk_ids - embed_ids

print("❌ Missing embeddings:", len(missing))
print("Sample missing chunk_ids:", list(missing)[:10])
