
#This script tests semantic retrieval by safely finding the most relevant transcript chunks for a single user’s question from faiss data structure.
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
CHUNKS_PATH = "data/chunks.json"
EMBEDDINGS_PATH = "chunk_embeddings.json"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
TOP_K = 3

def main():
    # Load data
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        embeddings_data = json.load(f)

    model = SentenceTransformer(MODEL_NAME)
    user_id = "ashmit"
    question = "While presenting to agent what he asked about langgraph?"
    print("USER:", user_id)    
    print("QUESTION:", question)
    print("=" * 60)

    user_embeddings = []
    user_chunk_refs = []   # store correct chunk references

    for emb in embeddings_data:
        if emb["user_id"] != user_id:
            continue
        # find the EXACT matching chunk
        matching_chunk = next(
            c for c in chunks
            if c["chunk_id"] == emb["chunk_id"]
            and c["user_id"] == emb["user_id"]
            and c["meeting_name"] == emb["meeting_name"]
        )

        user_embeddings.append(emb["embedding"])
        user_chunk_refs.append(matching_chunk)

    if not user_embeddings:
        print("No data found for this user.")
        return

    user_embeddings_np = np.array(user_embeddings, dtype="float32")
    dim = user_embeddings_np.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(user_embeddings_np)

    query_embedding = model.encode(
        "query: " + question,
        normalize_embeddings=True
    )
    query_embedding = np.array([query_embedding], dtype="float32")

    scores, indices = index.search(query_embedding, TOP_K)

    for idx in indices[0]:
        chunk = user_chunk_refs[idx]
        print("Chunk ID:", chunk["chunk_id"])
        print("Meeting:", chunk["meeting_name"])
        print("Text preview:")
        print(chunk["text"].splitlines()[:2])
        print("-" * 60)


if __name__ == "__main__":
    main()
