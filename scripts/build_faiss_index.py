import json
import numpy as np
import faiss

EMBEDDINGS_PATH = "chunk_embeddings.json"
INDEX_PATH = "vector.index"

def main():
    # Load embeddings + metadata
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract vectors
    embeddings = [item["embedding"] for item in data]
    embeddings_np = np.array(embeddings, dtype="float32")

    # SAFETY: normalize again for cosine similarity
    faiss.normalize_L2(embeddings_np)

    # Extract REAL chunk_ids
    ids = np.array(
        [item["chunk_id"] for item in data],
        dtype="int64"
    )

    num_vectors, dim = embeddings_np.shape
    print("Embedding dimension:", dim)
    print("Number of vectors:", num_vectors)

    # FAISS index with ID mapping
    base_index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(base_index)

    # Add vectors WITH IDs
    index.add_with_ids(embeddings_np, ids)
    print("Total vectors indexed:", index.ntotal)

    # Save index
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved to {INDEX_PATH}")

if __name__ == "__main__":
    main()
