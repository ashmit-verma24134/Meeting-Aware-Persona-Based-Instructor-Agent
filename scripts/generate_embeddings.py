# Uses BAAI/bge-base-en-v1.5 (SentenceTransformers)
# One chunk -> one vector (with full metadata)

import json
from sentence_transformers import SentenceTransformer
from scripts.embedding_utils import build_embedding_text

CHUNKS_PATH = "data/chunks.json"
OUTPUT_PATH = "chunk_embeddings.json"
MODEL_NAME = "BAAI/bge-base-en-v1.5"


def main():
    print("Loading BGE embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")

    # Load transcript chunks
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    chunk_embeddings = []

    prev_chunk = None
    prev_meeting_key = None   # (meeting_name, meeting_index)

    for chunk in chunks:
        meeting_key = (chunk["meeting_name"], chunk["meeting_index"])

        # Prevent cross-meeting context bleed
        if meeting_key != prev_meeting_key:
            embedding_text = build_embedding_text(chunk, None)
        else:
            embedding_text = build_embedding_text(chunk, prev_chunk)

        # Generate embedding
        embedding_vector = model.encode(
            embedding_text,
            normalize_embeddings=True
        ).tolist()

        # Store embedding WITH FULL METADATA
        chunk_embeddings.append({
            "chunk_id": chunk["chunk_id"],
            "user_id": chunk["user_id"],
            "meeting_name": chunk["meeting_name"],
            "meeting_index": chunk["meeting_index"],
            "meeting_type": chunk["meeting_type"],
            "project_type": chunk.get("project_type"),
            "chunk_index": chunk["chunk_index"],
            "embedding": embedding_vector
        })

        prev_chunk = chunk
        prev_meeting_key = meeting_key

    # Save embeddings to disk
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_embeddings, f, indent=2)

    print(f"✅ Saved {len(chunk_embeddings)} embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
