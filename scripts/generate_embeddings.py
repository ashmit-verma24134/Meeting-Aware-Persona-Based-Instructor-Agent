
#Uses BAAI/bge-base-en-v1.5 (SentenceTransformers)
#One chunk -> one vector

import json                     #one chunk -> one vector 
from sentence_transformers import SentenceTransformer
from embedding_utils import build_embedding_text

CHUNKS_PATH = "data/chunks.json"
OUTPUT_PATH = "chunk_embeddings.json"
MODEL_NAME = "BAAI/bge-base-en-v1.5"      #embedding model


def main():
    print("Loading BGE embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")

    # loading transcript chunks
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    chunk_embeddings = []

    prev_chunk = None
    prev_meeting = None

    for chunk in chunks:
        if chunk["meeting_name"] != prev_meeting:  #cross meeting prevention
            embedding_text= build_embedding_text(chunk, None)
        
        else:     
            embedding_text= build_embedding_text(chunk, prev_chunk)

        # Generate embedding
        embedding_vector = model.encode(
            embedding_text,
            normalize_embeddings=True
        ).tolist()          #json serialized

        # Store embedding with metadata
        chunk_embeddings.append({
            "chunk_id": chunk["chunk_id"],
            "user_id": chunk["user_id"],
            "meeting_name": chunk["meeting_name"],
            "meeting_type": chunk["meeting_type"],
            "embedding": embedding_vector
        })

        
        prev_chunk = chunk                      # Update trackers
        prev_meeting = chunk["meeting_name"]

    # Save embeddings to disk
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_embeddings, f, indent=2)

    print(f"Saved {len(chunk_embeddings)} embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
