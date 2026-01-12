'''it loads all vectors builds faiss index, saves it to disk -> for fast semantic search based on cosine similarities'''

import json
import numpy as np
import faiss

EMBEDDINGS_PATH = "chunk_embeddings.json"
INDEX_PATH = "vector.index"

def main():
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        data= json.load(f)

    embeddings= [item["embedding"] for item in data]        # Extract embeddings
    embeddings_np = np.array(embeddings, dtype="float32")   #converted to numpy arr

    num_vectors,dim= embeddings_np.shape     # Check dimensions
    print("Embedding dimension:", dim)
    print("Number of vectors:", num_vectors)
    
    index= faiss.IndexFlatIP(dim) #faiss data structure for fast optimization

    index.add(embeddings_np)                # Add vectors to index
    print("Total vectors indexed:", index.ntotal)
    
    # Save index to disk
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved to {INDEX_PATH}")

if __name__ == "__main__":
    main()
