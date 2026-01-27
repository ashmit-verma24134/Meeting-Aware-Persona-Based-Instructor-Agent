import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple


class ChatVectorStore:
    """
    Incremental FAISS-based vector store for CHAT memory.

    DESIGN:
    - Embeds each chat turn ONCE
    - FAISS index persists in-memory
    - Old chat NEVER re-embedded
    - Scales to long conversations
    """

    _MODEL = None  # shared embedding model

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        if ChatVectorStore._MODEL is None:
            ChatVectorStore._MODEL = SentenceTransformer(model_name)

        self.model = ChatVectorStore._MODEL
        self.index = None
        self.texts: List[str] = []
        self.dim: int | None = None


    def add_texts(self, texts: List[str]) -> None:
        if not texts:
            return

        new_texts = [t for t in texts if t not in self.texts]
        if not new_texts:
            return

        embeddings = self.model.encode(
            new_texts,
            normalize_embeddings=True
        )
        embeddings = np.asarray(embeddings, dtype="float32")

        if self.index is None:
            self.dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dim)

        self.index.add(embeddings)
        self.texts.extend(new_texts)

    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0


    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        if self.is_empty():
            return []

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )
        query_embedding = np.asarray(query_embedding, dtype="float32")

        scores, indices = self.index.search(query_embedding, k)

        results: List[Tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.texts):
                results.append((self.texts[idx], float(score)))

        return results


#  GLOBAL SINGLETON (IMPORTANT)
chat_vector_store = ChatVectorStore()
