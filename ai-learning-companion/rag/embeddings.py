import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper around SentenceTransformers for embedding texts and queries."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        return emb.astype(np.float32)