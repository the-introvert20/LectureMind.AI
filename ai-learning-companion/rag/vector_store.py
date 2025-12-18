import os
import json
from typing import List, Tuple

import numpy as np
import faiss


class LectureVectorStore:
    """
    Simple FAISS vector store per lecture.
    Stores:
      - FAISS index file
      - texts.json mapping id -> text
    """

    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.texts_path = os.path.join(index_dir, "texts.json")
        self.index = None
        self.texts: List[str] = []

    def exists(self) -> bool:
        return os.path.exists(self.index_path) and os.path.exists(self.texts_path)

    def build(self, texts: List[str], embeddings: np.ndarray):
        os.makedirs(self.index_dir, exist_ok=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity using normalized embeddings
        self.index.add(embeddings)
        self.texts = texts

    def save(self):
        if self.index is None or not self.texts:
            raise ValueError("Index not built.")
        faiss.write_index(self.index, self.index_path)
        with open(self.texts_path, "w", encoding="utf-8") as f:
            json.dump(self.texts, f, ensure_ascii=False, indent=2)

    def load(self):
        if not self.exists():
            raise FileNotFoundError("Index or texts not found.")
        self.index = faiss.read_index(self.index_path)
        with open(self.texts_path, "r", encoding="utf-8") as f:
            self.texts = json.load(f)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None:
            raise ValueError("Index not loaded.")
        query_embedding = query_embedding.reshape(1, -1)
        scores, idxs = self.index.search(query_embedding, top_k)
        results = []
        for i, sc in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            text = self.texts[i]
            results.append((text, float(sc)))
        return results