import json
import os
from typing import Dict, List

import faiss
import numpy as np


class FaissIndex:
    def __init__(self, index_dir: str, faiss_index_filename: str, metadata_filename: str):
        self.index_dir = index_dir
        self.faiss_index_path = os.path.join(index_dir, faiss_index_filename)
        self.metadata_path = os.path.join(index_dir, metadata_filename)
        self.index = None
        self.chunks: List[Dict] = []

    def build(self, vectors: np.ndarray, chunks: List[Dict]):
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D array")
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self.chunks = chunks

    def save(self):
        os.makedirs(self.index_dir, exist_ok=True)
        if self.index is None:
            raise RuntimeError("Index not built")
        faiss.write_index(self.index, self.faiss_index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

    def load(self):
        if not (os.path.exists(self.faiss_index_path) and os.path.exists(self.metadata_path)):
            raise FileNotFoundError("Index files not found. Build first.")
        self.index = faiss.read_index(self.faiss_index_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Dict]:
        if self.index is None:
            raise RuntimeError("Index not loaded")
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        scores, idxs = self.index.search(query_vec.astype("float32"), top_k)
        results: List[Dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            c = self.chunks[idx]
            results.append({**c, "score": float(score)})
        return results



