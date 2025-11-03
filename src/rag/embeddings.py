from typing import Iterable, List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        emb = self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return emb.astype("float32")



