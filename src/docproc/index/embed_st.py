from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class STEmbedder:
    model_name: str
    batch_size: int = 64
    normalize_embeddings: bool = True
    device: str = "cpu"

    def __post_init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    def transform(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )