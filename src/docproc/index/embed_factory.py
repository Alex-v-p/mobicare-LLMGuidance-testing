from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .embed import TfidfEmbedder


@dataclass
class STEmbedder:
    model_name: str
    batch_size: int = 64
    normalize_embeddings: bool = True
    device: str = "cpu"

    def __post_init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def build_embedder(cfg: dict):
    cfg = cfg or {}
    typ = str(cfg.get("type", "tfidf")).lower()

    if typ == "tfidf":
        return TfidfEmbedder()

    if typ in ("sentence_transformers", "st", "sbert"):
        model_name = cfg.get("model_name", "BAAI/bge-small-en-v1.5")
        batch_size = int(cfg.get("batch_size", 64))
        normalize = bool(cfg.get("normalize_embeddings", True))
        device = cfg.get("device", "auto")
        if device == "auto":
            # let sentence-transformers decide; "cuda" if available, else cpu
            device = "cuda" if _has_cuda() else "cpu"

        return STEmbedder(
            model_name=str(model_name),
            batch_size=batch_size,
            normalize_embeddings=normalize,
            device=str(device),
        )

    raise ValueError(f"Unknown embedder type: {typ}")


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False