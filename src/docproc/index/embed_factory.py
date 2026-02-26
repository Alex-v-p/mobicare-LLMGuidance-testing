from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embed import TfidfEmbedder
from sentence_transformers import SentenceTransformer


@dataclass
class STEmbedder:
    model_name: str
    batch_size: int = 64
    normalize_embeddings: bool = True
    device: str = "cpu"
    trust_remote_code: bool = False

    model: SentenceTransformer = None

    def __post_init__(self):
        # Newer SentenceTransformers versions support trust_remote_code directly.
        # Older ones require forwarding via model_kwargs.
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
            )
        except TypeError:
            # Fallback for older versions
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                model_kwargs={"trust_remote_code": self.trust_remote_code},
            )

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
        model_name = cfg.get("model") or cfg.get("model_name") or "BAAI/bge-small-en-v1.5"

        batch_size = int(cfg.get("batch_size", 64))
        normalize = bool(cfg.get("normalize_embeddings", True))

        trust_remote_code = bool(cfg.get("trust_remote_code", False))

        device = cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if _has_cuda() else "cpu"

        return STEmbedder(
            model_name=str(model_name),
            batch_size=batch_size,
            normalize_embeddings=normalize,
            device=str(device),
            trust_remote_code=trust_remote_code,  
        )

    raise ValueError(f"Unknown embedder type: {typ}")


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False