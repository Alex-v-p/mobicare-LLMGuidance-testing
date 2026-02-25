from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CrossEncoderReranker:
    model_name: str = "BAAI/bge-reranker-base"
    device: str = "auto"
    batch_size: int = 32

    def __post_init__(self):
        from sentence_transformers import CrossEncoder
        dev = self.device
        if dev == "auto":
            dev = "cuda" if _has_cuda() else "cpu"
        self.model = CrossEncoder(self.model_name, device=dev)

    def predict(self, pairs, batch_size: int | None = None, show_progress_bar: bool = False):
        bs = batch_size or self.batch_size
        return self.model.predict(pairs, batch_size=bs, show_progress_bar=show_progress_bar)


def build_reranker(cfg: dict | None):
    cfg = cfg or {}
    typ = str(cfg.get("type", "none")).lower()
    if typ in ("none", "", "null"):
        return None
    if typ in ("cross_encoder", "cross-encoder", "reranker"):
        return CrossEncoderReranker(
            model_name=str(cfg.get("model_name", "BAAI/bge-reranker-base")),
            device=str(cfg.get("device", "auto")),
            batch_size=int(cfg.get("batch_size", 32)),
        )
    raise ValueError(f"Unknown reranker type: {typ}")


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
