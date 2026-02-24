from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalResult:
    idx: int
    score: float


def topk_cosine(query_vec: np.ndarray, doc_matrix: np.ndarray, k: int = 5) -> list[RetrievalResult]:
    """Return top-k cosine similarity indices.

    Works with sparse matrices too (scikit handles it).
    """
    sims = cosine_similarity(query_vec, doc_matrix)[0]
    if k >= len(sims):
        top_idx = np.argsort(-sims)
    else:
        top_idx = np.argpartition(-sims, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
    return [RetrievalResult(idx=int(i), score=float(sims[i])) for i in top_idx]
