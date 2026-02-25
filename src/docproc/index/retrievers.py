from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

import numpy as np

from .retrieve import RetrievalResult, topk_cosine


def _as_2d(x: np.ndarray) -> np.ndarray:
    if hasattr(x, "ndim") and x.ndim == 1:
        return x.reshape(1, -1)
    return x


def _is_numpy_matrix(x) -> bool:
    return type(x).__module__.startswith("numpy") and hasattr(x, "shape")


def topk_dense_dot(query_vec, doc_mat, k: int) -> list[RetrievalResult]:
    """
    Dense top-k using dot product.
    If embeddings are L2-normalized, dot == cosine.
    """
    q = _as_2d(query_vec)
    scores = (doc_mat @ q.T).reshape(-1)
    k = min(k, len(scores))
    idx = np.argsort(-scores)[:k]
    return [RetrievalResult(idx=int(i), score=float(scores[int(i)])) for i in idx]


def topk_dense_cosine(query_vec, doc_mat, k: int) -> list[RetrievalResult]:
    return topk_cosine(_as_2d(query_vec), doc_mat, k=k)


def pick_dense_retriever(matrix):
    return topk_dense_dot if _is_numpy_matrix(matrix) else topk_dense_cosine


def minmax_norm(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    mn = float(scores.min())
    mx = float(scores.max())
    if mx - mn < 1e-9:
        return np.ones_like(scores, dtype=float)
    return (scores - mn) / (mx - mn)


@dataclass
class Candidate:
    idx: int
    score: float


class CrossEncoderLike(Protocol):
    def predict(self, sentences: Sequence[tuple[str, str]], batch_size: int = 32, show_progress_bar: bool = False):
        ...


class BM25Index:
    """
    Lightweight BM25 using rank-bm25.
    Stores tokenized documents and exposes get_scores(query_tokens).
    """
    def __init__(self, tokenized_corpus: list[list[str]]):
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(tokenized_corpus)

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        return np.array(self._bm25.get_scores(query_tokens), dtype=float)


def merge_hybrid(
    dense: list[RetrievalResult],
    bm25_scores: np.ndarray,
    dense_weight: float = 0.5,
    bm25_weight: float = 0.5,
    k: int = 5,
) -> list[RetrievalResult]:
    """
    Combine dense results with bm25 scores over full corpus.
    We min-max normalize:
      - dense candidate scores within candidates
      - bm25 over full corpus
    Then compute weighted sum and return top-k globally over candidate set union.
    """
    # Candidate set = dense top-N union bm25 top-N
    N = max(50, k * 10)
    dense_idx = [r.idx for r in dense[:min(N, len(dense))]]

    bm25_top = np.argsort(-bm25_scores)[: min(N, len(bm25_scores))]
    cand = set(dense_idx) | set(int(i) for i in bm25_top)

    if not cand:
        return []

    cand = sorted(cand)
    # Dense scores for candidates (missing => 0)
    dense_map = {r.idx: r.score for r in dense}
    dense_vec = np.array([dense_map.get(i, 0.0) for i in cand], dtype=float)
    bm25_vec = np.array([bm25_scores[i] for i in cand], dtype=float)

    dense_vec = minmax_norm(dense_vec)
    bm25_vec = minmax_norm(bm25_vec)

    final = dense_weight * dense_vec + bm25_weight * bm25_vec
    top = np.argsort(-final)[: min(k, len(final))]
    return [RetrievalResult(idx=int(cand[i]), score=float(final[i])) for i in top]


def rerank_cross_encoder(
    query: str,
    candidates: list[RetrievalResult],
    doc_texts: list[str],
    cross_encoder: CrossEncoderLike,
    k: int = 5,
    batch_size: int = 32,
) -> list[RetrievalResult]:
    if not candidates:
        return []

    pairs = [(query, doc_texts[r.idx]) for r in candidates]
    scores = cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    scores = np.array(scores, dtype=float)

    order = np.argsort(-scores)[: min(k, len(scores))]
    return [RetrievalResult(idx=int(candidates[i].idx), score=float(scores[i])) for i in order]
