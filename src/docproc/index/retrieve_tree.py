from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .retrieve import topk_cosine, RetrievalResult


@dataclass
class TreeRetrievalResult:
    idx: int
    score: float
    parent_id: str


def topk_tree_parent_child(
    query_vec: np.ndarray,
    parent_matrix: np.ndarray,
    child_matrix: np.ndarray,
    child_parent_ids: list[str],
    parent_top_k: int = 3,
    k: int = 5,
) -> list[RetrievalResult]:
    """
    1) retrieve top parent sections
    2) only consider children from those parents
    3) retrieve top-k children globally
    """
    top_parents = topk_cosine(query_vec, parent_matrix, k=parent_top_k)
    allowed = {str(pid.idx) for pid in top_parents}  # indices as strings not helpful; we need parent IDs

    # NOTE: we can't infer parent IDs from indices alone here, so we will pass parent_ids via eval runner
    # This function assumes eval runner builds a boolean mask of allowed children.

    raise NotImplementedError("Use runner version that supplies allowed child mask.")