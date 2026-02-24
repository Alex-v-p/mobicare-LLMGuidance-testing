from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    n: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr: float
    avg_first_rank: float


def compute_metrics(first_ranks: list[int | None]) -> RetrievalMetrics:
    n = len(first_ranks)
    hits1 = 0
    hits3 = 0
    hits5 = 0
    rr_sum = 0.0
    ranks_present: list[int] = []

    for r in first_ranks:
        if r is None:
            continue
        ranks_present.append(r)
        if r <= 1:
            hits1 += 1
        if r <= 3:
            hits3 += 1
        if r <= 5:
            hits5 += 1
        rr_sum += 1.0 / r

    avg_rank = sum(ranks_present) / len(ranks_present) if ranks_present else float("inf")
    return RetrievalMetrics(
        n=n,
        hit_at_1=hits1 / n if n else 0.0,
        hit_at_3=hits3 / n if n else 0.0,
        hit_at_5=hits5 / n if n else 0.0,
        mrr=rr_sum / n if n else 0.0,
        avg_first_rank=avg_rank,
    )
