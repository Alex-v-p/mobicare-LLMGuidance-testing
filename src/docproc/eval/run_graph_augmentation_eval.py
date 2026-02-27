from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics.pairwise import cosine_similarity

from ..chunkers.base import Page
from ..clean_profiles import apply_cleaning, is_noise_chunk, profile_from_cfg
from ..index.embed_factory import build_embedder
from ..index.retrieve import RetrievalResult, topk_cosine
from .datasets import load_retrieval_questions
from .metrics import compute_metrics


def load_pages_jsonl(path: Path) -> list[Page]:
    pages: list[Page] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pages.append(Page(page=int(obj["page"]), text=str(obj["text"])))
    return pages

def _find_repo_root(start: Path) -> Path:
    """
    Walk upwards to find the repo root. We look for common repo markers.
    Falls back to the start's 3-up parent if nothing is found.
    """
    p = start.resolve()
    for _ in range(10):
        if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / "src").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    # fallback for your layout: .../src/docproc/eval -> repo
    return start.resolve().parents[3]


def _short_path(path: Path, repo_root: Path) -> str:
    """Return repo-relative path if possible; otherwise a readable tail path."""
    try:
        return str(path.resolve().relative_to(repo_root))
    except Exception:
        parts = path.parts
        return str(Path(*parts[-3:])) if len(parts) >= 3 else str(path)


def build_chunker(cfg: dict):
    # import locally to avoid extra imports in other scripts
    from ..chunkers.naive_tokens import NaiveTokenChunker
    from ..chunkers.page_index import PageIndexChunker
    from ..chunkers.structured import StructuredHeadingChunker
    from ..chunkers.tree_parent_child import TreeParentChildChunker

    strategy = cfg.get("strategy")
    if strategy == "page_index":
        return PageIndexChunker()
    if strategy == "naive_tokens":
        return NaiveTokenChunker(chunk_size=int(cfg["chunk_size"]), overlap=int(cfg["overlap"]))
    if strategy == "structured_headings":
        return StructuredHeadingChunker(
            max_tokens=int(cfg.get("max_tokens", 800)),
            min_tokens=int(cfg.get("min_tokens", 120)),
        )
    if strategy == "tree_parent_child":
        return TreeParentChildChunker(
            parent_max_tokens=int(cfg.get("parent_max_tokens", 900)),
            child_chunk_size=int(cfg.get("child_chunk_size", 450)),
            child_overlap=int(cfg.get("child_overlap", 100)),
        )
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def _is_numpy_matrix(x: Any) -> bool:
    return type(x).__module__.startswith("numpy") and hasattr(x, "shape")


def build_knn_graph(doc_matrix: np.ndarray, graph_k: int) -> list[list[int]]:
    """Build a directed kNN graph over chunks based on cosine similarity."""
    # cosine_similarity returns dense; fine for our doc sizes (guidelines doc)
    sims = cosine_similarity(doc_matrix, doc_matrix)
    np.fill_diagonal(sims, -1.0)  # exclude self
    n = sims.shape[0]
    graph: list[list[int]] = []
    for i in range(n):
        # take top graph_k neighbors
        if graph_k >= n:
            nbrs = np.argsort(-sims[i]).tolist()
        else:
            idx = np.argpartition(-sims[i], kth=graph_k - 1)[:graph_k]
            idx = idx[np.argsort(-sims[i][idx])]
            nbrs = idx.tolist()
        graph.append([int(j) for j in nbrs])
    return graph


def expand_candidates_with_dists(
    seed_idx: list[int],
    graph: list[list[int]],
    hops: int,
    expand_per_node: int,
) -> dict[int, int]:
    """
    BFS expansion from seed indices with caps to avoid runaway.

    Returns a dict of {idx: min_hop_distance_from_any_seed}.
    Seed nodes have distance 0.
    """
    dists: dict[int, int] = {int(i): 0 for i in seed_idx}
    frontier: deque[int] = deque(seed_idx)

    for hop in range(1, int(hops) + 1):
        if not frontier:
            break
        next_frontier: deque[int] = deque()
        while frontier:
            i = int(frontier.popleft())
            nbrs = graph[i]
            if expand_per_node and expand_per_node > 0:
                nbrs = nbrs[: int(expand_per_node)]
            for j in nbrs:
                j = int(j)
                if j not in dists:
                    dists[j] = hop
                    next_frontier.append(j)
        frontier = next_frontier

    return dists


def rerank_with_graph(
    q_vec: np.ndarray,
    doc_matrix: np.ndarray,
    base_results: list[RetrievalResult],
    cand_dists: dict[int, int],
    beta: float,
) -> list[RetrievalResult]:
    """
    Rerank candidates using a mixture of:
      - query similarity (q_sims)
      - proximity to the baseline seeds (prox, computed in embedding space)
      - a small "non-seed hop bonus" (encourages nearby neighbors to surface)

    IMPORTANT: we prevent seed chunks from dominating proximity by excluding self-matches.
    """
    if not cand_dists:
        return []

    candidate_idx = list(cand_dists.keys())

    # query similarities for all candidates
    cand_matrix = doc_matrix[candidate_idx]
    q_sims = cosine_similarity(q_vec, cand_matrix)[0]  # shape (C,)

    # base (seed) indices/matrix
    base_idx = [int(r.idx) for r in base_results]
    base_set = set(base_idx)
    base_mat = doc_matrix[base_idx]

    # proximity: similarity to the best base chunk (embedding-to-embedding),
    # but for seed chunks we exclude the self-match that would otherwise be 1.0.
    if base_mat.size:
        bb = cosine_similarity(cand_matrix, base_mat)  # (C, B)
        # Exclude self-match for seed candidates
        for ci, doc_i in enumerate(candidate_idx):
            if doc_i in base_set:
                # find its position(s) in base_idx and zero them out
                for bj, bdoc in enumerate(base_idx):
                    if bdoc == doc_i:
                        bb[ci, bj] = -1.0
        prox = bb.max(axis=1)
        prox = np.maximum(prox, 0.0)  # clip negatives to 0
    else:
        prox = np.zeros_like(q_sims)

    # hop bonus (ONLY for non-seeds): closer neighbors get more bonus
    d = np.array([int(cand_dists[i]) for i in candidate_idx], dtype=np.float32)
    hop_bonus = np.where(d >= 1, 1.0 / (1.0 + d), 0.0)

    beta = float(beta)
    beta = 0.0 if beta < 0 else (1.0 if beta > 1 else beta)

    # Blend: query similarity stays primary; graph signal can nudge neighbors upward.
    # prox and hop_bonus are both in [0,1], so we can safely add them.
    graph_signal = np.clip(prox + 0.15 * hop_bonus, 0.0, 1.0)
    final = (1.0 - beta) * q_sims + beta * graph_signal

    order = np.argsort(-final)
    out: list[RetrievalResult] = []
    for rpos in order:
        out.append(RetrievalResult(idx=int(candidate_idx[int(rpos)]), score=float(final[int(rpos)])))
    return out



def main() -> None:
    ap = argparse.ArgumentParser(description="Graph-augmented retrieval evaluation (neighbor expansion + rerank).")
    ap.add_argument("--pages", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--chunking", type=Path, required=True)
    ap.add_argument("--cleaning", type=Path, required=True)
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)

    ap.add_argument("--graph-k", type=int, default=8, dest="graph_k")
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--expand-per-node", type=int, default=4, dest="expand_per_node")
    ap.add_argument("--beta", type=float, default=0.25)

    ap.add_argument("--out-dir", "--outdir", type=Path, default=Path("results/runs"), dest="outdir")
    ap.add_argument("--leaderboard", type=Path, default=None, help="Where to append the CSV summary row (optional).")
    args = ap.parse_args()
    repo_root = _find_repo_root(Path.cwd())

    args.outdir.mkdir(parents=True, exist_ok=True)

    chunk_cfg = yaml.safe_load(args.chunking.read_text(encoding="utf-8"))
    clean_cfg = yaml.safe_load(args.cleaning.read_text(encoding="utf-8"))
    embed_cfg = yaml.safe_load(args.embeddings.read_text(encoding="utf-8"))

    profile = profile_from_cfg(clean_cfg)
    embedder = build_embedder(embed_cfg)

    pages = load_pages_jsonl(args.pages)
    # Apply page-level cleaning before chunking (same convention as other eval runners)
    pages = [Page(page=p.page, text=apply_cleaning(p.text, profile)) for p in pages]
    chunker = build_chunker(chunk_cfg)

    strategy = chunk_cfg.get("strategy")

    # ---- chunking + cleaning ----
    if strategy == "tree_parent_child":
        parent_chunks, child_chunks = chunker.chunk(pages)
        parent_chunks = [c for c in parent_chunks if not is_noise_chunk(c.text, profile)]
        child_chunks = [c for c in child_chunks if not is_noise_chunk(c.text, profile)]

        chunks = child_chunks
        chunk_texts = [c.text for c in chunks]

        parent_texts = [c.text for c in parent_chunks]
        parent_matrix = embedder.fit_transform(parent_texts)
        child_matrix = embedder.transform(chunk_texts) if _is_numpy_matrix(parent_matrix) else embedder.fit_transform(chunk_texts)

        # retrieval on children; parent ids are stored by the chunker
        child_parent_ids = [getattr(c, "parent_id", None) for c in chunks]
        parent_ids = [getattr(c, "chunk_id", None) for c in parent_chunks]

        # pre-build parent retriever
        def parent_topk(qv: np.ndarray, k: int):
            return topk_cosine(qv, parent_matrix, k=k)

        doc_matrix = child_matrix
        # allowed children from top parents per question:
        parent_top_k = int(chunk_cfg.get("parent_top_k", 3))

        def base_retrieve(qv: np.ndarray, k: int):
            top_parents = parent_topk(qv, k=parent_top_k)
            allowed_parent = {parent_ids[r.idx] for r in top_parents}
            allowed_child_idx = [i for i, pid in enumerate(child_parent_ids) if pid in allowed_parent]
            if not allowed_child_idx:
                return topk_cosine(qv, doc_matrix, k=k)
            sub = doc_matrix[allowed_child_idx]
            sub_res = topk_cosine(qv, sub, k=min(k, len(allowed_child_idx)))
            # map back
            out: list[RetrievalResult] = []
            for r in sub_res:
                out.append(RetrievalResult(idx=int(allowed_child_idx[r.idx]), score=float(r.score)))
            return out

    else:
        chunks = chunker.chunk(pages)
        chunks = [c for c in chunks if not is_noise_chunk(c.text, profile)]
        chunk_texts = [c.text for c in chunks]
        doc_matrix = embedder.fit_transform(chunk_texts)

        def base_retrieve(qv: np.ndarray, k: int):
            return topk_cosine(qv, doc_matrix, k=k)

    # ---- build chunk graph ----
    graph = build_knn_graph(doc_matrix, graph_k=int(args.graph_k))

    # ---- evaluation ----
    questions = load_retrieval_questions(args.questions)

    baseline_ranks: list[int | None] = []
    augmented_ranks: list[int | None] = []
    per_q: list[dict] = []

    for q in questions:
        q_vec = embedder.transform([q.question])

        base_results = base_retrieve(q_vec, k=args.k)

        # baseline hit rank
        base_hit_rank = None
        base_top = []
        for rank, r in enumerate(base_results, start=1):
            c = chunks[r.idx]
            base_top.append(
                {
                    "rank": rank,
                    "score": float(r.score),
                    "chunk_id": c.chunk_id,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "section_title": c.section_title,
                    "chunk_text": c.text,
                }
            )
            if base_hit_rank is None:
                for ep in q.expected_pages:
                    if c.page_start <= ep <= c.page_end:
                        base_hit_rank = rank
                        break

        baseline_ranks.append(base_hit_rank)

        # graph augmentation
        seed = [r.idx for r in base_results]
        cand_dists = expand_candidates_with_dists(
            seed_idx=seed,
            graph=graph,
            hops=int(args.hops),
            expand_per_node=int(args.expand_per_node),
        )
        # rerank and take top-k again
        aug_results_all = rerank_with_graph(
            q_vec=q_vec,
            doc_matrix=doc_matrix,
            base_results=base_results,
            cand_dists=cand_dists,
            beta=float(args.beta),
        )
        aug_results = aug_results_all[: args.k]
        baseline_ids = [r.idx for r in base_results[: args.k]]
        aug_ids = [r.idx for r in aug_results]
        overlap_k = len(set(baseline_ids) & set(aug_ids))

        aug_hit_rank = None
        aug_top = []
        for rank, r in enumerate(aug_results, start=1):
            c = chunks[r.idx]
            aug_top.append(
                {
                    "rank": rank,
                    "score": float(r.score),
                    "chunk_id": c.chunk_id,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "section_title": c.section_title,
                    "chunk_text": c.text,
                }
            )
            if aug_hit_rank is None:
                for ep in q.expected_pages:
                    if c.page_start <= ep <= c.page_end:
                        aug_hit_rank = rank
                        break

        augmented_ranks.append(aug_hit_rank)

        per_q.append(
            {
                "id": q.id,
                "question": q.question,
                "expected_pages": q.expected_pages,
                "baseline": {"first_correct_rank": base_hit_rank, "top_k": base_top},
                "augmented": {
                    "first_correct_rank": aug_hit_rank,
                    "top_k": aug_top,
                    "graph_k": int(args.graph_k),
                    "hops": int(args.hops),
                    "expand_per_node": int(args.expand_per_node),
                    "beta": float(args.beta),
                    "candidate_count": len(cand_dists),
                    "added_candidates": max(0, len(cand_dists) - len(seed)),
                    "overlap_topk": int(overlap_k),
                },
            }
        )

    base_metrics = compute_metrics(baseline_ranks)
    aug_metrics = compute_metrics(augmented_ranks)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.outdir / f"graphaug_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "run_id": run_id,
        "pages": str(args.pages),
        "questions": str(args.questions),
        "chunking_config": str(args.chunking),
        "cleaning_config": str(args.cleaning),
        "embedding_config": str(args.embeddings),
        "k": int(args.k),
        "graph_k": int(args.graph_k),
        "hops": int(args.hops),
        "expand_per_node": int(args.expand_per_node),
        "beta": float(args.beta),
        "num_chunks": len(chunks),
        "baseline_metrics": base_metrics.__dict__,
        "augmented_metrics": aug_metrics.__dict__,
        "per_question": per_q,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # leaderboard
    if args.leaderboard is None:
        return

    leaderboard = Path(args.leaderboard)
    row = {
        "run_id": run_id,
        "chunking": _short_path(args.chunking, repo_root),
        "cleaning": _short_path(args.cleaning, repo_root),
        "embeddings": _short_path(args.embeddings, repo_root),
        "k": int(args.k),
        "graph_k": int(args.graph_k),
        "hops": int(args.hops),
        "expand_per_node": int(args.expand_per_node),
        "beta": float(args.beta),
        "num_chunks": len(chunks),
        "baseline_hit@1": base_metrics.hit_at_1,
        "baseline_hit@3": base_metrics.hit_at_3,
        "baseline_hit@5": base_metrics.hit_at_5,
        "baseline_mrr": base_metrics.mrr,
        "aug_hit@1": aug_metrics.hit_at_1,
        "aug_hit@3": aug_metrics.hit_at_3,
        "aug_hit@5": aug_metrics.hit_at_5,
        "aug_mrr": aug_metrics.mrr,
        "delta_hit@1": aug_metrics.hit_at_1 - base_metrics.hit_at_1,
        "delta_hit@3": aug_metrics.hit_at_3 - base_metrics.hit_at_3,
        "delta_hit@5": aug_metrics.hit_at_5 - base_metrics.hit_at_5,
        "delta_mrr": aug_metrics.mrr - base_metrics.mrr,
        "run_dir": _short_path(run_dir, repo_root),
    }

    write_header = not leaderboard.exists()
    with leaderboard.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print("=== Graph augmentation evaluation complete ===")
    print(f"Run directory: {run_dir}")
    print(f"Chunks: {len(chunks)} | graph_k={args.graph_k} hops={args.hops} expand_per_node={args.expand_per_node} beta={args.beta}")
    print(f"Baseline  Hit@1: {base_metrics.hit_at_1:.3f}  Hit@3: {base_metrics.hit_at_3:.3f}  Hit@5: {base_metrics.hit_at_5:.3f}  MRR: {base_metrics.mrr:.3f}")
    print(f"Augmented Hit@1: {aug_metrics.hit_at_1:.3f}  Hit@3: {aug_metrics.hit_at_3:.3f}  Hit@5: {aug_metrics.hit_at_5:.3f}  MRR: {aug_metrics.mrr:.3f}")
    print(f"Leaderboard appended: {leaderboard}")


if __name__ == "__main__":
    main()