from __future__ import annotations

"""Embedding model comparison runner.

Measures:
- Embedding time (documents + queries)
- Retrieval accuracy (Hit@K, MRR)
- Memory usage (CPU RSS + GPU peak if available)

This intentionally keeps the retrieval strategy fixed (dense cosine) so we compare
*embeddings* rather than the retriever.
"""

import argparse
import csv
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml

from ..chunkers.base import Page
from ..chunkers.naive_tokens import NaiveTokenChunker
from ..chunkers.structured import StructuredHeadingChunker
from ..chunkers.tree_parent_child import TreeParentChildChunker
from ..clean import normalize_text
from ..clean_profiles import apply_cleaning, is_noise_chunk, profile_from_cfg
from ..index.embed_factory import build_embedder
from ..index.retrieve import topk_cosine
from .datasets import load_retrieval_questions
from .metrics import compute_metrics


def load_pages_jsonl(path: Path) -> list[Page]:
    pages: list[Page] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pages.append(Page(page=int(obj["page"]), text=str(obj["text"])))
    return pages


def build_chunker(cfg: dict):
    strategy = cfg.get("strategy")
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


def topk_cosine_dense(q_vec, doc_mat, k: int):
    """Dense cosine top-k for numpy matrices.

    Assumes vectors are L2-normalized; uses dot product which equals cosine.
    """
    import numpy as np

    q = q_vec
    if hasattr(q_vec, "ndim") and q_vec.ndim == 1:
        q = q_vec.reshape(1, -1)

    scores = (doc_mat @ q.T).reshape(-1)
    idx = np.argsort(-scores)[: min(k, len(scores))]

    class R:
        def __init__(self, idx: int, score: float):
            self.idx = int(idx)
            self.score = float(score)

    return [R(int(i), float(scores[int(i)])) for i in idx]


def pick_retriever(matrix) -> Callable:
    return topk_cosine_dense if _is_numpy_matrix(matrix) else topk_cosine


def _rss_mb() -> float:
    """Resident Set Size (RSS) in MB."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def _gpu_peak_mb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return -1.0
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return -1.0


def main() -> None:
    p = argparse.ArgumentParser(description="Compare embedding models (time, memory, retrieval metrics)")
    p.add_argument("--pages", required=True, help="JSONL with per-page extracted text")
    p.add_argument("--questions", required=True, help="JSONL retrieval questions")
    p.add_argument("--chunking-config", required=True, help="YAML chunking config")
    p.add_argument("--cleaning-config", required=False, help="YAML cleaning profile config (optional)")
    p.add_argument("--embedding-config", required=True, help="YAML embedding config")
    p.add_argument("--k", type=int, default=5, help="Top-k to retrieve (default 5)")
    p.add_argument("--outdir", default="results/runs", help="Output directory for run artifacts")
    args = p.parse_args()

    pages_path = Path(args.pages)
    questions_path = Path(args.questions)
    chunk_cfg_path = Path(args.chunking_config)
    outdir = Path(args.outdir)

    chunk_cfg = yaml.safe_load(chunk_cfg_path.read_text(encoding="utf-8")) or {}
    chunker = build_chunker(chunk_cfg)

    # Cleaning profile (optional)
    clean_cfg: dict = {}
    if args.cleaning_config:
        clean_cfg = yaml.safe_load(Path(args.cleaning_config).read_text(encoding="utf-8")) or {}
    profile = profile_from_cfg(clean_cfg)

    # Embedding config (required)
    emb_cfg: dict = yaml.safe_load(Path(args.embedding_config).read_text(encoding="utf-8")) or {}
    embedder = build_embedder(emb_cfg)

    pages = load_pages_jsonl(pages_path)
    pages = [Page(page=p.page, text=apply_cleaning(p.text, profile)) for p in pages]

    # --- Chunking ---
    strategy = chunk_cfg.get("strategy")

    # Memory baselines
    rss_before = _rss_mb()
    gpu_peak_before = _gpu_peak_mb()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    # --- Document embedding timing ---
    t0 = time.perf_counter()

    if strategy == "tree_parent_child":
        parents, children = chunker.chunk(pages)

        if profile.drop_reference_like_chunks or profile.drop_abbreviation_like_chunks or profile.drop_low_signal_chunks:
            parents = [c for c in parents if not is_noise_chunk(c.text, profile)]
            children = [c for c in children if not is_noise_chunk(c.text, profile)]

        parent_texts = [normalize_text(c.text) for c in parents]
        child_texts = [normalize_text(c.text) for c in children]

        parent_matrix = embedder.fit_transform(parent_texts)
        child_matrix = embedder.transform(child_texts)

        child_parent_ids = [c.chunk_id.split("/")[0] for c in children]
        parent_ids = [p.chunk_id for p in parents]

        chunks = children
        parent_retrieve = pick_retriever(parent_matrix)
        child_retrieve = pick_retriever(child_matrix)
    else:
        chunks = chunker.chunk(pages)
        if profile.drop_reference_like_chunks or profile.drop_abbreviation_like_chunks or profile.drop_low_signal_chunks:
            chunks = [c for c in chunks if not is_noise_chunk(c.text, profile)]

        chunk_texts = [normalize_text(c.text) for c in chunks]
        doc_matrix = embedder.fit_transform(chunk_texts)
        doc_retrieve = pick_retriever(doc_matrix)

    doc_embed_seconds = time.perf_counter() - t0

    # --- Query embedding timing ---
    questions = load_retrieval_questions(questions_path)

    t1 = time.perf_counter()
    q_vecs = [embedder.transform([q.question]) for q in questions]
    query_embed_seconds = time.perf_counter() - t1

    # --- Retrieval evaluation ---
    first_ranks: list[int | None] = []
    per_q: list[dict] = []

    for q, q_vec in zip(questions, q_vecs, strict=True):
        if strategy == "tree_parent_child":
            parent_top_k = int(chunk_cfg.get("parent_top_k", 3))
            top_parents = parent_retrieve(q_vec, parent_matrix, k=parent_top_k)
            allowed_parent_ids = {parent_ids[r.idx] for r in top_parents}

            allowed_child_idx = [i for i, pid in enumerate(child_parent_ids) if pid in allowed_parent_ids]

            if not allowed_child_idx:
                results = child_retrieve(q_vec, child_matrix, k=args.k)
            else:
                sub_matrix = child_matrix[allowed_child_idx]
                sub_retrieve = pick_retriever(sub_matrix)
                sub_results = sub_retrieve(q_vec, sub_matrix, k=min(args.k, len(allowed_child_idx)))

                results = []
                for r in sub_results:
                    global_idx = allowed_child_idx[r.idx]
                    results.append(type(r)(idx=global_idx, score=r.score))
        else:
            results = doc_retrieve(q_vec, doc_matrix, k=args.k)

        retrieved = []
        hit_rank = None
        for rank, r in enumerate(results, start=1):
            c = chunks[r.idx]
            retrieved.append(
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
            if hit_rank is None:
                for ep in q.expected_pages:
                    if c.page_start <= ep <= c.page_end:
                        hit_rank = rank
                        break

        first_ranks.append(hit_rank)
        per_q.append(
            {
                "id": q.id,
                "question": q.question,
                "expected_pages": q.expected_pages,
                "first_correct_rank": hit_rank,
                "top_k": retrieved,
            }
        )

    metrics = compute_metrics(first_ranks)

    # Memory after
    rss_after = _rss_mb()
    gpu_peak_after = _gpu_peak_mb()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = outdir / f"embed_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save configs + artifacts
    (run_dir / "chunking_config.yaml").write_text(yaml.safe_dump(chunk_cfg, sort_keys=False), encoding="utf-8")
    if args.cleaning_config:
        (run_dir / "cleaning_config.yaml").write_text(yaml.safe_dump(clean_cfg, sort_keys=False), encoding="utf-8")
    (run_dir / "embedding_config.yaml").write_text(yaml.safe_dump(emb_cfg, sort_keys=False), encoding="utf-8")

    run_metrics = {
        **asdict(metrics),
        "chunking_strategy": strategy,
        "num_chunks": len(chunks),
        "doc_embedding_seconds": doc_embed_seconds,
        "query_embedding_seconds": query_embed_seconds,
        "rss_mb_before": rss_before,
        "rss_mb_after": rss_after,
        "gpu_peak_mb": gpu_peak_after,
        "gpu_peak_mb_before": gpu_peak_before,
    }

    (run_dir / "metrics.json").write_text(json.dumps(run_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "per_question.json").write_text(json.dumps(per_q, indent=2, ensure_ascii=False), encoding="utf-8")

    # Append summary leaderboard for embedding comparison
    summary_dir = Path("results/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = summary_dir / "leaderboard_embeddings.csv"

    row = {
        "run_id": run_id,
        "chunking_strategy": strategy,
        "cleaning": profile.name,
        "embedder_type": emb_cfg.get("type", ""),
        "embedder_model": emb_cfg.get("model_name", ""),
        "batch_size": emb_cfg.get("batch_size", ""),
        "chunk_size": chunk_cfg.get("child_chunk_size", chunk_cfg.get("chunk_size", "")),
        "overlap": chunk_cfg.get("child_overlap", chunk_cfg.get("overlap", "")),
        "num_chunks": len(chunks),
        "doc_embedding_seconds": doc_embed_seconds,
        "query_embedding_seconds": query_embed_seconds,
        "rss_mb_after": rss_after,
        "gpu_peak_mb": gpu_peak_after,
        "hit@1": metrics.hit_at_1,
        "hit@3": metrics.hit_at_3,
        "hit@5": metrics.hit_at_5,
        "mrr": metrics.mrr,
        "avg_first_rank": metrics.avg_first_rank,
        "run_dir": str(run_dir),
    }

    write_header = not leaderboard.exists()
    with leaderboard.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print("=== Embedding model evaluation complete ===")
    print(f"Run directory: {run_dir}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedder: {row['embedder_type']} {row['embedder_model']}".strip())
    print(f"Doc embedding seconds: {doc_embed_seconds:.3f} | Query embedding seconds: {query_embed_seconds:.3f}")
    if rss_after >= 0:
        print(f"CPU RSS (MB): {rss_after:.1f}")
    if gpu_peak_after >= 0:
        print(f"GPU peak allocated (MB): {gpu_peak_after:.1f}")
    print(f"Hit@1: {metrics.hit_at_1:.3f}  Hit@3: {metrics.hit_at_3:.3f}  Hit@5: {metrics.hit_at_5:.3f}")
    print(f"MRR: {metrics.mrr:.3f}  Avg first rank: {metrics.avg_first_rank}")
    print(f"Leaderboard appended: {leaderboard}")


if __name__ == "__main__":
    main()
