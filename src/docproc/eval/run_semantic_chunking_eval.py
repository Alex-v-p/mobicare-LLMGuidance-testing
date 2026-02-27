from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml

from ..chunkers.base import Page
from ..chunkers.semantic import SemanticSimilarityChunker
from ..clean import normalize_text
from ..clean_profiles import apply_cleaning, is_noise_chunk, profile_from_cfg
from ..index.embed_factory import STEmbedder, build_embedder
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


def _is_numpy_matrix(x: Any) -> bool:
    return type(x).__module__.startswith("numpy") and hasattr(x, "shape")


def topk_cosine_dense(q_vec, doc_mat, k: int):
    """Dense cosine top-k for numpy matrices.

    Assumes vectors are L2-normalized (true for many ST embeddings when configured).
    Falls back to dot-product which equals cosine when normalized.
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


def build_semantic_embedder(chunk_cfg: dict, emb_cfg: dict) -> STEmbedder:
    """Build the embedder used *only* for semantic boundary detection."""

    # Optional override in chunking config.
    sem_cfg = chunk_cfg.get("semantic_embedder") or {}

    # If not provided, reuse retrieval embedder *only if* it is ST.
    if not sem_cfg:
        typ = str(emb_cfg.get("type", "tfidf")).lower()
        if typ in ("sentence_transformers", "st", "sbert"):
            sem_cfg = emb_cfg

    # Default.
    if not sem_cfg:
        sem_cfg = {
            "type": "sentence_transformers",
            "model_name": "BAAI/bge-small-en-v1.5",
            "normalize_embeddings": True,
            "device": "auto",
        }

    sem_embedder = build_embedder(sem_cfg)
    if not isinstance(sem_embedder, STEmbedder):
        raise ValueError(
            "Semantic chunking requires a SentenceTransformers embedder. "
            "Set chunking_config.semantic_embedder.type=sentence_transformers (or pass an ST embedding-config)."
        )
    return sem_embedder


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate semantic chunking via retrieval accuracy")
    p.add_argument("--pages", required=True, help="JSONL with per-page extracted text")
    p.add_argument("--questions", required=True, help="JSONL retrieval questions")
    p.add_argument("--chunking-config", required=True, help="YAML semantic chunking config")
    p.add_argument("--cleaning-config", required=False, help="YAML cleaning profile config (optional)")
    p.add_argument("--embedding-config", required=False, help="YAML embedding config for retrieval (optional)")
    p.add_argument("--k", type=int, default=5, help="Top-k to retrieve (default 5)")
    p.add_argument(
        "--outdir",
        default="results/runs",
        help="Output directory for run artifacts (default results/runs)",
    )
    args = p.parse_args()

    pages_path = Path(args.pages)
    questions_path = Path(args.questions)
    chunk_cfg_path = Path(args.chunking_config)
    outdir = Path(args.outdir)

    chunk_cfg = yaml.safe_load(chunk_cfg_path.read_text(encoding="utf-8")) or {}
    strategy = str(chunk_cfg.get("strategy", "semantic_similarity"))
    if strategy not in ("semantic", "semantic_similarity"):
        raise ValueError(
            f"This runner only supports semantic chunking. Got strategy={strategy!r}. "
            "Use a config with strategy: semantic (or semantic_similarity)."
        )

    # Cleaning profile (optional)
    clean_cfg: dict = {}
    if args.cleaning_config:
        clean_cfg = yaml.safe_load(Path(args.cleaning_config).read_text(encoding="utf-8")) or {}
    profile = profile_from_cfg(clean_cfg)

    # Retrieval embedder (optional)
    emb_cfg: dict = {}
    if args.embedding_config:
        emb_cfg = yaml.safe_load(Path(args.embedding_config).read_text(encoding="utf-8")) or {}
    retrieval_embedder = build_embedder(emb_cfg)

    semantic_embedder = build_semantic_embedder(chunk_cfg, emb_cfg)

    chunker = SemanticSimilarityChunker(
        embedder=semantic_embedder,
        max_tokens=int(chunk_cfg.get("max_tokens", chunk_cfg.get("chunk_size", 450))),
        min_tokens=int(chunk_cfg.get("min_tokens", 80)),
        similarity_threshold=float(chunk_cfg.get("similarity_threshold", 0.55)),
        batch_size=int(chunk_cfg.get("batch_size", 128)),
    )

    pages = load_pages_jsonl(pages_path)
    pages = [Page(page=p.page, text=apply_cleaning(p.text, profile)) for p in pages]

    chunks = chunker.chunk(pages)

    if profile.drop_reference_like_chunks or profile.drop_abbreviation_like_chunks or profile.drop_low_signal_chunks:
        chunks = [c for c in chunks if not is_noise_chunk(c.text, profile)]

    chunk_texts = [normalize_text(c.text) for c in chunks]
    doc_matrix = retrieval_embedder.fit_transform(chunk_texts)
    doc_retrieve = pick_retriever(doc_matrix)

    questions = load_retrieval_questions(questions_path)
    first_ranks: list[int | None] = []
    per_q: list[dict] = []

    for q in questions:
        q_vec = retrieval_embedder.transform([q.question])
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

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = outdir / f"semantic_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save run artifacts
    (run_dir / "chunking_config.yaml").write_text(yaml.safe_dump(chunk_cfg, sort_keys=False), encoding="utf-8")
    if args.cleaning_config:
        (run_dir / "cleaning_config.yaml").write_text(yaml.safe_dump(clean_cfg, sort_keys=False), encoding="utf-8")
    if args.embedding_config:
        (run_dir / "embedding_config.yaml").write_text(
            yaml.safe_dump(emb_cfg, sort_keys=False), encoding="utf-8"
        )

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics.__dict__, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "per_question.json").write_text(
        json.dumps(per_q, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Append summary CSV
    summary_dir = Path("results/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = summary_dir / "semantic_chunking.csv"

    row = {
        "run_id": run_id,
        "strategy": "semantic_similarity",
        "cleaning": profile.name,
        "retrieval_embedder_type": emb_cfg.get("type", "tfidf"),
        "retrieval_embedder_model": emb_cfg.get("model_name", emb_cfg.get("model", "")),
        "semantic_embedder_model": getattr(semantic_embedder, "model_name", ""),
        "max_tokens": chunker.max_tokens,
        "min_tokens": chunker.min_tokens,
        "similarity_threshold": chunker.similarity_threshold,
        "num_chunks": len(chunks),
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

    print("=== Semantic chunking evaluation complete ===")
    print(f"Run directory: {run_dir}")
    print(f"Chunks: {len(chunks)}")
    print(
        f"Cleaning: {profile.name} | Retrieval embedder: {row['retrieval_embedder_type']} {row['retrieval_embedder_model']}"
    )
    print(f"Semantic embedder: {row['semantic_embedder_model']}")
    print(f"Hit@1: {metrics.hit_at_1:.3f}  Hit@3: {metrics.hit_at_3:.3f}  Hit@5: {metrics.hit_at_5:.3f}")
    print(f"MRR: {metrics.mrr:.3f}  Avg first rank: {metrics.avg_first_rank}")
    print(f"Leaderboard appended: {leaderboard}")


if __name__ == "__main__":
    main()
