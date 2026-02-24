from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import yaml

from ..chunkers.base import Page
from ..chunkers.naive_tokens import NaiveTokenChunker
from ..chunkers.structured import StructuredHeadingChunker
from ..chunkers.tree_parent_child import TreeParentChildChunker
from ..clean import normalize_text
from ..index.embed import TfidfEmbedder
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


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate chunking via retrieval accuracy")
    p.add_argument("--pages", required=True, help="JSONL with per-page extracted text")
    p.add_argument("--questions", required=True, help="JSONL retrieval questions")
    p.add_argument("--chunking-config", required=True, help="YAML chunking config")
    p.add_argument("--k", type=int, default=5, help="Top-k to retrieve (default 5)")
    p.add_argument(
        "--outdir",
        default="results/runs",
        help="Output directory for run artifacts (default results/runs)",
    )
    args = p.parse_args()

    pages_path = Path(args.pages)
    questions_path = Path(args.questions)
    cfg_path = Path(args.chunking_config)
    outdir = Path(args.outdir)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    chunker = build_chunker(cfg)
    pages = load_pages_jsonl(pages_path)

    embedder = TfidfEmbedder()

    if cfg.get("strategy") == "tree_parent_child":
        parents, children = chunker.chunk(pages)

        parent_texts = [normalize_text(c.text) for c in parents]
        child_texts = [normalize_text(c.text) for c in children]

        parent_matrix = embedder.fit_transform(parent_texts)
        child_matrix = embedder.transform(child_texts)

        # map child -> parent_id
        child_parent_ids = [c.chunk_id.split("/")[0] for c in children]
        parent_ids = [p.chunk_id for p in parents]

        chunks = children  # for evaluation: we score based on retrieved child chunks
    else:
        chunks = chunker.chunk(pages)
        chunk_texts = [normalize_text(c.text) for c in chunks]
        doc_matrix = embedder.fit_transform(chunk_texts)

    questions = load_retrieval_questions(questions_path)
    first_ranks: list[int | None] = []
    per_q: list[dict] = []


    for q in questions:
        q_vec = embedder.transform([q.question])

        if cfg.get("strategy") == "tree_parent_child":
            parent_top_k = int(cfg.get("parent_top_k", 3))
            top_parents = topk_cosine(q_vec, parent_matrix, k=parent_top_k)

            allowed_parent_ids = {parent_ids[r.idx] for r in top_parents}

            # Only search children under the retrieved parents
            allowed_child_idx = [i for i, pid in enumerate(child_parent_ids) if pid in allowed_parent_ids]

            # If nothing matched, fall back to searching all children
            if not allowed_child_idx:
                results = topk_cosine(q_vec, child_matrix, k=args.k)
            else:
                sub_matrix = child_matrix[allowed_child_idx]
                sub_results = topk_cosine(q_vec, sub_matrix, k=min(args.k, len(allowed_child_idx)))

                # map back to global child indices
                results = []
                for r in sub_results:
                    global_idx = allowed_child_idx[r.idx]
                    results.append(type(r)(idx=global_idx, score=r.score))
        else:
            results = topk_cosine(q_vec, doc_matrix, k=args.k)

        # ---- evaluation (this is what was missing) ----
        retrieved = []
        hit_rank = None

        for rank, r in enumerate(results, start=1):
            c = chunks[r.idx]
            retrieved.append({
                "rank": rank,
                "score": float(r.score),
                "chunk_id": c.chunk_id,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "section_title": c.section_title,
            })

            # page match rule: any expected page in [page_start, page_end]
            if hit_rank is None:
                for ep in q.expected_pages:
                    if c.page_start <= ep <= c.page_end:
                        hit_rank = rank
                        break

        first_ranks.append(hit_rank)

        per_q.append({
            "id": q.id,
            "question": q.question,
            "expected_pages": q.expected_pages,
            "first_correct_rank": hit_rank,
            "top_k": retrieved,
        })


    metrics = compute_metrics(first_ranks)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = outdir / f"docproc_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save run artifacts
    (run_dir / "chunking_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics.__dict__, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "per_question.json").write_text(
        json.dumps(per_q, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Append/update summary leaderboard
    summary_dir = Path("results/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = summary_dir / "leaderboard.csv"
    row = {
        "run_id": run_id,
        "strategy": cfg.get("strategy"),
        "chunk_size": cfg.get("child_chunk_size", cfg.get("chunk_size", "")),
        "overlap": cfg.get("child_overlap", cfg.get("overlap", "")),
        "max_tokens": cfg.get("parent_max_tokens", cfg.get("max_tokens", "")),
        "min_tokens": cfg.get("min_tokens", ""),
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

    print("=== Retrieval evaluation complete ===")
    print(f"Run directory: {run_dir}")
    print(f"Chunks: {len(chunks)}")
    print(f"Hit@1: {metrics.hit_at_1:.3f}  Hit@3: {metrics.hit_at_3:.3f}  Hit@5: {metrics.hit_at_5:.3f}")
    print(f"MRR: {metrics.mrr:.3f}  Avg first rank: {metrics.avg_first_rank}")
    print(f"Leaderboard appended: {leaderboard}")


if __name__ == "__main__":
    main()
