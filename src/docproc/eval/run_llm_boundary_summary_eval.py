from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import yaml

from ..chunkers.base import Page
from ..chunkers.llm_boundary import LLMBoundaryChunker, build_ollama_client_from_env
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


def topk_cosine_dense(q_vec, doc_mat, k: int):
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


def _is_numpy_matrix(x) -> bool:
    return type(x).__module__.startswith("numpy") and hasattr(x, "shape")


def pick_retriever(matrix):
    return topk_cosine_dense if _is_numpy_matrix(matrix) else topk_cosine


def summarize_chunk_llm(client, chunk_text: str, max_sentences: int = 3) -> str:
    # Keep prompt size bounded.
    text = chunk_text.strip()
    if len(text) > 6000:
        text = text[:6000] + "…"

    prompt = (
        "Summarize the following guideline passage for retrieval augmentation.\n"
        f"Constraints:\n- {max_sentences} sentences maximum\n- focus on concrete rules, thresholds, and validation steps\n"
        "Return ONLY the summary text (no markdown, no preamble).\n\n"
        "PASSAGE:\n"
        + text
    )
    out = client.generate(prompt).strip()
    # Clean common wrapper artifacts
    return out.strip().strip('"').strip()


def eval_variant(
    *,
    variant: str,
    chunks,
    retrieval_embedder,
    questions,
    k: int,
    augment_summaries: bool,
    client,
):
    if augment_summaries:
        augmented_texts = []
        summaries = []
        for c in chunks:
            s = summarize_chunk_llm(client, c.text)
            summaries.append(s)
            augmented_texts.append(f"Summary: {s}\n\n{c.text}")
        doc_texts = [normalize_text(t) for t in augmented_texts]
    else:
        summaries = [None] * len(chunks)
        doc_texts = [normalize_text(c.text) for c in chunks]

    doc_matrix = retrieval_embedder.fit_transform(doc_texts)
    doc_retrieve = pick_retriever(doc_matrix)

    first_ranks: list[int | None] = []
    per_q: list[dict] = []

    for q in questions:
        q_vec = retrieval_embedder.transform([q.question])
        results = doc_retrieve(q_vec, doc_matrix, k=k)

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
                    "summary": summaries[r.idx],
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
    return metrics, per_q, len(doc_texts)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate LLM boundary detection + summary augmentation")
    p.add_argument("--pages", required=True, help="JSONL with per-page extracted text")
    p.add_argument("--questions", required=True, help="JSONL retrieval questions")
    p.add_argument("--chunking-config", required=True, help="YAML chunking config for llm_boundary")
    p.add_argument("--cleaning-config", required=False, help="YAML cleaning profile config (optional)")
    p.add_argument("--embedding-config", required=False, help="YAML embedding config for retrieval (optional)")
    p.add_argument("--k", type=int, default=5, help="Top-k to retrieve (default 5)")
    p.add_argument("--outdir", default="results/runs", help="Output directory for run artifacts (default results/runs)")
    args = p.parse_args()

    pages_path = Path(args.pages)
    questions_path = Path(args.questions)
    chunk_cfg_path = Path(args.chunking_config)
    outdir = Path(args.outdir)

    chunk_cfg = yaml.safe_load(chunk_cfg_path.read_text(encoding="utf-8")) or {}
    strategy = str(chunk_cfg.get("strategy", "llm_boundary"))
    if strategy not in ("llm_boundary", "llm_boundary_summary"):
        raise ValueError(
            f"This runner only supports llm_boundary configs. Got strategy={strategy!r}. "
            "Use a config with strategy: llm_boundary (or llm_boundary_summary)."
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

    client = build_ollama_client_from_env()

    chunker = LLMBoundaryChunker(
        client=client,
        max_tokens=int(chunk_cfg.get("max_tokens", chunk_cfg.get("chunk_size", 450))),
        min_tokens=int(chunk_cfg.get("min_tokens", 80)),
        boundary_window=int(chunk_cfg.get("boundary_window", 6)),
        max_retries=int(chunk_cfg.get("max_retries", 2)),
        retry_backoff_s=float(chunk_cfg.get("retry_backoff_s", 0.8)),
    )

    pages = load_pages_jsonl(pages_path)
    pages = [Page(page=p.page, text=apply_cleaning(p.text, profile)) for p in pages]

    chunks = chunker.chunk(pages)

    if profile.drop_reference_like_chunks or profile.drop_abbreviation_like_chunks or profile.drop_low_signal_chunks:
        chunks = [c for c in chunks if not is_noise_chunk(c.text, profile)]

    questions = load_retrieval_questions(questions_path)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = outdir / f"llm_boundary_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save configs
    (run_dir / "chunking_config.yaml").write_text(yaml.safe_dump(chunk_cfg, sort_keys=False), encoding="utf-8")
    if args.cleaning_config:
        (run_dir / "cleaning_config.yaml").write_text(yaml.safe_dump(clean_cfg, sort_keys=False), encoding="utf-8")
    if args.embedding_config:
        (run_dir / "embedding_config.yaml").write_text(yaml.safe_dump(emb_cfg, sort_keys=False), encoding="utf-8")

    # Evaluate two variants:
    rows = []
    variants = [
        ("llm_boundary_raw", False),
        ("llm_boundary_summary_aug", True),
    ]

    for vname, do_aug in variants:
        metrics, per_q, _ = eval_variant(
            variant=vname,
            chunks=chunks,
            retrieval_embedder=retrieval_embedder,
            questions=questions,
            k=args.k,
            augment_summaries=do_aug,
            client=client,
        )

        v_dir = run_dir / vname
        v_dir.mkdir(parents=True, exist_ok=True)
        (v_dir / "metrics.json").write_text(json.dumps(metrics.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")
        (v_dir / "per_question.json").write_text(json.dumps(per_q, indent=2, ensure_ascii=False), encoding="utf-8")

        row = {
            "run_id": run_id,
            "variant": vname,
            "strategy": "llm_boundary",
            "cleaning": profile.name,
            "retrieval_embedder_type": emb_cfg.get("type", "tfidf"),
            "retrieval_embedder_model": emb_cfg.get("model_name", emb_cfg.get("model", "")),
            "ollama_base_url": client.base_url,
            "ollama_model": client.model,
            "max_tokens": chunker.max_tokens,
            "min_tokens": chunker.min_tokens,
            "boundary_window": chunker.boundary_window,
            "num_chunks": len(chunks),
            "hit@1": metrics.hit_at_1,
            "hit@3": metrics.hit_at_3,
            "hit@5": metrics.hit_at_5,
            "mrr": metrics.mrr,
            "avg_first_rank": metrics.avg_first_rank,
            "run_dir": str(v_dir),
        }
        rows.append(row)

        print(f"[{vname}] Hit@1={metrics.hit_at_1:.3f} Hit@3={metrics.hit_at_3:.3f} Hit@5={metrics.hit_at_5:.3f}  MRR={metrics.mrr:.3f}")

    summary_dir = Path("results/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = summary_dir / "llm_boundary_summary.csv"

    write_header = not leaderboard.exists()
    with leaderboard.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    print("=== LLM boundary + summary augmentation evaluation complete ===")
    print(f"Run directory: {run_dir}")
    print(f"Chunks: {len(chunks)}")
    print(f"Ollama: {client.base_url} | model={client.model}")
    print(f"Leaderboard appended: {leaderboard}")


if __name__ == "__main__":
    main()
