from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from ..chunkers.base import Page
from ..chunkers.naive_tokens import NaiveTokenChunker
from ..chunkers.page_index import PageIndexChunker
from ..chunkers.structured import StructuredHeadingChunker
from ..chunkers.tree_parent_child import TreeParentChildChunker
from ..chunkers.late_chunking import LateChunkingChunker
from ..clean import normalize_text, simple_tokenize
from ..clean_profiles import apply_cleaning, is_noise_chunk, profile_from_cfg
from ..index.embed_factory import build_embedder
from ..index.reranker_factory import build_reranker
from ..index.retrievers import BM25Index, merge_hybrid, pick_dense_retriever, rerank_cross_encoder
from ..index.rewriters import build_rewriter
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
    if strategy == "page_index":
        return PageIndexChunker()
    if strategy == "late_chunking":
        return LateChunkingChunker()
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

def chunk_text_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    toks = simple_tokenize(text)
    if chunk_size <= 0:
        return [" ".join(toks)]
    step = max(1, chunk_size - overlap)
    out = []
    for i in range(0, len(toks), step):
        window = toks[i : i + chunk_size]
        if not window:
            continue
        out.append(" ".join(window))
        if i + chunk_size >= len(toks):
            break
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate retrieval strategies (dense/bm25/hybrid/parent-child/reranking/rewriting)")
    p.add_argument("--pages", required=True, help="JSONL with per-page extracted text")
    p.add_argument("--questions", required=True, help="JSONL retrieval questions")
    p.add_argument("--chunking-config", required=True, help="YAML chunking config")
    p.add_argument("--retrieval-config", required=True, help="YAML retrieval strategy config")
    p.add_argument("--cleaning-config", required=False, help="YAML cleaning profile config (optional)")
    p.add_argument("--embedding-config", required=False, help="YAML embedding config (optional, dense part)")
    p.add_argument("--k", type=int, default=5, help="Top-k to retrieve (default 5)")
    p.add_argument("--outdir", default="results/runs", help="Output directory for run artifacts (default results/runs)")
    args = p.parse_args()

    pages_path = Path(args.pages)
    questions_path = Path(args.questions)
    chunk_cfg_path = Path(args.chunking_config)
    retr_cfg_path = Path(args.retrieval_config)
    outdir = Path(args.outdir)

    chunk_cfg = yaml.safe_load(chunk_cfg_path.read_text(encoding="utf-8")) or {}
    retr_cfg = yaml.safe_load(retr_cfg_path.read_text(encoding="utf-8")) or {}

    chunker = build_chunker(chunk_cfg)

    # Cleaning profile (optional)
    clean_cfg: dict = {}
    if args.cleaning_config:
        clean_cfg = yaml.safe_load(Path(args.cleaning_config).read_text(encoding="utf-8")) or {}
    profile = profile_from_cfg(clean_cfg)

    # Embedding config (optional)
    emb_cfg: dict = {}
    if args.embedding_config:
        emb_cfg = yaml.safe_load(Path(args.embedding_config).read_text(encoding="utf-8")) or {}
    embedder = build_embedder(emb_cfg)

    # Optional query rewriting + optional cross-encoder reranking
    rewriter = build_rewriter(retr_cfg.get("rewriter"))
    reranker = build_reranker(retr_cfg.get("reranker"))

    pages = load_pages_jsonl(pages_path)
    pages = [Page(page=p.page, text=apply_cleaning(p.text, profile)) for p in pages]

    # ---- Chunking ----
    strategy = chunk_cfg.get("strategy")
    if strategy == "tree_parent_child":
        parents, children = chunker.chunk(pages)
        if profile.drop_reference_like_chunks or profile.drop_abbreviation_like_chunks or profile.drop_low_signal_chunks:
            parents = [c for c in parents if not is_noise_chunk(c.text, profile)]
            children = [c for c in children if not is_noise_chunk(c.text, profile)]

        parent_texts = [normalize_text(c.text) for c in parents]
        child_texts = [normalize_text(c.text) for c in children]

        parent_matrix = embedder.fit_transform(parent_texts)
        child_matrix = embedder.transform(child_texts)

        parent_ids = [c.chunk_id for c in parents]
        child_parent_ids = [c.chunk_id.split("/")[0] for c in children]

        chunks = children
        dense_child_retrieve = pick_dense_retriever(child_matrix)
        dense_parent_retrieve = pick_dense_retriever(parent_matrix)
        # BM25 over children (optional for hybrid within allowed set)
        bm25_children = BM25Index([simple_tokenize(t) for t in child_texts])
    else:
        chunks = chunker.chunk(pages)
        if profile.drop_reference_like_chunks or profile.drop_abbreviation_like_chunks or profile.drop_low_signal_chunks:
            chunks = [c for c in chunks if not is_noise_chunk(c.text, profile)]
        doc_texts = [normalize_text(c.text) for c in chunks]

        doc_matrix = embedder.fit_transform(doc_texts)
        dense_retrieve = pick_dense_retriever(doc_matrix)
        bm25 = BM25Index([simple_tokenize(t) for t in doc_texts])

    # ---- Eval loop ----
    questions = load_retrieval_questions(questions_path)
    first_ranks: list[int | None] = []
    per_q: list[dict] = []

    base_type = str(retr_cfg.get("type", "dense")).lower()

    if base_type == "late_chunking":
        page_retr = str(retr_cfg.get("page_retriever", "dense")).lower()
        retr_type = f"late_chunking_{page_retr}"
    else:
        retr_type = base_type

    is_late = (base_type == "late_chunking")

    dense_w = float(retr_cfg.get("dense_weight", 0.5))
    bm25_w = float(retr_cfg.get("bm25_weight", 0.5))
    candidate_k = int(retr_cfg.get("candidate_k", max(50, args.k * 10)))

    for q in questions:
        rr = rewriter.rewrite(q.question)
        query_text = rr.rewritten

        # Embed query once (dense path)
        q_vec = embedder.transform([query_text])

        # Late chunking retrieval
        if is_late:
            # You indexed pages (1 chunk per page). Now:
            # 1) retrieve top pages
            # 2) sub-chunk those pages at query time
            # 3) dense score sub-chunks and return top-k

            page_top_k = int(retr_cfg.get("page_top_k", retr_cfg.get("parent_top_k", 12)))
            sub_chunk_size = int(retr_cfg.get("sub_chunk_size", 300))
            sub_overlap = int(retr_cfg.get("sub_overlap", 100))
            sub_candidate_k = int(retr_cfg.get("sub_candidate_k", max(50, args.k * 10)))

            # Retrieve pages using the configured "page_retriever" (default dense)
            page_retriever = str(retr_cfg.get("page_retriever", "dense")).lower()

            if page_retriever in ("bm25", "sparse"):
                page_scores = bm25.get_scores(simple_tokenize(query_text))
                page_idx = np.argsort(-page_scores)[: min(page_top_k, len(page_scores))]
                top_pages = [int(i) for i in page_idx]
            elif page_retriever in ("hybrid", "bm25+dense", "bm25_bge"):
                dense_pages = dense_retrieve(q_vec, doc_matrix, k=min(sub_candidate_k, len(chunks)))
                page_scores = bm25.get_scores(simple_tokenize(query_text))
                hybrid_pages = merge_hybrid(
                    dense_pages, page_scores, dense_weight=dense_w, bm25_weight=bm25_w, k=min(page_top_k, len(chunks))
                )
                top_pages = [r.idx for r in hybrid_pages]
            else:
                dense_pages = dense_retrieve(q_vec, doc_matrix, k=min(page_top_k, len(chunks)))
                top_pages = [r.idx for r in dense_pages]

            # Build query-time subchunks for these pages
            sub_texts: list[str] = []
            sub_meta: list[dict[str, Any]] = []

            for pi in top_pages:
                page_chunk = chunks[pi]
                pieces = chunk_text_tokens(normalize_text(page_chunk.text), sub_chunk_size, sub_overlap)
                for si, piece in enumerate(pieces):
                    sub_texts.append(piece)
                    sub_meta.append(
                        {
                            "chunk_id": f"{page_chunk.chunk_id}/sub{si}",
                            "page_start": page_chunk.page_start,
                            "page_end": page_chunk.page_end,
                            "section_title": page_chunk.section_title,
                            "text": piece,
                        }
                    )

            if not sub_texts:
                results = []
            else:
                sub_matrix = embedder.transform(sub_texts)
                sub_dense = pick_dense_retriever(sub_matrix)
                sub_candidates = sub_dense(q_vec, sub_matrix, k=min(sub_candidate_k, len(sub_texts)))
                results = sub_candidates[: args.k]

            # Optional rerank on subchunks
            if reranker is not None and str(retr_cfg.get("rerank_on", "candidates")).lower() in ("candidates", "topk", "all"):
                base = results
                # build list-like object compatible with rerank_cross_encoder: needs idx+score and doc_texts array
                doc_texts_sub = sub_texts
                results = rerank_cross_encoder(
                    query_text, base, doc_texts_sub, reranker, k=args.k, batch_size=int(retr_cfg.get("reranker_batch_size", 32))
                )

            # Convert to the same "chunks[r.idx]" style by synthesizing lightweight chunk objects
            # (so your existing evaluation block can stay unchanged)
            from ..chunkers.base import Chunk
            synth_chunks: list[Chunk] = []
            for m in sub_meta:
                synth_chunks.append(
                    Chunk(
                        chunk_id=m["chunk_id"],
                        text=m["text"],
                        page_start=m["page_start"],
                        page_end=m["page_end"],
                        section_title=m["section_title"],
                        strategy="late_chunking",
                    )
                )

            # Use synthesized chunks for evaluation output
            chunks_backup = chunks
            chunks = synth_chunks

            # fall through to your existing evaluation block below (it uses `chunks[r.idx]`)
            # but restore at the end of this question to avoid affecting next question’s indices
            # We'll restore right after the evaluation section.

        if not is_late:
            # 1) get initial candidates
            if strategy == "tree_parent_child":
                parent_top_k = int(retr_cfg.get("parent_top_k", chunk_cfg.get("parent_top_k", 3)))
                top_parents = dense_parent_retrieve(q_vec, parent_matrix, k=parent_top_k)
                allowed_parent_ids = {parent_ids[r.idx] for r in top_parents}
                allowed_child_idx = [i for i, pid in enumerate(child_parent_ids) if pid in allowed_parent_ids]

                # Dense-only within allowed set
                if not allowed_child_idx:
                    dense_candidates = dense_child_retrieve(q_vec, child_matrix, k=min(candidate_k, len(chunks)))
                    bm25_scores = bm25_children.get_scores(simple_tokenize(query_text))
                    candidates = dense_candidates
                else:
                    sub_mat = child_matrix[allowed_child_idx]
                    sub_dense = pick_dense_retriever(sub_mat)
                    sub_dense_res = sub_dense(q_vec, sub_mat, k=min(candidate_k, len(allowed_child_idx)))
                    dense_candidates = [type(r)(idx=int(allowed_child_idx[r.idx]), score=r.score) for r in sub_dense_res]
                    bm25_scores_full = bm25_children.get_scores(simple_tokenize(query_text))
                    candidates = dense_candidates

                if retr_type in ("bm25", "sparse"):
                    bm25_scores = bm25_children.get_scores(simple_tokenize(query_text))
                    top = np.argsort(-bm25_scores)[: min(args.k, len(bm25_scores))]
                    from ..index.retrieve import RetrievalResult
                    results = [RetrievalResult(idx=int(i), score=float(bm25_scores[int(i)])) for i in top]
                elif retr_type in ("hybrid", "bm25+dense", "bm25_bge"):
                    bm25_scores = bm25_children.get_scores(simple_tokenize(query_text))
                    results = merge_hybrid(dense_candidates, bm25_scores, dense_weight=dense_w, bm25_weight=bm25_w, k=args.k)
                else:
                    # dense / parent-child default
                    results = dense_candidates[: args.k]

            else:
                dense_candidates = dense_retrieve(q_vec, doc_matrix, k=min(candidate_k, len(chunks)))

                if retr_type in ("dense", "vector"):
                    results = dense_candidates[: args.k]
                elif retr_type in ("bm25", "sparse"):
                    bm25_scores = bm25.get_scores(simple_tokenize(query_text))
                    top = np.argsort(-bm25_scores)[: min(args.k, len(bm25_scores))]
                    results = [type(dense_candidates[0] if dense_candidates else object())(idx=int(i), score=float(bm25_scores[int(i)])) for i in top]
                    # above type hack; normalize to RetrievalResult
                    from ..index.retrieve import RetrievalResult
                    results = [RetrievalResult(idx=int(i), score=float(bm25_scores[int(i)])) for i in top]
                elif retr_type in ("hybrid", "hybrid_bge", "hybrid_dense", "bm25+dense", "bm25_bge"):
                    bm25_scores = bm25.get_scores(simple_tokenize(query_text))
                    results = merge_hybrid(dense_candidates, bm25_scores, dense_weight=dense_w, bm25_weight=bm25_w, k=args.k)
                else:
                    raise ValueError(f"Unknown retrieval type: {retr_type}")

        else:
            # Late chunking already produced 'results' and synthesized 'chunks'
            dense_candidates = results

        # 2) optional cross-encoder reranking
        if reranker is not None and str(retr_cfg.get("rerank_on", "candidates")).lower() in ("candidates", "topk", "all"):
            # rerank over a bigger set to be meaningful
            base = results
            if str(retr_cfg.get("rerank_on", "candidates")).lower() == "candidates":
                base = dense_candidates[: min(candidate_k, len(dense_candidates))]
                if retr_type in ("hybrid", "bm25+dense", "bm25_bge"):
                    # union candidates via hybrid merge already handled by merge_hybrid, but we can rerank top candidate_k from dense
                    base = dense_candidates[: min(candidate_k, len(dense_candidates))]
            doc_texts = [normalize_text(c.text) for c in chunks]
            results = rerank_cross_encoder(query_text, base, doc_texts, reranker, k=args.k, batch_size=int(retr_cfg.get("reranker_batch_size", 32)))

        # ---- evaluation ----
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

        # Restore chunks after late chunking synthesized list
        if is_late:
            chunks = chunks_backup

        first_ranks.append(hit_rank)
        per_q.append(
            {
                "id": q.id,
                "question": q.question,
                "rewritten_question": query_text if query_text != q.question else None,
                "rewriter_method": rr.method,
                "expected_pages": q.expected_pages,
                "first_correct_rank": hit_rank,
                "top_k": retrieved,
            }
        )

    metrics = compute_metrics(first_ranks)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = outdir / f"retrieval_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "chunking_config.yaml").write_text(yaml.safe_dump(chunk_cfg, sort_keys=False), encoding="utf-8")
    (run_dir / "retrieval_config.yaml").write_text(yaml.safe_dump(retr_cfg, sort_keys=False), encoding="utf-8")
    if args.cleaning_config:
        (run_dir / "cleaning_config.yaml").write_text(yaml.safe_dump(clean_cfg, sort_keys=False), encoding="utf-8")
    if args.embedding_config:
        (run_dir / "embedding_config.yaml").write_text(yaml.safe_dump(emb_cfg, sort_keys=False), encoding="utf-8")

    (run_dir / "metrics.json").write_text(json.dumps(metrics.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "per_question.json").write_text(json.dumps(per_q, indent=2, ensure_ascii=False), encoding="utf-8")

    # Leaderboard
    summary_dir = Path("results/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = summary_dir / "leaderboard_retrieval.csv"

    row = {
        "run_id": run_id,
        "chunking_strategy": strategy,
        "retrieval_type": retr_type,
        "page_retriever": retr_cfg.get("page_retriever", "") if base_type == "late_chunking" else "",
        "rewriter": retr_cfg.get("rewriter", {}).get("type", "none") if isinstance(retr_cfg.get("rewriter"), dict) else "none",
        "reranker": retr_cfg.get("reranker", {}).get("model_name", "") if isinstance(retr_cfg.get("reranker"), dict) else "",
        "cleaning": profile.name,
        "embedder_type": emb_cfg.get("type", "tfidf"),
        "embedder_model": emb_cfg.get("model_name", ""),
        "chunk_size": chunk_cfg.get("child_chunk_size", chunk_cfg.get("chunk_size", "")),
        "overlap": chunk_cfg.get("child_overlap", chunk_cfg.get("overlap", "")),
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

    print("=== Retrieval strategy evaluation complete ===")
    print(f"Run directory: {run_dir}")
    print(f"Chunks: {len(chunks)}")
    print(f"Chunking: {strategy} | Retrieval: {retr_type}")
    print(f"Hit@1: {metrics.hit_at_1:.3f}  Hit@3: {metrics.hit_at_3:.3f}  Hit@5: {metrics.hit_at_5:.3f}")
    print(f"MRR: {metrics.mrr:.3f}  Avg first rank: {metrics.avg_first_rank}")
    print(f"Leaderboard appended: {leaderboard}")


if __name__ == "__main__":
    main()
