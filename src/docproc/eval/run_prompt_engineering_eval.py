from __future__ import annotations

"""Prompt engineering eval runner (simple, reproducible).

This experiment compares multiple *prompt templates* for RAG answering over the
same retrieval setup.

Design goals
------------
- Re-use existing chunking/cleaning/embedding configs.
- Use the existing Ollama client implementation.
- Produce artifacts under results/runs and append a CSV row under results/summary.

Metrics (lightweight heuristics)
-------------------------------
- answer_nonempty_rate: fraction of questions with a non-empty answer.
- cites_rate: fraction where the model produced at least one citation.
- citation_hallucination_rate: fraction of cited pages not present in retrieved context.
- expected_page_hit_rate: fraction where any cited page overlaps expected_pages (if provided).
- length_p50_tokens / length_p95_tokens: token-length consistency.

Note: This runner intentionally does *not* attempt medical correctness; it only
measures grounding/format/consistency signals that are comparable across prompts.
"""

import argparse
import csv
import json
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..chunkers.base import Page
from ..chunkers.llm_boundary import build_ollama_client_from_env
from ..clean import normalize_text, simple_tokenize
from ..clean_profiles import apply_cleaning, is_noise_chunk, profile_from_cfg
from ..index.embed_factory import build_embedder
from ..index.retrieve import topk_cosine
from .datasets import load_retrieval_questions


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


def _topk_cosine_dense(q_vec, doc_mat, k: int):
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


def _pick_retriever(matrix):
    return _topk_cosine_dense if _is_numpy_matrix(matrix) else topk_cosine


def _p50(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else -1.0


def _p95(xs: list[float]) -> float:
    if not xs:
        return -1.0
    xs2 = sorted(xs)
    i = int(round(0.95 * (len(xs2) - 1)))
    return float(xs2[max(0, min(i, len(xs2) - 1))])

def _is_not_found(ans: str) -> bool:
    a = (ans or "").strip()
    if not a:
        return True
    # allow both plain and formatted variants
    if a.upper() == "NOT_FOUND":
        return True
    if re.search(r"(?i)\banswer\s*:\s*NOT_FOUND\b", a):
        return True
    return False


def _extract_cited_pages(text: str) -> list[int]:
    if not text:
        return []

    out: list[int] = []
    s = text.strip()

    # 1) JSON citations: {"citations": ["p28", "p 14", "P15"]}
    if s.startswith("{"):
        try:
            obj = json.loads(s)
            cites = obj.get("citations", [])
            if isinstance(cites, list):
                for c in cites:
                    if not isinstance(c, str):
                        continue
                    m = re.search(r"(?i)\bp\s*(\d{1,4})\b", c)
                    if m:
                        out.append(int(m.group(1)))
        except Exception:
            pass

    # 2) Bracket citations with optional ranges: [p12] or [p12-p14] or [p12-p12]
    for m in re.finditer(r"\[\s*[pP]\s*(\d{1,4})(?:\s*-\s*[pP]?\s*(\d{1,4}))?\s*\]", s):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        lo, hi = (a, b) if a <= b else (b, a)
        out.extend(list(range(lo, hi + 1)))

    # de-dup preserve order
    seen = set()
    deduped: list[int] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def _build_context_snippet(retrieved: list[dict[str, Any]], max_chars: int = 10_000) -> str:
    """Build bounded context string with explicit page range markers."""
    parts: list[str] = []
    for r in retrieved:
        ps, pe = int(r["page_start"]), int(r["page_end"])
        txt = str(r["text"]).strip()
        if len(txt) > 3500:
            txt = txt[:3500] + "…"
        parts.append(f"[p{ps}-p{pe}]\n{txt}")
    s = "\n\n".join(parts)
    if len(s) > max_chars:
        s = s[:max_chars] + "…"
    return s


@dataclass
class PromptTemplate:
    key: str
    name: str
    prompt: str


def _templates() -> list[PromptTemplate]:
    # Keep prompts compact; enforce citations so hallucination can be measured.
    strict_med = (
        "You are answering from a clinical guideline context.\n"
        "Return ONLY valid JSON on one line with keys:\n"
        "- answer (string)\n- contraindications (array of strings, optional)\n"
        "- citations (array of page markers like 'p12', 'p13')\n"
        "Rules:\n"
        "- Use ONLY the provided CONTEXT.\n"
        "- Every claim must be backed by a citation page in citations.\n"
        "- If the context does not support an answer, set answer='NOT_FOUND' and citations=[].\n\n"
        "QUESTION:\n{question}\n\nCONTEXT:\n{context}\n"
    )

    structured_reasoning = (
        "You are answering from a clinical guideline context.\n"
        "Write a short, structured response with exactly these sections:\n"
        "Reasoning (2-4 bullet points)\nAnswer (1-3 sentences)\nCitations (list of [p##])\n"
        "Rules:\n"
        "- Use ONLY the provided CONTEXT.\n"
        "- Cite pages as [p##] in the Citations section.\n"
        "- If unsupported, say 'NOT_FOUND' in Answer and provide no citations.\n\n"
        "QUESTION:\n{question}\n\nCONTEXT:\n{context}\n"
    )

    hard_grounding = (
        "Answer ONLY if the answer is explicitly supported by the CONTEXT.\n"
        "If it is not supported, respond with exactly: NOT_FOUND\n\n"
        "If supported, respond in this format:\n"
        "Answer: <your answer>\n"
        "Citations: [p##], [p##]\n\n"
        "QUESTION:\n{question}\n\nCONTEXT:\n{context}\n"
    )

    return [
        PromptTemplate("strict_med", "Strict medical JSON", strict_med),
        PromptTemplate("structured", "Structured reasoning", structured_reasoning),
        PromptTemplate("hard_grounding", "Hard grounding", hard_grounding),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate prompt templates for RAG answering")
    p.add_argument("--pages", required=True, help="JSONL with per-page extracted text")
    p.add_argument("--questions", required=True, help="JSONL retrieval questions (id, question, expected_pages)")
    p.add_argument("--chunking-config", required=True, help="YAML chunking config")
    p.add_argument("--cleaning-config", required=False, help="YAML cleaning profile config (optional)")
    p.add_argument("--embedding-config", required=False, help="YAML embedding config (optional)")
    p.add_argument("--k", type=int, default=5, help="Top-k to retrieve (default 5)")
    p.add_argument("--outdir", default="results", help="Results root (default results)")
    p.add_argument(
        "--variant",
        default="all",
        choices=["all", "strict_med", "structured", "hard_grounding"],
        help="Which prompt variant to run",
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[3]

    def _abs(pth: str | None) -> Path | None:
        if not pth:
            return None
        pp = Path(pth)
        return pp if pp.is_absolute() else (repo_root / pp)

    pages_path = _abs(args.pages)
    questions_path = _abs(args.questions)
    chunk_cfg_path = _abs(args.chunking_config)
    clean_cfg_path = _abs(args.cleaning_config)
    emb_cfg_path = _abs(args.embedding_config)
    outroot = _abs(args.outdir) or (repo_root / "results")

    runs_root = outroot / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    summary_root = outroot / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    # Configs
    chunk_cfg = yaml.safe_load(chunk_cfg_path.read_text(encoding="utf-8")) or {}
    clean_cfg: dict = yaml.safe_load(clean_cfg_path.read_text(encoding="utf-8")) if clean_cfg_path else {}
    emb_cfg: dict = yaml.safe_load(emb_cfg_path.read_text(encoding="utf-8")) if emb_cfg_path else {}

    profile = profile_from_cfg(clean_cfg)
    embedder = build_embedder(emb_cfg)
    client = build_ollama_client_from_env()

    # Chunking strategy reuse (same helper as run_retrieval_eval)
    from .run_retrieval_eval import build_chunker  # local import to avoid circular

    chunker = build_chunker(chunk_cfg)

    pages = load_pages_jsonl(pages_path)
    pages = [Page(page=p.page, text=apply_cleaning(p.text, profile)) for p in pages]

    chunks = chunker.chunk(pages)
    if profile.drop_reference_like_chunks or profile.drop_abbreviation_like_chunks or profile.drop_low_signal_chunks:
        chunks = [c for c in chunks if not is_noise_chunk(c.text, profile)]

    # Build retrieval index
    doc_texts = [normalize_text(c.text) for c in chunks]
    doc_matrix = embedder.fit_transform(doc_texts)
    retrieve = _pick_retriever(doc_matrix)

    questions = load_retrieval_questions(questions_path)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"prompteng_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Audit: copy configs
    (run_dir / "chunking_config.yaml").write_text(chunk_cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "cleaning_config.yaml").write_text(
        (clean_cfg_path.read_text(encoding="utf-8") if clean_cfg_path else "{}\n"),
        encoding="utf-8",
    )
    (run_dir / "embedding_config.yaml").write_text(
        (emb_cfg_path.read_text(encoding="utf-8") if emb_cfg_path else "{}\n"),
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "ollama_base_url": client.base_url,
                "ollama_model": client.model,
                "k": int(args.k),
                "num_chunks": len(chunks),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    variants = _templates()
    if args.variant != "all":
        variants = [t for t in variants if t.key == args.variant]
        if not variants:
            raise SystemExit(f"Unknown variant: {args.variant}")

    leaderboard_path = summary_root / "leaderboard_prompt_eng.csv"
    write_header = not leaderboard_path.exists()

    all_rows: list[dict[str, Any]] = []
    for templ in variants:
        perq_path = run_dir / f"per_question_{templ.key}.jsonl"
        lengths: list[int] = []
        nonempty = 0
        has_cites = 0
        hallucinated = 0
        expected_hit = 0

        with perq_path.open("w", encoding="utf-8") as f:
            for q in questions:
                q_vec = embedder.transform([q.question])
                results = retrieve(q_vec, doc_matrix, k=int(args.k))

                retrieved: list[dict[str, Any]] = []
                context_pages: set[int] = set()
                for rank, r in enumerate(results, start=1):
                    c = chunks[r.idx]
                    retrieved.append(
                        {
                            "rank": rank,
                            "score": float(r.score),
                            "chunk_id": c.chunk_id,
                            "page_start": c.page_start,
                            "page_end": c.page_end,
                            "text": c.text,
                        }
                    )
                    context_pages.update(range(int(c.page_start), int(c.page_end) + 1))

                prompt = templ.prompt.format(question=q.question, context=_build_context_snippet(retrieved))
                answer = (client.generate(prompt) or "").strip()
                toks = len(simple_tokenize(answer))
                lengths.append(toks)
                if not _is_not_found(answer):
                    nonempty += 1

                cited_pages = _extract_cited_pages(answer)
                if cited_pages:
                    has_cites += 1
                    if any((p not in context_pages) for p in cited_pages):
                        hallucinated += 1

                if q.expected_pages and cited_pages:
                    if any((p in set(q.expected_pages)) for p in cited_pages):
                        expected_hit += 1

                f.write(
                    json.dumps(
                        {
                            "id": q.id,
                            "question": q.question,
                            "expected_pages": q.expected_pages,
                            "variant": templ.key,
                            "answer": answer,
                            "answer_tokens": toks,
                            "cited_pages": cited_pages,
                            "retrieved": [
                                {
                                    "rank": r["rank"],
                                    "score": r["score"],
                                    "chunk_id": r["chunk_id"],
                                    "page_start": r["page_start"],
                                    "page_end": r["page_end"],
                                }
                                for r in retrieved
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        n = max(1, len(questions))
        with_expected = sum(1 for q in questions if q.expected_pages)
        row = {
            "run_id": run_id,
            "variant": templ.key,
            "ollama_model": client.model,
            "k": int(args.k),
            "num_questions": len(questions),
            "answer_nonempty_rate": nonempty / n,
            "cites_rate": has_cites / n,
            "citation_hallucination_rate": hallucinated / max(1, has_cites),
            "expected_page_hit_rate": expected_hit / max(1, with_expected),
            "length_p50_tokens": _p50([float(x) for x in lengths]),
            "length_p95_tokens": _p95([float(x) for x in lengths]),
            "run_dir": str(run_dir),
        }
        all_rows.append(row)

        (run_dir / f"summary_{templ.key}.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")

    with leaderboard_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else ["run_id", "variant"])
        if write_header:
            w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"Repo root: {repo_root}")
    print(f"Run directory: {run_dir}")
    print(f"Leaderboard: {leaderboard_path}")


if __name__ == "__main__":
    main()
