from __future__ import annotations

"""Vector database benchmark runner.

Benchmarks are intentionally *practical* for a small-document RAG setup.

Measures (per backend):
  - ingest_time_s
  - query_p50_ms / query_p95_ms / qps
  - persist_time_s (best-effort)
  - reload_time_s
  - disk_size_mb
  - agreement_at_k vs numpy baseline (sanity check)

Backends:
  - faiss (local)
  - chroma (local persistent)
  - sqlite (simple baseline; stores vectors and brute-force searches)
  - qdrant (server; use docker-compose in experiments/004_vector_databases)
"""

import argparse
import csv
import json
import os
import shutil
import sqlite3
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from ..chunkers.base import Page
from ..chunkers.naive_tokens import NaiveTokenChunker
from ..chunkers.structured import StructuredHeadingChunker
from ..chunkers.tree_parent_child import TreeParentChildChunker
from ..clean import normalize_text
from ..clean_profiles import apply_cleaning, is_noise_chunk, profile_from_cfg
from ..index.embed_factory import build_embedder
from .datasets import load_retrieval_questions


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


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


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / denom


def _topk_dot(q: np.ndarray, mat: np.ndarray, k: int) -> list[int]:
    # q is (d,) or (1,d)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    scores = (mat @ q.T).reshape(-1)
    idx = np.argsort(-scores)[: min(k, len(scores))]
    return [int(i) for i in idx]


def _pctl(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    # simple linear interpolation
    k = (len(values) - 1) * (p / 100.0)
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return float(values[f])
    return float(values[f] + (values[c] - values[f]) * (k - f))


@dataclass
class BenchResult:
    backend: str
    n_vectors: int
    dim: int
    ingest_time_s: float
    query_p50_ms: float
    query_p95_ms: float
    qps: float
    persist_time_s: float
    reload_time_s: float
    first_query_ms_after_reload: float
    disk_size_mb: float
    rss_before_mb: float
    rss_after_mb: float
    agreement_at_k: float


class _FaissBackend:
    name = "faiss"

    def __init__(self, workdir: Path, dim: int):
        import faiss  # type: ignore

        self.faiss = faiss
        self.workdir = workdir
        self.dim = dim
        self.index_path = workdir / "faiss.index"
        self.index = faiss.IndexFlatIP(dim)

    def ingest(self, vecs: np.ndarray, ids: list[int]):
        # IndexFlatIP ignores ids; we keep implicit order.
        self.index.add(vecs.astype(np.float32, copy=False))

    def query(self, q: np.ndarray, k: int) -> list[int]:
        D, I = self.index.search(q.astype(np.float32, copy=False).reshape(1, -1), k)
        return [int(i) for i in I[0] if int(i) != -1]

    def persist(self) -> None:
        self.faiss.write_index(self.index, str(self.index_path))

    def reload(self) -> None:
        self.index = self.faiss.read_index(str(self.index_path))


class _SqliteBackend:
    """A simple baseline: vectors stored in SQLite, brute-force search in numpy.

This is *not* a true ANN index. It's useful as a correctness + persistence baseline.
"""

    name = "sqlite"

    def __init__(self, workdir: Path, dim: int):
        self.workdir = workdir
        self.dim = dim
        self.db_path = workdir / "vectors.sqlite"
        if self.db_path.exists():
            self.db_path.unlink()
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("CREATE TABLE IF NOT EXISTS vectors (id INTEGER PRIMARY KEY, vec BLOB)")
        self.conn.commit()
        self._cache: tuple[list[int], np.ndarray] | None = None

    def ingest(self, vecs: np.ndarray, ids: list[int]):
        # Store as float32 bytes
        rows = [(int(i), vecs[j].astype(np.float32).tobytes()) for j, i in enumerate(ids)]
        self.conn.executemany("INSERT INTO vectors(id, vec) VALUES (?, ?)", rows)
        self.conn.commit()
        self._cache = None

    def _load_cache(self) -> tuple[list[int], np.ndarray]:
        if self._cache is not None:
            return self._cache
        cur = self.conn.execute("SELECT id, vec FROM vectors ORDER BY id")
        ids: list[int] = []
        vec_list: list[np.ndarray] = []
        for i, blob in cur.fetchall():
            ids.append(int(i))
            vec = np.frombuffer(blob, dtype=np.float32)
            vec_list.append(vec)
        mat = np.vstack(vec_list) if vec_list else np.zeros((0, self.dim), dtype=np.float32)
        self._cache = (ids, mat)
        return self._cache

    def query(self, q: np.ndarray, k: int) -> list[int]:
        ids, mat = self._load_cache()
        if mat.size == 0:
            return []
        idx = _topk_dot(q.astype(np.float32), mat, k)
        return [ids[i] for i in idx]

    def persist(self) -> None:
        self.conn.commit()

    def reload(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
        self.conn = sqlite3.connect(self.db_path)
        self._cache = None


class _QdrantBackend:
    name = "qdrant"

    def __init__(self, workdir: Path, dim: int, host: str, port: int):
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.http import models  # type: ignore

        self.models = models
        self.workdir = workdir
        self.dim = dim
        self._host = host
        self._port = port
        self.client = QdrantClient(host=host, port=port)
        self.collection = "bench"

        # Recreate collection
        try:
            self.client.delete_collection(self.collection)
        except Exception:
            pass
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

    def ingest(self, vecs: np.ndarray, ids: list[int]):
        points = []
        for j, i in enumerate(ids):
            points.append(self.models.PointStruct(id=int(i), vector=vecs[j].astype(np.float32).tolist(), payload=None))
        self.client.upsert(collection_name=self.collection, points=points)

    def query(self, q: np.ndarray, k: int) -> list[int]:
        qvec = q.astype(np.float32).tolist()

        # qdrant-client < removal: had .search()
        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=qvec,
                limit=k,
            )
            return [int(h.id) for h in hits]

        # qdrant-client new API: use .query_points()
        res = self.client.query_points(
            collection_name=self.collection,
            query=qvec,
            limit=k,
        )
        # res is typically QueryResponse(points=[ScoredPoint...])
        points = getattr(res, "points", res)
        return [int(p.id) for p in points]

    def persist(self) -> None:
        # Qdrant persists server-side; force a call.
        _ = self.client.get_collection(self.collection)

    def reload(self) -> None:
        # Server stays up; "reload" for us means reconnect.
        from qdrant_client import QdrantClient  # type: ignore

        self.client = QdrantClient(host=self._host, port=self._port)


def _build_backend(name: str, *, workdir: Path, dim: int, qdrant_host: str, qdrant_port: int):
    name = name.lower().strip()
    if name == "faiss":
        return _FaissBackend(workdir, dim)
    if name == "sqlite":
        return _SqliteBackend(workdir, dim)
    if name == "qdrant":
        return _QdrantBackend(workdir, dim, host=qdrant_host, port=qdrant_port)
    raise ValueError(f"Unknown backend: {name}")


def _agreement_at_k(baseline: list[int], got: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    b = set(baseline[:k])
    g = set(got[:k])
    return len(b & g) / float(k)


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark vector stores (speed, persistence, sanity)")
    p.add_argument("--pages", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--chunking-config", required=True)
    p.add_argument("--cleaning-config", required=False)
    p.add_argument("--embedding-config", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--backends", default="faiss,chroma,sqlite,qdrant")
    p.add_argument("--outdir", default="results")
    p.add_argument("--qdrant-host", default="localhost")
    p.add_argument("--qdrant-port", type=int, default=6333)
    args = p.parse_args()

    pages_path = Path(args.pages)
    questions_path = Path(args.questions)
    chunk_cfg = yaml.safe_load(Path(args.chunking_config).read_text(encoding="utf-8")) or {}
    clean_cfg: dict = {}
    if args.cleaning_config:
        clean_cfg = yaml.safe_load(Path(args.cleaning_config).read_text(encoding="utf-8")) or {}

    profile = profile_from_cfg(clean_cfg)
    chunker = build_chunker(chunk_cfg)
    emb_cfg: dict = yaml.safe_load(Path(args.embedding_config).read_text(encoding="utf-8")) or {}
    embedder = build_embedder(emb_cfg)

    query_prefix = str(emb_cfg.get("query_prefix", ""))
    passage_prefix = str(emb_cfg.get("passage_prefix", ""))

    def _with_prefix(prefix: str, text: str) -> str:
        if not prefix:
            return text
        if text.lower().startswith(prefix.lower()):
            return text
        return f"{prefix}{text}"

    # Load + clean pages
    pages = load_pages_jsonl(pages_path)
    pages = [Page(page=p.page, text=apply_cleaning(p.text, profile)) for p in pages]

    # Chunk
    strategy = chunk_cfg.get("strategy")
    if strategy == "tree_parent_child":
        parents, children = chunker.chunk(pages)
        if profile.drop_reference_like_chunks or profile.drop_abbreviation_like_chunks or profile.drop_low_signal_chunks:
            parents = [c for c in parents if not is_noise_chunk(c.text, profile)]
            children = [c for c in children if not is_noise_chunk(c.text, profile)]
        chunks = children
    else:
        chunks = chunker.chunk(pages)
        if profile.drop_reference_like_chunks or profile.drop_abbreviation_like_chunks or profile.drop_low_signal_chunks:
            chunks = [c for c in chunks if not is_noise_chunk(c.text, profile)]

    doc_texts = [_with_prefix(passage_prefix, normalize_text(c.text)) for c in chunks]
    doc_ids = list(range(len(doc_texts)))

    # Embed docs + queries
    doc_vecs = embedder.fit_transform(doc_texts)
    doc_vecs = np.asarray(doc_vecs, dtype=np.float32)
    doc_vecs = _l2_normalize(doc_vecs)
    dim = int(doc_vecs.shape[1]) if doc_vecs.ndim == 2 else 0

    questions = load_retrieval_questions(questions_path)
    queries = [_with_prefix(query_prefix, q.question) for q in questions]
    query_vecs = embedder.transform(queries)
    query_vecs = np.asarray(query_vecs, dtype=np.float32)
    query_vecs = _l2_normalize(query_vecs)

    # Baseline for agreement
    baseline_topk = [_topk_dot(query_vecs[i], doc_vecs, args.k) for i in range(len(query_vecs))]

    root = Path(args.outdir)

    def _rel_to_repo(p: str | None) -> str:
        if not p:
            return ""
        try:
            return str(Path(p).resolve().relative_to(Path.cwd().resolve()))
        except Exception:
            return Path(p).name  # fallback to filename only
              
    runs_dir = root / "runs"             
    summary_dir = root / "summary"        
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"vecdb_{run_id}"  
    run_dir.mkdir(parents=True, exist_ok=True)

    backends = [b.strip() for b in str(args.backends).split(",") if b.strip()]
    results: list[BenchResult] = []

    for b in backends:
        work = run_dir / b
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)

        rss_before = _rss_mb()

        # Build backend
        backend = _build_backend(b, workdir=work, dim=dim, qdrant_host=args.qdrant_host, qdrant_port=args.qdrant_port)

        # Ingest
        t0 = time.perf_counter()
        backend.ingest(doc_vecs, doc_ids)
        ingest_time = time.perf_counter() - t0

        # Query benchmark
        lat_ms: list[float] = []
        agree: list[float] = []
        t1 = time.perf_counter()
        for i in range(len(query_vecs)):
            qv = query_vecs[i]
            s = time.perf_counter()
            got = backend.query(qv, args.k)
            e = time.perf_counter()
            lat_ms.append((e - s) * 1000.0)
            agree.append(_agreement_at_k(baseline_topk[i], got, args.k))
        total_q = time.perf_counter() - t1
        qps = (len(query_vecs) / total_q) if total_q > 0 else 0.0

        # Persist + reload
        t2 = time.perf_counter()
        try:
            backend.persist()
            persist_time = time.perf_counter() - t2
        except Exception:
            persist_time = -1.0

        disk_mb = _dir_size_mb(work)

        t3 = time.perf_counter()
        try:
            backend.reload()
            reload_time = time.perf_counter() - t3
        except Exception:
            reload_time = -1.0

        # First query after reload (cold-ish)
        try:
            s = time.perf_counter()
            _ = backend.query(query_vecs[0], args.k)
            first_after_reload = (time.perf_counter() - s) * 1000.0
        except Exception:
            first_after_reload = -1.0

        rss_after = _rss_mb()

        results.append(
            BenchResult(
                backend=b,
                n_vectors=int(doc_vecs.shape[0]),
                dim=dim,
                ingest_time_s=float(ingest_time),
                query_p50_ms=float(_pctl(lat_ms, 50)),
                query_p95_ms=float(_pctl(lat_ms, 95)),
                qps=float(qps),
                persist_time_s=float(persist_time),
                reload_time_s=float(reload_time),
                first_query_ms_after_reload=float(first_after_reload),
                disk_size_mb=float(disk_mb),
                rss_before_mb=float(rss_before),
                rss_after_mb=float(rss_after),
                agreement_at_k=float(statistics.fmean(agree) if agree else 0.0),
            )
        )

    # Write CSV summary
    out_csv = run_dir / "leaderboard_vector_db.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["run_id"] + list(BenchResult.__annotations__.keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({"run_id": run_id, **r.__dict__})

    summary_csv = summary_dir / "leaderboard_vecdb.csv"
    write_header = not summary_csv.exists()

    meta = {
        "run_id": run_id,
        "embedding_config": _rel_to_repo(args.embedding_config),
        "chunking_config": _rel_to_repo(args.chunking_config),
        "cleaning_config": _rel_to_repo(args.cleaning_config),
        "k": int(args.k),
    }

    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        fieldnames = list(meta.keys()) + list(BenchResult.__annotations__.keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in results:
            w.writerow({**meta, **r.__dict__})

    print(f"Wrote run CSV: {out_csv}")
    print(f"Updated summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
