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

from pgvector import Vector 

from concurrent.futures import ThreadPoolExecutor, as_completed
import random

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
    flood_concurrency: int
    flood_requests: int
    flood_qps: float
    flood_p50_ms: float
    flood_p95_ms: float
    flood_p99_ms: float
    flood_max_ms: float
    flood_error_rate: float


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


class _LanceDBBackend:
    name = "lancedb"

    def __init__(self, workdir: Path, dim: int):
        import lancedb  # type: ignore
        import pyarrow as pa  # type: ignore

        self.dim = dim
        self.db_path = workdir / "lancedb"
        self.db = lancedb.connect(str(self.db_path))
        self.table_name = "bench"

        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ]
        )

        # recreate table
        try:
            self.db.drop_table(self.table_name)
        except Exception:
            pass
        self.table = self.db.create_table(self.table_name, schema=schema)

    def ingest(self, vecs: np.ndarray, ids: list[int]):
        rows = [{"id": int(ids[i]), "vector": vecs[i].astype(np.float32).tolist()} for i in range(len(ids))]
        self.table.add(rows)
        # index optional; for small N it can be slower to build than query
        try:
            self.table.create_index("vector", metric="cosine")
        except Exception:
            pass

    def query(self, q: np.ndarray, k: int) -> list[int]:
        qvec = q.astype(np.float32).tolist()
        res = self.table.search(qvec).limit(k).select(["id"]).to_list()
        return [int(r["id"]) for r in res]

    def persist(self) -> None:
        # LanceDB is persistent on disk automatically
        pass

    def reload(self) -> None:
        import lancedb  # type: ignore

        self.db = lancedb.connect(str(self.db_path))
        self.table = self.db.open_table(self.table_name)

class _PgvectorBackend:
    name = "pgvector"

    def __init__(self, workdir: Path, dim: int, host: str, port: int, user: str, password: str, db: str):
        import psycopg  # type: ignore
        from pgvector.psycopg import register_vector  # type: ignore

        self.dim = dim
        self._conn = psycopg.connect(host=host, port=port, user=user, password=password, dbname=db)

        # Ensure extension exists before registering the type
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        self._conn.commit()

        register_vector(self._conn)

        self._conn.execute("DROP TABLE IF EXISTS vectors;")
        self._conn.execute(f"CREATE TABLE vectors (id BIGINT PRIMARY KEY, vec vector({dim}));")
        self._conn.commit()

    def ingest(self, vecs: np.ndarray, ids: list[int]):
        from pgvector import Vector  # type: ignore

        rows = [(int(ids[i]), Vector(vecs[i].astype(np.float32).tolist())) for i in range(len(ids))]
        with self._conn.cursor() as cur:
            cur.executemany("INSERT INTO vectors(id, vec) VALUES (%s, %s);", rows)
        self._conn.commit()

        # optional IVF index (good for larger N; for tiny N can be unnecessary)
        try:
            self._conn.execute("CREATE INDEX IF NOT EXISTS vec_ivf ON vectors USING ivfflat (vec vector_cosine_ops) WITH (lists = 100);")
            self._conn.execute("ANALYZE vectors;")
            self._conn.commit()
        except Exception:
            self._conn.rollback()

    def query(self, q: np.ndarray, k: int) -> list[int]:
        from pgvector import Vector  # type: ignore

        qvec = Vector(q.astype(np.float32).tolist())
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM vectors ORDER BY vec <=> %s LIMIT %s;",
                (qvec, int(k)),
            )
            return [int(r[0]) for r in cur.fetchall()]

    def persist(self) -> None:
        self._conn.commit()

    def reload(self) -> None:
        # server persists; reconnect for "reload"
        try:
            self._conn.close()
        except Exception:
            pass
        # reconnect details are stored in DSN inside psycopg; easiest is just keep connection open
        # so reload is a no-op if close fails
        pass

class _MilvusBackend:
    name = "milvus"

    def __init__(self, workdir: Path, dim: int, host: str, port: int):
        from pymilvus import (  # type: ignore
            connections,
            FieldSchema,
            CollectionSchema,
            DataType,
            Collection,
            utility,
        )

        self.dim = dim
        self._collection_name = "bench"

        connections.connect(alias="default", host=host, port=str(port))

        if utility.has_collection(self._collection_name):
            utility.drop_collection(self._collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="bench")
        self.col = Collection(self._collection_name, schema)

        self.col.create_index(
            field_name="vector",
            index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}},
        )

    def ingest(self, vecs: np.ndarray, ids: list[int]):
        self.col.insert([ids, vecs.astype(np.float32).tolist()])
        self.col.flush()
        self.col.load()

    def query(self, q: np.ndarray, k: int) -> list[int]:
        res = self.col.search(
            data=[q.astype(np.float32).tolist()],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=int(k),
            output_fields=["id"],
        )
        return [int(h.entity.get("id")) for h in res[0]]

    def persist(self) -> None:
        # server-side persistence
        pass

    def reload(self) -> None:
        # server stays up; reload means ensure loaded
        self.col.load()

class _WeaviateBackend:
    name = "weaviate"

    def __init__(self, workdir: Path, dim: int, url: str):
        import weaviate  # type: ignore
        from weaviate.classes.config import Configure  # type: ignore

        self.dim = dim
        self.client = weaviate.Client(url)

        self.class_name = "Bench"

        # recreate schema
        try:
            self.client.schema.delete_class(self.class_name)
        except Exception:
            pass

        self.client.schema.create_class(
            {
                "class": self.class_name,
                "vectorizer": "none",
                "properties": [{"name": "id", "dataType": ["int"]}],
                "vectorIndexConfig": {"distance": "cosine"},
            }
        )

    def ingest(self, vecs: np.ndarray, ids: list[int]):
        with self.client.batch as batch:
            batch.batch_size = 256
            for i in range(len(ids)):
                batch.add_data_object(
                    data_object={"id": int(ids[i])},
                    class_name=self.class_name,
                    vector=vecs[i].astype(np.float32).tolist(),
                )

    def query(self, q: np.ndarray, k: int) -> list[int]:
        qvec = q.astype(np.float32).tolist()
        res = (
            self.client.query.get(self.class_name, ["id"])
            .with_near_vector({"vector": qvec})
            .with_limit(int(k))
            .do()
        )
        hits = res.get("data", {}).get("Get", {}).get(self.class_name, [])
        return [int(h["id"]) for h in hits]

    def persist(self) -> None:
        pass

    def reload(self) -> None:
        # reconnect
        import weaviate  # type: ignore

        # (client is cheap; keep url via ._connection.url)
        url = getattr(getattr(self.client, "_connection", None), "url", None) or "http://localhost:8080"
        self.client = weaviate.Client(url)


def _build_backend(
    name: str,
    *,
    workdir: Path,
    dim: int,
    qdrant_host: str,
    qdrant_port: int,
    pg_host: str,
    pg_port: int,
    pg_user: str,
    pg_password: str,
    pg_db: str,
    milvus_host: str,
    milvus_port: int,
    weaviate_url: str,
):
    name = name.lower().strip()
    if name == "faiss":
        return _FaissBackend(workdir, dim)
    if name == "sqlite":
        return _SqliteBackend(workdir, dim)
    if name == "qdrant":
        return _QdrantBackend(workdir, dim, host=qdrant_host, port=qdrant_port)
    if name == "lancedb":
        return _LanceDBBackend(workdir, dim)
    if name == "pgvector":
        return _PgvectorBackend(workdir, dim, host=pg_host, port=pg_port, user=pg_user, password=pg_password, db=pg_db)
    if name == "milvus":
        return _MilvusBackend(workdir, dim, host=milvus_host, port=milvus_port)
    if name == "weaviate":
        return _WeaviateBackend(workdir, dim, url=weaviate_url)
    raise ValueError(f"Unknown backend: {name}")


def _agreement_at_k(baseline: list[int], got: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    b = set(baseline[:k])
    g = set(got[:k])
    return len(b & g) / float(k)

def _flood_test(backend, query_vecs: np.ndarray, k: int, concurrency: int, n_reqs: int, timeout_ms: int):
    lat_ms: list[float] = []
    errors = 0

    # sample queries (with replacement) to simulate real traffic
    idxs = [random.randrange(len(query_vecs)) for _ in range(n_reqs)]

    def one(i: int) -> float:
        qv = query_vecs[idxs[i]]
        t0 = time.perf_counter()
        backend.query(qv, k)
        return (time.perf_counter() - t0) * 1000.0

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(one, i): i for i in range(n_reqs)}
        for fut in as_completed(futures, timeout=max(1.0, timeout_ms / 1000.0 * n_reqs / max(1, concurrency))):
            try:
                lat_ms.append(fut.result(timeout=timeout_ms / 1000.0))
            except Exception:
                errors += 1
    elapsed = time.perf_counter() - start

    qps = (len(lat_ms) / elapsed) if elapsed > 0 else 0.0
    p50 = _pctl(lat_ms, 50)
    p95 = _pctl(lat_ms, 95)
    p99 = _pctl(lat_ms, 99)
    mx = max(lat_ms) if lat_ms else float("nan")
    err_rate = errors / float(n_reqs)

    return {
        "flood_concurrency": concurrency,
        "flood_requests": n_reqs,
        "flood_qps": qps,
        "flood_p50_ms": p50,
        "flood_p95_ms": p95,
        "flood_p99_ms": p99,
        "flood_max_ms": mx,
        "flood_error_rate": err_rate,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark vector stores (speed, persistence, sanity)")
    p.add_argument("--pages", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--chunking-config", required=True)
    p.add_argument("--cleaning-config", required=False)
    p.add_argument("--embedding-config", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--backends", default="faiss,sqlite,qdrant,lancedb,pgvector,milvus,weaviate")
    p.add_argument("--outdir", default="results")
    p.add_argument("--qdrant-host", default="localhost")
    p.add_argument("--qdrant-port", type=int, default=6333)
    p.add_argument("--pg-host", default="localhost")
    p.add_argument("--pg-port", type=int, default=5432)
    p.add_argument("--pg-user", default="vec")
    p.add_argument("--pg-password", default="vec")
    p.add_argument("--pg-db", default="vecdb")
    p.add_argument("--milvus-host", default="localhost")
    p.add_argument("--milvus-port", type=int, default=19530)
    p.add_argument("--weaviate-url", default="http://localhost:8080")
    p.add_argument("--flood", action="store_true", help="Run flood test (concurrent burst).")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--burst-requests", type=int, default=200)
    p.add_argument("--timeout-ms", type=int, default=2000)
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
    flood_rows: list[dict] = []

    for b in backends:
        work = run_dir / b
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)

        rss_before = _rss_mb()

        # Build backend
        try:
            backend = _build_backend(
                b,
                workdir=work,
                dim=dim,
                qdrant_host=args.qdrant_host,
                qdrant_port=args.qdrant_port,
                pg_host=args.pg_host,
                pg_port=args.pg_port,
                pg_user=args.pg_user,
                pg_password=args.pg_password,
                pg_db=args.pg_db,
                milvus_host=args.milvus_host,
                milvus_port=args.milvus_port,
                weaviate_url=args.weaviate_url,
            )
        except Exception as e:
            print(f"[WARN] Skipping backend '{b}' (init failed): {e}")
            continue

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

        # Flood metrics defaults (meaning: not run)
        flood_conc = -1
        flood_reqs = -1
        flood_qps = -1.0
        flood_p50 = -1.0
        flood_p95 = -1.0
        flood_p99 = -1.0
        flood_max = -1.0
        flood_err = -1.0

        if args.flood:
            # pgvector uses psycopg connection; not thread-safe if shared.
            # Skip by default unless you later implement a connection pool.
            if b.lower() == "pgvector":
                print("[WARN] Skipping flood test for pgvector (shared psycopg connection not thread-safe).")
            else:
                fm = _flood_test(
                    backend=backend,
                    query_vecs=query_vecs,
                    k=args.k,
                    concurrency=args.concurrency,
                    n_reqs=args.burst_requests,
                    timeout_ms=args.timeout_ms,
                )
                flood_conc = int(fm["flood_concurrency"])
                flood_reqs = int(fm["flood_requests"])
                flood_qps = float(fm["flood_qps"])
                flood_p50 = float(fm["flood_p50_ms"])
                flood_p95 = float(fm["flood_p95_ms"])
                flood_p99 = float(fm["flood_p99_ms"])
                flood_max = float(fm["flood_max_ms"])
                flood_err = float(fm["flood_error_rate"])

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
                flood_concurrency=int(flood_conc),
                flood_requests=int(flood_reqs),
                flood_qps=float(flood_qps),
                flood_p50_ms=float(flood_p50),
                flood_p95_ms=float(flood_p95),
                flood_p99_ms=float(flood_p99),
                flood_max_ms=float(flood_max),
                flood_error_rate=float(flood_err),
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
