# 004 Vector database benchmarking

Goal: compare vector stores on **speed**, **memory footprint**, **disk / persistence behavior**, and a few practical metrics that matter for RAG.

We measure per backend:

- **Ingest time**: time to create index + upsert all vectors.
- **Query latency**: p50/p95 over the evaluation questions.
- **Throughput**: queries/sec.
- **Recall proxy**: top-k agreement vs a numpy dot-product baseline (optional sanity check).
- **Index size on disk**: after persist.
- **Cold start**: load time + first-query latency.
- **Stability**: can the index be reopened and return the same results.
- **Server resource usage (docker backends)**: peak RSS and CPU% sampled via `docker stats`.

Backends included:

- FAISS (local, file persistence)
- SQLite (simple baseline; stores vectors, brute-force search)
- Qdrant (docker server)

Optional extras (easy to add later): LanceDB, pgvector, Milvus.




cd .\experiments\004_vector_databases
docker compose up -d 


