# 003 Embedding Model Comparison

This experiment compares embedding models on:

- **Embedding time** (documents + queries)
- **Retrieval accuracy** (Hit@K, MRR)
- **Memory usage** (CPU RSS + GPU peak if available)

Run the full matrix with:

```powershell
./experiments/003_embedding_models/run_matrix.ps1
```

Outputs:
- Per-run artifacts: `results/runs/embed_<timestamp>/...`
- Summary: `results/summary/leaderboard_embeddings.csv`
