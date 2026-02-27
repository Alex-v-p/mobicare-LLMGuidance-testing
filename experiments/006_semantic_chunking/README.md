# Experiment 006 - Semantic chunking

This experiment evaluates **embedding-based semantic chunking** (sentence-level boundary detection) using the same retrieval question set as the other experiments.

Outputs:
- Run artifacts are saved to `results/runs/semantic_<timestamp>/`
- Summary CSV is appended at `results/summary/semantic_chunking.csv`

## What to run

### Quick run (default configs)

Run a single semantic chunking config with default embedding/cleaning behavior:

```powershell
./experiments/006_semantic_chunking/run_matrix.ps1
```

### Full matrix (cleaning x embedder x semantic config)

```powershell
./experiments/006_semantic_chunking/run_matrix_clean_embed.ps1
```


./experiments/006_semantic_chunking/run_matrix_clean_embed.ps1 -EmbeddingConfig "bge_large.yaml"
