# Experiment 007 — LLM boundary detection + summary augmentation (Ollama)

This experiment evaluates two variants:

1. **llm_boundary_raw** — chunk boundaries chosen by an LLM (via Ollama), retrieval over raw chunk text.
2. **llm_boundary_summary_aug** — same chunks, but each chunk is augmented by a short LLM summary prepended to the text before embedding.

Outputs:
- Run artifacts: `results/runs/llm_boundary_<timestamp>/...`
- Summary CSV: `results/summary/llm_boundary_summary.csv`

## Requirements
- Local Ollama running (default: `http://localhost:11434`)
- A model pulled locally (configure in `.env`)

## Configuration
Chunking configs are in:
- `configs/chunking/llm_boundary.yaml`
- `configs/chunking/llm_boundary_summary.yaml`

Environment (optional):
Create a `.env` in the repo root:

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
OLLAMA_TIMEOUT_S=120
```

## Run
From repo root (PowerShell):

```
./experiments/007_llm_boundary_summary/run_matrix.ps1
```
