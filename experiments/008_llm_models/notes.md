# Experiment 005 — Ollama LLM Model Benchmark

Benchmarks local Ollama models (CPU vs GPU) on a small fixed prompt set.

./experiments/008_llm_models/run_matrix.ps1

## What it logs
For each model:
- `avg_total_s` and `p50_total_s`
- token counts (`avg_prompt_tokens`, `avg_out_tokens`)
- generation throughput (`avg_out_tokens_per_s`)
- a best-effort `processor` label (`GPU` / `CPU` / `unknown`) from `ollama ps`

Raw per-prompt results are stored in `experiments/008_llm_models/llmbench_<run_id>/...`.

## Run
From the repo root (PowerShell):

```powershell
./experiments/008_llm_models/run_matrix.ps1 -Models "llama3.1:8b,qwen2.5:7b" -Repeats 3 -Warmup 1
```

Or rely on your `.env` / environment variable `OLLAMA_MODEL`:

```powershell
$env:OLLAMA_MODEL = "llama3.1:8b"
./experiments/008_llm_models/run_matrix.ps1
```

## GPU usage
Ollama decides CPU vs GPU automatically (based on your install and available VRAM).
This experiment *records* where the model is running by parsing `ollama ps` after warmup.

If you expect GPU but see `CPU`:
- Ensure you're using a CUDA-capable Ollama build on Windows
- Make sure the model fits in VRAM (or reduce context / choose a smaller or more-quantized model)
- Verify with `ollama ps` while a request is running


## Selecting models (no manual typing)

You have three options (in priority order):

1) Pass `-Models "modelA,modelB"` to the script.
2) Put one model name per line in `experiments/008_llm_models/models.txt` (lines starting with `#` are ignored).
3) Leave both empty and it will auto-discover installed models using `ollama list`.
