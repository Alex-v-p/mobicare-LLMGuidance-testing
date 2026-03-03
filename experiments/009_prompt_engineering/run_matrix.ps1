$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$PY = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (!(Test-Path $PY)) {
    throw "Venv python not found at: $PY. Are you sure .venv exists?"
}

$env:PYTHONPATH = (Join-Path $RepoRoot "src")

# Use your standard assets. These files are already referenced by other experiments.
$Pages = Join-Path $RepoRoot "data\processed\guidelines_pages.jsonl"
$Questions = Join-Path $RepoRoot "data\eval\retrieval_questions.jsonl"

$Chunk = Join-Path $RepoRoot "configs\chunking\naive_300_50.yaml"
$Clean = Join-Path $RepoRoot "configs\cleaning\deep.yaml"
$Embed = Join-Path $RepoRoot "configs\embeddings\bge_small.yaml"

& $PY -m docproc.eval.run_prompt_engineering_eval `
    --pages $Pages `
    --questions $Questions `
    --chunking-config $Chunk `
    --cleaning-config $Clean `
    --embedding-config $Embed `
    --k 5
