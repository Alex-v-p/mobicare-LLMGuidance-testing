param(
    # If not provided, defaults to bge_large.yaml
    [string] $EmbeddingConfig = "bge_large.yaml"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$PY = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (!(Test-Path $PY)) {
    throw "Venv python not found at: $PY. Are you sure .venv exists?"
}

$env:PYTHONPATH = (Join-Path $RepoRoot "src")

$chunkConfigs = Get-ChildItem -Path (Join-Path $RepoRoot "configs\chunking") -Filter "semantic*.yaml"
$cleanConfigs = Get-ChildItem -Path (Join-Path $RepoRoot "configs\cleaning") -Filter "*.yaml"

# Resolve embedding config path:
$embedPath = $EmbeddingConfig
if (-not (Test-Path $embedPath)) {
    $embedPath = Join-Path $RepoRoot ("configs\embeddings\" + $EmbeddingConfig)
}
if (!(Test-Path $embedPath)) {
    throw "Embedding config not found. Tried: '$EmbeddingConfig' and '$embedPath'"
}

if ($chunkConfigs.Count -eq 0) {
    throw "No semantic chunking configs found. Expected configs/chunking/semantic*.yaml"
}

foreach ($clean in $cleanConfigs) {
    foreach ($chunk in $chunkConfigs) {
        Write-Host "Running: clean=$($clean.Name) embed=$(Split-Path $embedPath -Leaf) chunk=$($chunk.Name)"

        & $PY -m docproc.eval.run_semantic_chunking_eval `
            --pages (Join-Path $RepoRoot "data\processed\guidelines_pages.jsonl") `
            --questions (Join-Path $RepoRoot "data\eval\retrieval_questions.jsonl") `
            --chunking-config $chunk.FullName `
            --cleaning-config $clean.FullName `
            --embedding-config $embedPath `
            --k 5
    }
}