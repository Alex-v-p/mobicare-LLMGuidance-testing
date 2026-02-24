$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$PY = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (!(Test-Path $PY)) {
    throw "Venv python not found at: $PY. Are you sure .venv exists?"
}

$env:PYTHONPATH = (Join-Path $RepoRoot "src")

$chunkConfigs = Get-ChildItem -Path (Join-Path $RepoRoot "configs\chunking") -Filter "*.yaml"
$cleanConfigs = Get-ChildItem -Path (Join-Path $RepoRoot "configs\cleaning") -Filter "*.yaml"
$embedConfigs = Get-ChildItem -Path (Join-Path $RepoRoot "configs\embeddings") -Filter "*.yaml"

foreach ($clean in $cleanConfigs) {
    foreach ($embed in $embedConfigs) {
        foreach ($chunk in $chunkConfigs) {
            Write-Host "Running: clean=$($clean.Name) embed=$($embed.Name) chunk=$($chunk.Name)"

            & $PY -m docproc.eval.run_retrieval_eval `
                --pages (Join-Path $RepoRoot "data\processed\guidelines_pages.jsonl") `
                --questions (Join-Path $RepoRoot "data\eval\retrieval_questions.jsonl") `
                --chunking-config $chunk.FullName `
                --cleaning-config $clean.FullName `
                --embedding-config $embed.FullName `
                --k 5
        }
    }
}