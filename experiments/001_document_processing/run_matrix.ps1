$ErrorActionPreference = "Stop"

# Repo root is 2 levels up from this script:
# experiments/001_document_processing -> experiments -> repo root
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")

$PY = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (!(Test-Path $PY)) {
    throw "Venv python not found at: $PY. Are you sure .venv exists?"
}

$env:PYTHONPATH = (Join-Path $RepoRoot "src")

$configDir = Join-Path $RepoRoot "configs\chunking"
$configs = Get-ChildItem -Path $configDir -Filter "*.yaml"

foreach ($config in $configs) {
    Write-Host "Running config: $($config.Name)"

    & $PY -m docproc.eval.run_retrieval_eval `
        --pages (Join-Path $RepoRoot "data\processed\guidelines_pages.jsonl") `
        --questions (Join-Path $RepoRoot "data\eval\retrieval_questions.jsonl") `
        --chunking-config $config.FullName `
        --k 5
}