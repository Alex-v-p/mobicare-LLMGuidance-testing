$ErrorActionPreference = "Stop"

# Repo root is 2 levels up from this script:
# experiments/007_llm_boundary_summary -> experiments -> repo root
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")

$PY = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (!(Test-Path $PY)) {
    throw "Venv python not found at: $PY. Are you sure .venv exists?"
}

$env:PYTHONPATH = (Join-Path $RepoRoot "src")

$configDir = Join-Path $RepoRoot "configs\chunking"
$configs = Get-ChildItem -Path $configDir -Filter "llm_boundary*.yaml"

if ($configs.Count -eq 0) {
    throw "No llm boundary configs found in: $configDir (expected llm_boundary*.yaml)"
}

foreach ($config in $configs) {
    Write-Host "Running LLM boundary config: $($config.Name)"

    & $PY -m docproc.eval.run_llm_boundary_summary_eval `
        --pages (Join-Path $RepoRoot "data\processed\guidelines_pages.jsonl") `
        --questions (Join-Path $RepoRoot "data\eval\retrieval_questions.jsonl") `
        --chunking-config $config.FullName `
        --k 5
}
