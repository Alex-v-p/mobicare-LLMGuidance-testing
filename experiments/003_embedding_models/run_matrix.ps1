param(
    [string]$ChunkingConfig = "configs\chunking\naive_300_100.yaml",
    [string]$CleaningConfig = "configs\cleaning\deep.yaml",
    [int]$K = 5
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$PY = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (!(Test-Path $PY)) {
    throw "Venv python not found at: $PY. Are you sure .venv exists?"
}

$env:PYTHONPATH = (Join-Path $RepoRoot "src")

# Silence the Windows symlink warning spam from HF cache
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$PagesPath     = Join-Path $RepoRoot "data\processed\guidelines_pages.jsonl"
$QuestionsPath = Join-Path $RepoRoot "data\eval\retrieval_questions.jsonl"

if (!(Test-Path $PagesPath))     { throw "Pages file not found: $PagesPath" }
if (!(Test-Path $QuestionsPath)) { throw "Questions file not found: $QuestionsPath" }

$ChunkingConfigPath = Join-Path $RepoRoot $ChunkingConfig
$CleaningConfigPath = Join-Path $RepoRoot $CleaningConfig

if (!(Test-Path $ChunkingConfigPath)) { throw "Chunking config not found: $ChunkingConfigPath" }
if (!(Test-Path $CleaningConfigPath)) { throw "Cleaning config not found: $CleaningConfigPath" }

$embedConfigs = @(
    "configs\embeddings\all_minilm.yaml",
    "configs\embeddings\bge_small.yaml",
    "configs\embeddings\nomic_embed_text.yaml"
)

foreach ($rel in $embedConfigs) {
    $embedPath = Join-Path $RepoRoot $rel
    if (!(Test-Path $embedPath)) {
        throw "Embedding config not found: $embedPath"
    }

    Write-Host ""
    Write-Host "---- Running embedding eval ----"
    Write-Host "Embed:   $rel"
    Write-Host "Chunk:   $ChunkingConfig"
    Write-Host "Clean:   $CleaningConfig"
    Write-Host "------------------------------"

    & $PY -m docproc.eval.run_embedding_eval `
        --pages $PagesPath `
        --questions $QuestionsPath `
        --chunking-config $ChunkingConfigPath `
        --cleaning-config $CleaningConfigPath `
        --embedding-config $embedPath `
        --k $K

    if ($LASTEXITCODE -ne 0) {
        throw "Embedding eval failed for: $rel (exit code $LASTEXITCODE)"
    }
}

Write-Host ""
Write-Host "✅ Done. Check results/summary/leaderboard_embeddings.csv"
