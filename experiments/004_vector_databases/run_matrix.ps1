param(
    [string]$Pages = "",
    [string]$Questions = "",
    [string]$ChunkingConfig = "configs\chunking\naive_300_100.yaml",
    [string]$CleaningConfig = "configs\cleaning\deep.yaml",
    [string]$EmbeddingConfig = "configs\embeddings\bge_base.yaml",
    [int]$K = 5,

    # Comma-separated list: faiss,sqlite,qdrant
    [string]$Backends = "faiss,sqlite,qdrant",

    [string]$QdrantHost = "localhost",
    [int]$QdrantPort = 6333,

    # Install optional deps for these backends
    [switch]$InstallVectorDeps
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$PY = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (!(Test-Path $PY)) {
    throw "Venv python not found at: $PY. Are you sure .venv exists?"
}

if ([string]::IsNullOrWhiteSpace($Pages)) {
    $Pages = Join-Path $RepoRoot "data\processed\guidelines_pages.jsonl"
}
if ([string]::IsNullOrWhiteSpace($Questions)) {
    $Questions = Join-Path $RepoRoot "data\eval\retrieval_questions.jsonl"
}

$ChunkingConfig = Join-Path $RepoRoot $ChunkingConfig
$CleaningConfig = Join-Path $RepoRoot $CleaningConfig
$EmbeddingConfig = Join-Path $RepoRoot $EmbeddingConfig

if (!(Test-Path $Pages)) { throw "Pages not found: $Pages" }
if (!(Test-Path $Questions)) { throw "Questions not found: $Questions" }
if (!(Test-Path $ChunkingConfig)) { throw "Chunking config not found: $ChunkingConfig" }
if (!(Test-Path $CleaningConfig)) { throw "Cleaning config not found: $CleaningConfig" }
if (!(Test-Path $EmbeddingConfig)) { throw "Embedding config not found: $EmbeddingConfig" }

$env:PYTHONPATH = (Join-Path $RepoRoot "src")

Write-Host "" 
Write-Host "== Vector DB benchmark ==" -ForegroundColor Cyan
Write-Host ("RepoRoot:   {0}" -f $RepoRoot)
Write-Host ("Pages:      {0}" -f $Pages)
Write-Host ("Questions:  {0}" -f $Questions)
Write-Host ("Chunking:   {0}" -f $ChunkingConfig)
Write-Host ("Cleaning:   {0}" -f $CleaningConfig)
Write-Host ("Embedding:  {0}" -f $EmbeddingConfig)
Write-Host ("Backends:   {0}" -f $Backends)
Write-Host ("K:          {0}" -f $K)
Write-Host ""

if ($InstallVectorDeps) {
    Write-Host "Installing vector backend deps into venv..." -ForegroundColor Yellow
    & $PY -m pip install -r (Join-Path $RepoRoot "requirements-vector.txt")
    if ($LASTEXITCODE -ne 0) { throw "pip install failed (requirements-vector.txt)" }
    Write-Host "Done installing vector deps." -ForegroundColor Green
    Write-Host ""
}

& $PY -m docproc.eval.run_vector_db_bench `
    --pages $Pages `
    --questions $Questions `
    --chunking-config $ChunkingConfig `
    --cleaning-config $CleaningConfig `
    --embedding-config $EmbeddingConfig `
    --k $K `
    --backends $Backends `
    --qdrant-host $QdrantHost `
    --qdrant-port $QdrantPort `
    --outdir "results"

if ($LASTEXITCODE -ne 0) { throw "Vector DB benchmark failed (exit code $LASTEXITCODE)" }

Write-Host "Done. Check results/vector_db/<run_id>/leaderboard_vector_db.csv" -ForegroundColor Green
