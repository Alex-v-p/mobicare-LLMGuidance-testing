param(
    [string]$Pages = "",
    [string]$Questions = "",
    [string]$ChunkingConfig = "configs\chunking\naive_300_100.yaml",
    [string]$CleaningConfig = "configs\cleaning\deep.yaml",
    [int]$K = 5,

    # Filter embedding YAMLs by wildcard, e.g. "*large*" or "bge_*"
    [string]$OnlyEmbeddings = "",

    # Keep running even if one embedding config fails
    [switch]$ContinueOnError,

    # Install optional dependencies (e.g. einops for nomic)
    [switch]$InstallDeps,

    # Pre-download all models (so eval timings aren't dominated by downloads)
    [switch]$PreDownloadModels
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$PY = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (!(Test-Path $PY)) {
    throw "Venv python not found at: $PY. Are you sure .venv exists?"
}

# Defaults if not provided
if ([string]::IsNullOrWhiteSpace($Pages)) {
    $Pages = Join-Path $RepoRoot "data\processed\guidelines_pages.jsonl"
}
if ([string]::IsNullOrWhiteSpace($Questions)) {
    $Questions = Join-Path $RepoRoot "data\eval\retrieval_questions.jsonl"
}

$ChunkingConfig = Join-Path $RepoRoot $ChunkingConfig
$CleaningConfig = Join-Path $RepoRoot $CleaningConfig

if (!(Test-Path $Pages)) { throw "Pages not found: $Pages" }
if (!(Test-Path $Questions)) { throw "Questions not found: $Questions" }
if (!(Test-Path $ChunkingConfig)) { throw "Chunking config not found: $ChunkingConfig" }
if (!(Test-Path $CleaningConfig)) { throw "Cleaning config not found: $CleaningConfig" }

$env:PYTHONPATH = (Join-Path $RepoRoot "src")

Write-Host ""
Write-Host "== Embedding model matrix ==" -ForegroundColor Cyan
Write-Host ("RepoRoot:   {0}" -f $RepoRoot)
Write-Host ("Pages:      {0}" -f $Pages)
Write-Host ("Questions:  {0}" -f $Questions)
Write-Host ("Chunking:   {0}" -f $ChunkingConfig)
Write-Host ("Cleaning:   {0}" -f $CleaningConfig)
Write-Host ("K:          {0}" -f $K)
Write-Host ""

if ($InstallDeps) {
    Write-Host "Installing optional deps into venv..." -ForegroundColor Yellow
    & $PY -m pip install -q einops
    if ($LASTEXITCODE -ne 0) { throw "pip install failed (einops)" }
    Write-Host "Done installing deps." -ForegroundColor Green
    Write-Host ""
}

$EmbDir = Join-Path $RepoRoot "configs\embeddings"
if (!(Test-Path $EmbDir)) { throw "Embeddings dir not found: $EmbDir" }

$embedConfigs = Get-ChildItem -Path $EmbDir -Filter "*.yaml" | Sort-Object Name
if (![string]::IsNullOrWhiteSpace($OnlyEmbeddings)) {
    $embedConfigs = $embedConfigs | Where-Object { $_.Name -like $OnlyEmbeddings }
}

if ($embedConfigs.Count -eq 0) {
    throw "No embedding configs matched. OnlyEmbeddings='$OnlyEmbeddings'"
}

Write-Host "Embeddings to run:" -ForegroundColor Cyan
$embedConfigs | ForEach-Object { Write-Host (" - {0}" -f $_.Name) }
Write-Host ""

if ($PreDownloadModels) {
    Write-Host "Pre-downloading models (best-effort)..." -ForegroundColor Yellow
    foreach ($emb in $embedConfigs) {
        Write-Host ("Warmup load: {0}" -f $emb.Name)
        & $PY -m docproc.eval.run_embedding_eval `
            --pages $Pages `
            --questions $Questions `
            --chunking-config $ChunkingConfig `
            --cleaning-config $CleaningConfig `
            --embedding-config $emb.FullName `
            --k 1 `
            --outdir "results\runs" | Out-Null

        # ignore warmup failures here; real run will handle it
    }
    Write-Host "Pre-download pass complete." -ForegroundColor Green
    Write-Host ""
}

$failed = @()

foreach ($emb in $embedConfigs) {
    Write-Host "---- Running embedding eval ----" -ForegroundColor Cyan
    Write-Host ("Embed:   {0}" -f (Resolve-Path $emb.FullName))
    Write-Host ("Chunk:   {0}" -f (Resolve-Path $ChunkingConfig))
    Write-Host ("Clean:   {0}" -f (Resolve-Path $CleaningConfig))
    Write-Host "------------------------------"

    & $PY -m docproc.eval.run_embedding_eval `
        --pages $Pages `
        --questions $Questions `
        --chunking-config $ChunkingConfig `
        --cleaning-config $CleaningConfig `
        --embedding-config $emb.FullName `
        --k $K `
        --outdir "results\runs"

    if ($LASTEXITCODE -ne 0) {
        $msg = "Embedding eval failed for: $($emb.Name) (exit code $LASTEXITCODE)"
        if ($ContinueOnError) {
            Write-Host $msg -ForegroundColor Red
            $failed += $emb.Name
            Write-Host ""
            continue
        } else {
            throw $msg
        }
    }

    Write-Host ""
}

Write-Host "Done. Check results/summary/leaderboard_embeddings.csv" -ForegroundColor Green
if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "Some configs failed:" -ForegroundColor Yellow
    $failed | ForEach-Object { Write-Host (" - {0}" -f $_) }
}