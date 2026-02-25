param(
    [switch]$OnlyHybridAndRerank,
    [switch]$IncludeOverlap300100,
    [switch]$IncludeBioEmbedding
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$PY = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (!(Test-Path $PY)) {
    throw "Venv python not found at: $PY. Are you sure .venv exists?"
}

# Make sure Python can import from src/
$env:PYTHONPATH = (Join-Path $RepoRoot "src")

# Use the same canonical dataset paths as your 001 script
$PagesPath     = Join-Path $RepoRoot "data\processed\guidelines_pages.jsonl"
$QuestionsPath = Join-Path $RepoRoot "data\eval\retrieval_questions.jsonl"

if (!(Test-Path $PagesPath))     { throw "Pages file not found: $PagesPath" }
if (!(Test-Path $QuestionsPath)) { throw "Questions file not found: $QuestionsPath" }

# Base configs
$CleaningConfig  = Join-Path $RepoRoot "configs\cleaning\deep.yaml"
$EmbeddingConfig = Join-Path $RepoRoot "configs\embeddings\bge_small.yaml"

if ($IncludeBioEmbedding) {
    $EmbeddingConfig = Join-Path $RepoRoot "configs\embeddings\bioclinical_sbert.yaml"
}

if (!(Test-Path $CleaningConfig))  { throw "Cleaning config not found: $CleaningConfig" }
if (!(Test-Path $EmbeddingConfig)) { throw "Embedding config not found: $EmbeddingConfig" }

# Chunking configs
$chunkingVariants = @()
$chunkingVariants += (Join-Path $RepoRoot "configs\chunking\naive_300_50.yaml")

if ($IncludeOverlap300100) {
    $chunkingVariants += (Join-Path $RepoRoot "configs\chunking\naive_300_100.yaml")
}

# Retrieval configs
$retrievalConfigs = @()

if ($OnlyHybridAndRerank) {
    $retrievalConfigs += (Join-Path $RepoRoot "configs\retrieval\hybrid_bm25_bge.yaml")
    $retrievalConfigs += (Join-Path $RepoRoot "configs\retrieval\dense_rerank_bge.yaml")
    $retrievalConfigs += (Join-Path $RepoRoot "configs\retrieval\hybrid_rerank_bge.yaml")
} else {
    $retrievalConfigs += (Join-Path $RepoRoot "configs\retrieval\dense.yaml")
    $retrievalConfigs += (Join-Path $RepoRoot "configs\retrieval\bm25.yaml")
    $retrievalConfigs += (Join-Path $RepoRoot "configs\retrieval\hybrid_bm25_dense.yaml")
    $retrievalConfigs += (Join-Path $RepoRoot "configs\retrieval\hybrid_bm25_bge.yaml")
    $retrievalConfigs += (Join-Path $RepoRoot "configs\retrieval\dense_rerank_bge.yaml")
    $retrievalConfigs += (Join-Path $RepoRoot "configs\retrieval\hybrid_rerank_bge.yaml")
    $retrievalConfigs += (Join-Path $RepoRoot "configs\retrieval\rewrite_hf_then_hybrid.yaml")
    $retrievalConfigs += (Join-Path $RepoRoot "configs\retrieval\parent_child_dense.yaml")
}

# Validate files exist
foreach ($p in $chunkingVariants + $retrievalConfigs) {
    if (!(Test-Path $p)) { throw "Missing config file: $p" }
}

Write-Host "== Retrieval strategy matrix =="
Write-Host "RepoRoot:   $RepoRoot"
Write-Host "Pages:      $PagesPath"
Write-Host "Questions:  $QuestionsPath"
Write-Host "Cleaning:   $CleaningConfig"
Write-Host "Embedding:  $EmbeddingConfig"
Write-Host "Chunkings:  $($chunkingVariants | ForEach-Object { Split-Path $_ -Leaf } | Sort-Object | Out-String)"
Write-Host "Retrievals: $($retrievalConfigs | ForEach-Object { Split-Path $_ -Leaf } | Sort-Object | Out-String)"

foreach ($retrievalConfig in $retrievalConfigs) {

    # Parent-child retrieval requires the parent/child chunking config
    $localChunkings = $chunkingVariants
    if ($retrievalConfig -like "*parent_child*") {
        $localChunkings = @((Join-Path $RepoRoot "configs\chunking\tree_parent_child.yaml"))
        if (!(Test-Path $localChunkings[0])) { throw "Missing parent-child chunking config: $($localChunkings[0])" }
    }

    foreach ($chunkCfg in $localChunkings) {
        Write-Host ""
        Write-Host "---- Running ----"
        Write-Host "Chunking:  $(Split-Path $chunkCfg -Leaf)"
        Write-Host "Retrieval: $(Split-Path $retrievalConfig -Leaf)"
        Write-Host "-----------------"

        & $PY -m src.docproc.eval.run_retrieval_strategy_eval `
            --pages $PagesPath `
            --questions $QuestionsPath `
            --chunking-config $chunkCfg `
            --retrieval-config $retrievalConfig `
            --cleaning-config $CleaningConfig `
            --embedding-config $EmbeddingConfig
    }
}

Write-Host ""
Write-Host "Done. Check results/summary/leaderboard_retrieval.csv"