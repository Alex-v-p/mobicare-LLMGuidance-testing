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

$env:PYTHONPATH = (Join-Path $RepoRoot "src")

$PagesPath     = Join-Path $RepoRoot "data\processed\guidelines_pages.jsonl"
$QuestionsPath = Join-Path $RepoRoot "data\eval\retrieval_questions.jsonl"

if (!(Test-Path $PagesPath))     { throw "Pages file not found: $PagesPath" }
if (!(Test-Path $QuestionsPath)) { throw "Questions file not found: $QuestionsPath" }

$CleaningConfig  = Join-Path $RepoRoot "configs\cleaning\deep.yaml"
$EmbeddingConfig = Join-Path $RepoRoot "configs\embeddings\bge_small.yaml"

if ($IncludeBioEmbedding) {
    $EmbeddingConfig = Join-Path $RepoRoot "configs\embeddings\bioclinical_sbert.yaml"
}

if (!(Test-Path $CleaningConfig))  { throw "Cleaning config not found: $CleaningConfig" }
if (!(Test-Path $EmbeddingConfig)) { throw "Embedding config not found: $EmbeddingConfig" }

function Get-Leaf($p) { return (Split-Path $p -Leaf) }

# -----------------------
# Chunking configs
# -----------------------
$TreeChunkCfg = (Join-Path $RepoRoot "configs\chunking\tree_parent_child.yaml")
$LateChunkCfg = (Join-Path $RepoRoot "configs\chunking\late_chunking.yaml")

$chunkingVariants = @()
$chunkingVariants += (Join-Path $RepoRoot "configs\chunking\naive_300_50.yaml")
$chunkingVariants += (Join-Path $RepoRoot "configs\chunking\page_index.yaml")
$chunkingVariants += $TreeChunkCfg
$chunkingVariants += $LateChunkCfg

if ($IncludeOverlap300100) {
    $chunkingVariants += (Join-Path $RepoRoot "configs\chunking\naive_300_100.yaml")
}

# -----------------------
# Retrieval configs (base)
# -----------------------
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

    # tree retrieval variants
    $retrievalConfigs += (Join-Path $RepoRoot "configs\retrieval\tree_dense.yaml")
    $retrievalConfigs += (Join-Path $RepoRoot "configs\retrieval\tree_bm25.yaml")
    $retrievalConfigs += (Join-Path $RepoRoot "configs\retrieval\tree_hybrid.yaml")
    $retrievalConfigs += (Join-Path $RepoRoot "configs\retrieval\tree_dense_rerank.yaml")
    $retrievalConfigs += (Join-Path $RepoRoot "configs\retrieval\tree_rewrite_then_hybrid.yaml")
}

# -----------------------
# Late-chunking retrieval configs (auto-discover)
# -----------------------
$retrievalDir = Join-Path $RepoRoot "configs\retrieval"
$lateRetrievalConfigs = @()

# This picks up: late_chunking_dense.yaml, late_chunking_page_dense.yaml, late_chunking_page_bm25.yaml, etc.
$lateRetrievalConfigs = Get-ChildItem -Path $retrievalDir -Filter "late_chunking*.yaml" -File |
    Sort-Object Name |
    ForEach-Object { $_.FullName }

# -----------------------
# Validate files exist
# -----------------------
foreach ($p in $chunkingVariants + $retrievalConfigs + $lateRetrievalConfigs) {
    if (!(Test-Path $p)) { throw "Missing config file: $p" }
}

Write-Host "== Retrieval strategy matrix =="
Write-Host "RepoRoot:   $RepoRoot"
Write-Host "Pages:      $PagesPath"
Write-Host "Questions:  $QuestionsPath"
Write-Host "Cleaning:   $CleaningConfig"
Write-Host "Embedding:  $EmbeddingConfig"
Write-Host "Chunkings:  $($chunkingVariants | ForEach-Object { Get-Leaf $_ } | Sort-Object | Out-String)"
Write-Host "Retrievals: $($retrievalConfigs | ForEach-Object { Get-Leaf $_ } | Sort-Object | Out-String)"
Write-Host "Late-Retrievals: $($lateRetrievalConfigs | ForEach-Object { Get-Leaf $_ } | Sort-Object | Out-String)"

# -------------------------
# 1) Run ALL late-chunking retrieval configs (never skipped)
# -------------------------
foreach ($lateCfg in $lateRetrievalConfigs) {
    Write-Host ""
    Write-Host "==== Late Chunking Run ===="
    Write-Host "Chunking:  $(Get-Leaf $LateChunkCfg)"
    Write-Host "Retrieval: $(Get-Leaf $lateCfg)"
    Write-Host "==========================="

    & $PY -m src.docproc.eval.run_retrieval_strategy_eval `
        --pages $PagesPath `
        --questions $QuestionsPath `
        --chunking-config $LateChunkCfg `
        --retrieval-config $lateCfg `
        --cleaning-config $CleaningConfig `
        --embedding-config $EmbeddingConfig
}

# -------------------------
# 2) Normal matrix (EXCLUDING late-chunking retrieval configs)
# -------------------------
foreach ($retrievalConfig in $retrievalConfigs) {
    $retrLeaf = Get-Leaf $retrievalConfig
    $localChunkings = @()

    if ($retrLeaf -like "*parent_child*" -or $retrLeaf -like "tree_*") {
        # Tree retrieval configs must use tree chunking
        $localChunkings = @($TreeChunkCfg)
    } else {
        # Normal retrieval configs: run on everything EXCEPT late_chunking chunking
        $localChunkings = $chunkingVariants | Where-Object { (Get-Leaf $_) -notlike "*late_chunking*" }
    }

    foreach ($chunkCfg in $localChunkings) {
        Write-Host ""
        Write-Host "---- Running ----"
        Write-Host "Chunking:  $(Get-Leaf $chunkCfg)"
        Write-Host "Retrieval: $retrLeaf"
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