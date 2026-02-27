param(
    [string]$Chunking = "configs\chunking\naive_300_50.yaml",
    [int]$K = 5
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$PY = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (!(Test-Path $PY)) { throw "Venv python not found at: $PY. Are you sure .venv exists?" }

$env:PYTHONPATH = (Join-Path $RepoRoot "src")

$PagesPath     = Join-Path $RepoRoot "data\processed\guidelines_pages.jsonl"
$QuestionsPath = Join-Path $RepoRoot "data\eval\retrieval_questions.jsonl"

$CleaningConfig  = Join-Path $RepoRoot "configs\cleaning\deep.yaml"
$EmbeddingConfig = Join-Path $RepoRoot "configs\embeddings\bge_small.yaml"

$ChunkingConfig  = Join-Path $RepoRoot $Chunking

if (!(Test-Path $PagesPath))       { throw "Pages file not found: $PagesPath" }
if (!(Test-Path $QuestionsPath))   { throw "Questions file not found: $QuestionsPath" }
if (!(Test-Path $CleaningConfig))  { throw "Cleaning config not found: $CleaningConfig" }
if (!(Test-Path $EmbeddingConfig)) { throw "Embedding config not found: $EmbeddingConfig" }
if (!(Test-Path $ChunkingConfig))  { throw "Chunking config not found: $ChunkingConfig" }

$OutDir = Join-Path $RepoRoot "results\runs"
$Leaderboard = Join-Path $PSScriptRoot "leaderboard_graph_aug.csv"

# A small matrix of graph settings (edit freely)
$graphK = @(8, 16)
$hops = @(1)
$expandPerNode = @(4, 8)
$beta = @(0.15, 0.25, 0.35)

Write-Host "== Graph augmentation matrix =="
Write-Host ("RepoRoot:   {0}" -f $RepoRoot)
Write-Host ("Pages:      {0}" -f $PagesPath)
Write-Host ("Questions:  {0}" -f $QuestionsPath)
Write-Host ("Chunking:   {0}" -f $ChunkingConfig)
Write-Host ("Cleaning:   {0}" -f $CleaningConfig)
Write-Host ("Embeddings: {0}" -f $EmbeddingConfig)
Write-Host ("K:          {0}" -f $K)
Write-Host ""

foreach ($gk in $graphK) {
  foreach ($h in $hops) {
    foreach ($e in $expandPerNode) {
      foreach ($b in $beta) {

        Write-Host ("--- run: graph_k={0} hops={1} expand_per_node={2} beta={3} ---" -f $gk, $h, $e, $b)

        & $PY -m docproc.eval.run_graph_augmentation_eval `
          --pages $PagesPath `
          --questions $QuestionsPath `
          --chunking $ChunkingConfig `
          --cleaning $CleaningConfig `
          --embeddings $EmbeddingConfig `
          --k $K `
          --graph-k $gk `
          --hops $h `
          --expand-per-node $e `
          --beta $b `
          --out-dir $OutDir `
          --leaderboard $Leaderboard

        if ($LASTEXITCODE -ne 0) { throw "Run failed with exit code $LASTEXITCODE" }
      }
    }
  }
}

Write-Host "== Done =="
