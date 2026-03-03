param(
    # Comma-separated list of Ollama models to benchmark, e.g. "llama3.1:8b,qwen2.5:7b"
    # If empty, the script will read models from -ModelsFile (default: experiments/008_llm_models/models.txt).
    # If the file does not exist or is empty, it will fall back to auto-discovering installed models via `ollama list`.
    [string]$Models = "",

    # Path to a file that contains one model name per line. Lines starting with # are ignored.
    [string]$ModelsFile = "experiments\008_llm_models\models.txt",

    [string]$Prompts = "data\eval\llm_bench_prompts.jsonl",

    [int]$Repeats = 3,
    [int]$Warmup = 1,

    # Override Ollama base URL if needed
    [string]$OllamaBaseUrl = "",

    # Keep running even if one model fails
    [switch]$ContinueOnError
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$PY = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (!(Test-Path $PY)) {
    throw "Venv python not found at: $PY. Are you sure .venv exists?"
}

$env:PYTHONPATH = (Join-Path $RepoRoot "src")

if ([string]::IsNullOrWhiteSpace($OllamaBaseUrl) -and $env:OLLAMA_BASE_URL) {
    $OllamaBaseUrl = $env:OLLAMA_BASE_URL
}
if ([string]::IsNullOrWhiteSpace($OllamaBaseUrl)) {
    $OllamaBaseUrl = "http://localhost:11434"
}

$PromptsPath = Join-Path $RepoRoot $Prompts
if (!(Test-Path $PromptsPath)) {
    throw "Prompts file not found: $PromptsPath"
}

Write-Host "== Ollama LLM benchmark =="
Write-Host "RepoRoot:       $RepoRoot"
Write-Host "Prompts:        $PromptsPath"
Write-Host "Repeats:        $Repeats"
Write-Host "Warmup:         $Warmup"
Write-Host "Ollama base URL: $OllamaBaseUrl"

function Get-ModelsFromFile([string]$Path) {
    if (!(Test-Path $Path)) { return @() }
    return Get-Content $Path |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -and -not $_.StartsWith("#") }
}

function Get-ModelsFromOllamaList() {
    try {
        $out = & ollama list 2>$null
        if (!$out) { return @() }
        $lines = $out -split "`r?`n" | Where-Object { $_ -and ($_ -notmatch '^\s*NAME\s+') }
        $names = @()
        foreach ($line in $lines) {
            $parts = ($line -split '\s+')
            if ($parts.Length -gt 0 -and $parts[0]) { $names += $parts[0] }
        }
        return $names
    } catch {
        return @()
    }
}

# Resolve models (priority): -Models, models file, `ollama list`, $env:OLLAMA_MODEL
$ModelsList = @()

if (-not [string]::IsNullOrWhiteSpace($Models)) {
    $ModelsList = $Models -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ }
} else {
    $ModelsFilePath = Join-Path $RepoRoot $ModelsFile
    $ModelsList = Get-ModelsFromFile $ModelsFilePath

    if ($ModelsList.Count -eq 0) {
        $ModelsList = Get-ModelsFromOllamaList
    }
    if ($ModelsList.Count -eq 0 -and $env:OLLAMA_MODEL) {
        $ModelsList = $env:OLLAMA_MODEL -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    }
}

if ($ModelsList.Count -eq 0) {
    throw "No Ollama models found. Provide -Models, add names to $ModelsFile, or install models (ollama pull ...)."
}

$Models = ($ModelsList -join ",")
Write-Host ("Models:         " + $Models)
if ($ModelsFilePath) { Write-Host ("ModelsFile:     " + $ModelsFilePath) }

# Optional: show current processor placement (best-effort)
try {
    $ps = & ollama ps 2>$null
    if ($ps) {
        Write-Host "\nCurrent ollama ps:\n$ps\n"
    }
} catch { }

$script = Join-Path $RepoRoot "src\docproc\eval\run_ollama_model_bench.py"

$cmd = @(
    $PY,
    "-m", "docproc.eval.run_ollama_model_bench",
    "--outdir", "results",
    "--prompts", $PromptsPath,
    "--models", $Models,
    "--repeats", $Repeats,
    "--warmup", $Warmup,
    "--base-url", $OllamaBaseUrl
)

Write-Host ("Running: " + ($cmd -join " "))

try {
    & $cmd[0] $cmd[1..($cmd.Length-1)]
} catch {
    if ($ContinueOnError) {
        Write-Warning "Benchmark failed: $_"
    } else {
        throw
    }
}
