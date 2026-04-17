param(
    [string]$TrainingProfile = "sft_qwen35_2b",
    [string]$AppProfile = "default"
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Resolve-Path (Join-Path $repoRoot ".venv\Scripts\python.exe")
$env:PYTHONPATH = (Resolve-Path (Join-Path $repoRoot "src")).Path

$args = @(
    "-m", "decomp_clarifier.cli",
    "train-sft",
    "--training-profile", $TrainingProfile,
    "--app-profile", $AppProfile
)

& $python.Path @args
$returnCode = $LASTEXITCODE
if ($returnCode -ne 0) {
    throw "train-sft failed with exit code $returnCode"
}
