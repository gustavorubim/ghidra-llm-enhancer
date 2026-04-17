param(
    [string]$TrainingProfile = "grpo_qwen35_2b",
    [string]$AppProfile = "default"
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Resolve-Path (Join-Path $repoRoot ".venv\Scripts\python.exe")
$env:PYTHONPATH = (Resolve-Path (Join-Path $repoRoot "src")).Path

$args = @(
    "-m", "decomp_clarifier.cli",
    "train-grpo",
    "--training-profile", $TrainingProfile,
    "--app-profile", $AppProfile
)

& $python.Path @args
$returnCode = $LASTEXITCODE
if ($returnCode -ne 0) {
    throw "train-grpo failed with exit code $returnCode"
}
