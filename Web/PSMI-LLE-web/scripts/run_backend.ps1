param(
    [string]$PythonExecutable = "python"
)

$webRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $webRoot "..\..")).Path
$sourceRoot = Join-Path $repoRoot "src"

# Reuse the repository PSMI package instead of a Web-specific code copy.
$env:PYTHONPATH = "$sourceRoot;$repoRoot"
Set-Location $webRoot
& $PythonExecutable -m backend.main
