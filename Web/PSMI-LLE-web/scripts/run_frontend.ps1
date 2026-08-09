$webRoot = Split-Path -Parent $PSScriptRoot
$frontendRoot = Join-Path $webRoot "frontend"

# Dependencies are installed explicitly with `npm ci` before the first run.
Set-Location $frontendRoot
npm run dev
