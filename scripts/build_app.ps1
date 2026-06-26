$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path "venv")) {
  throw "venv/ not found. Create it first (python -m venv venv)."
}

& .\scripts\build_sidecar.ps1

npm ci
# Tauri beforeBuildCommand runs `python scripts/build_frontend.py` — use venv on PATH.
& .\venv\Scripts\Activate.ps1
npm run tauri build

$version = (& .\venv\Scripts\python -c "import sys; sys.path.insert(0, 'backend'); from dronmakr.version import __version__; print(__version__)").Trim()
$artifactDir = Join-Path $Root "dist-artifacts"
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$zipPath = Join-Path $artifactDir "dronmakr-v$version-windows-x64.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
$bundleRoot = Join-Path $Root "src-tauri\target\release\bundle"
Compress-Archive -Path (Join-Path $bundleRoot "*") -DestinationPath $zipPath
Write-Host "Built $zipPath"
