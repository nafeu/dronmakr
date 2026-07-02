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
npm run tauri build -- --bundles nsis
if ($LASTEXITCODE -ne 0) {
  throw "tauri build failed with exit code $LASTEXITCODE"
}

$version = (& .\venv\Scripts\python -c "import sys; sys.path.insert(0, 'backend'); from dronmakr.version import __version__; print(__version__)").Trim()
$artifactDir = Join-Path $Root "dist-artifacts"
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$readmeDest = Join-Path $artifactDir "README-windows.txt"
Copy-Item -Path (Join-Path $Root "scripts\windows_release_readme.txt") -Destination $readmeDest -Force
$zipPath = Join-Path $artifactDir "dronmakr-v$version-windows-x64.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
$bundleRoot = Join-Path $Root "src-tauri\target\release\bundle"
if (-not (Test-Path $bundleRoot)) {
  throw "Tauri bundle output missing at $bundleRoot (build may have failed earlier)."
}
$stagingDir = Join-Path $artifactDir "windows-bundle-staging"
if (Test-Path $stagingDir) { Remove-Item $stagingDir -Recurse -Force }
New-Item -ItemType Directory -Path $stagingDir -Force | Out-Null
Copy-Item -Path (Join-Path $bundleRoot "*") -Destination $stagingDir -Recurse -Force
Copy-Item -Path $readmeDest -Destination (Join-Path $stagingDir "README-windows.txt") -Force
Compress-Archive -Path (Join-Path $stagingDir "*") -DestinationPath $zipPath
Remove-Item $stagingDir -Recurse -Force
Write-Host "Built $zipPath"
Write-Host "Built $readmeDest"
