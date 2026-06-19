$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path "venv")) {
  throw "venv/ not found. Create it first (python -m venv venv)."
}

& .\venv\Scripts\python -m pip install --upgrade pip | Out-Null
& .\venv\Scripts\python -m pip install -r requirements.txt pyinstaller | Out-Null
& .\venv\Scripts\python scripts\build_frontend.py
& .\venv\Scripts\python scripts\vendor_ffmpeg.py
& .\venv\Scripts\pyinstaller --noconfirm --clean backend/backend.spec

$target = (& rustc --print host-tuple).Trim()
$binDir = Join-Path $Root "src-tauri\binaries"
New-Item -ItemType Directory -Path $binDir -Force | Out-Null
$src = Join-Path $Root "dist\dronmakr-backend\dronmakr-backend.exe"
$dest = Join-Path $binDir "dronmakr-backend-$target.exe"
Copy-Item $src $dest -Force

npm ci
npm run tauri build

$version = (& .\venv\Scripts\python -c "import sys; sys.path.insert(0, 'backend'); from dronmakr.version import __version__; print(__version__)").Trim()
$artifactDir = Join-Path $Root "dist-artifacts"
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$zipPath = Join-Path $artifactDir "dronmakr-v$version-windows-x64.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
$bundleRoot = Join-Path $Root "src-tauri\target\release\bundle"
Compress-Archive -Path (Join-Path $bundleRoot "*") -DestinationPath $zipPath
Write-Host "Built $zipPath"
