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
$src = Join-Path $Root "dist\dronmakr-backend.exe"
if (-not (Test-Path $src)) {
  throw "PyInstaller output missing at $src"
}
$dest = Join-Path $binDir "dronmakr-backend-$target.exe"
Copy-Item $src $dest -Force
Write-Host "Sidecar ready: $dest"
