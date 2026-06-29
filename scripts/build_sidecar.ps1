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
& .\venv\Scripts\python scripts\stage_sidecar_onedir_dist.py

$target = (& rustc --print host-tuple).Trim()
Write-Host "Sidecar onedir staged under src-tauri/dronmakr-backend/"
