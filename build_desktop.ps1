$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path "venv\Scripts\python.exe")) {
  Write-Host "venv not found. Create it first with: python -m venv venv"
  exit 1
}

& "venv\Scripts\python.exe" -m pip install --upgrade pip
& "venv\Scripts\python.exe" -m pip install -r requirements.txt pyinstaller
& "venv\Scripts\pyinstaller.exe" --noconfirm --clean desktop.spec

$version = & "venv\Scripts\python.exe" -c "from version import __version__; print(__version__)"
$artifactDir = Join-Path $root "dist-artifacts"
New-Item -ItemType Directory -Path $artifactDir -Force | Out-Null
$zipPath = Join-Path $artifactDir "dronmakr-v$version-windows-x64.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Compress-Archive -Path "dist\dronmakr\*" -DestinationPath $zipPath
Write-Host "Built artifact: $zipPath"
