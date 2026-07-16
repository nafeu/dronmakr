# PowerShell stub for Windows UTM guest E2E.
# Usage (inside golden VM): .\scripts\e2e\run-windows-e2e.ps1

$ErrorActionPreference = "Stop"

$env:DRONMAKR_TEST = "1"
if (-not $env:DRONMAKR_TEST_ROOT) {
  $env:DRONMAKR_TEST_ROOT = Join-Path $env:TEMP ("dronmakr-win-e2e-" + [guid]::NewGuid().ToString("n"))
}
$env:DRONMAKR_TEST_FILES_ROOT = Join-Path $env:DRONMAKR_TEST_ROOT "files"
$env:E2E_PORT = if ($env:E2E_PORT) { $env:E2E_PORT } else { "3766" }
$env:E2E_HOST = if ($env:E2E_HOST) { $env:E2E_HOST } else { "127.0.0.1" }
$env:E2E_BASE_URL = "http://$($env:E2E_HOST):$($env:E2E_PORT)"

Write-Host "e2e-windows: test root=$($env:DRONMAKR_TEST_ROOT)"
Write-Host "e2e-windows: start packaged dronmakr or set E2E_BACKEND_ONLY=1"

if ($env:E2E_BACKEND_ONLY -eq "1") {
  $backend = Start-Process -FilePath ".\venv\Scripts\python.exe" `
    -ArgumentList @("backend\backend_server.py", "--port", $env:E2E_PORT, "--debug", "--dev-frontend") `
    -PassThru -WindowStyle Hidden
  try {
    bash scripts/e2e/wait-backend.sh
    npx playwright test --config=playwright.config.ts @args
  } finally {
    if ($backend -and -not $backend.HasExited) { Stop-Process -Id $backend.Id -Force }
  }
} else {
  throw "Packaged Windows E2E: launch dronmakr.exe manually, then run: npx playwright test"
}
