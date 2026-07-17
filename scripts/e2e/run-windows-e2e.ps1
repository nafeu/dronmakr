# Windows VM E2E — run inside guest after one-time setup (docs/e2e-vm-setup.md).
# Usage: .\scripts\e2e\run-windows-e2e.ps1

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PlaywrightArgs
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $RootDir

$env:DRONMAKR_TEST = "1"
if (-not $env:DRONMAKR_TEST_ROOT) {
    $env:DRONMAKR_TEST_ROOT = Join-Path $env:TEMP ("dronmakr-win-e2e-" + [guid]::NewGuid().ToString("n"))
}
$env:DRONMAKR_TEST_FILES_ROOT = Join-Path $env:DRONMAKR_TEST_ROOT "files"
$env:E2E_PORT = if ($env:E2E_PORT) { $env:E2E_PORT } else { "3766" }
$env:E2E_HOST = if ($env:E2E_HOST) { $env:E2E_HOST } else { "127.0.0.1" }
$LogDir = if ($env:E2E_LOG_DIR) { $env:E2E_LOG_DIR } else { Join-Path $env:DRONMAKR_TEST_ROOT "logs" }
$ArtifactsDir = if ($env:E2E_ARTIFACTS_DIR) { $env:E2E_ARTIFACTS_DIR } else { Join-Path $env:DRONMAKR_TEST_ROOT "playwright" }
New-Item -ItemType Directory -Force -Path $LogDir, $ArtifactsDir | Out-Null

function Require-SurgeXt {
    $needles = @("surge xt", "surge_xt", "surge-xt")
    $roots = @()
    if ($env:DRONMAKR_TEST_PLUGIN_PATHS) {
        $roots += ($env:DRONMAKR_TEST_PLUGIN_PATHS -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    }
    $localAppData = [Environment]::GetFolderPath("LocalApplicationData")
    $programFiles = ${env:ProgramFiles}
    $roots += @(
        (Join-Path $localAppData "Programs\Common\VST3"),
        (Join-Path $programFiles "Common Files\VST3")
    )
    foreach ($root in $roots | Select-Object -Unique) {
        if (-not (Test-Path $root)) { continue }
        $matches = Get-ChildItem -Path $root -Recurse -Include *.vst3,*.dll -ErrorAction SilentlyContinue |
            Where-Object { $n = $_.Name.ToLower(); $needles | Where-Object { $n -like "*$_*" } }
        if ($matches) {
            Write-Host "e2e-windows: Surge XT found at $($matches[0].FullName)"
            return
        }
    }
    throw "Surge XT not found — install VST3 and/or set DRONMAKR_TEST_PLUGIN_PATHS"
}

function Wait-Backend {
    param([int]$TimeoutSec = 240)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $ports = @(3766, 3767, 3768, 3769)
    while ((Get-Date) -lt $deadline) {
        foreach ($port in $ports) {
            try {
                $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$port/api/health" -TimeoutSec 2
                if ($resp.ok) {
                    $env:E2E_PORT = "$port"
                    $env:E2E_BASE_URL = "http://127.0.0.1:$port"
                    Write-Host "e2e-windows: backend ready at $($env:E2E_BASE_URL)"
                    return
                }
            } catch { }
        }
        Start-Sleep -Seconds 1
    }
    throw "backend /api/health not ready after ${TimeoutSec}s"
}

function Find-WindowsApp {
    if ($env:E2E_APP_BINARY -and (Test-Path $env:E2E_APP_BINARY)) {
        return $env:E2E_APP_BINARY
    }
    $bundle = Join-Path $RootDir "src-tauri\target\release\bundle"
    $candidates = @(
        (Join-Path $bundle "msi\dronmakr*.exe"),
        (Join-Path $bundle "nsis\dronmakr*.exe")
    )
    foreach ($pattern in $candidates) {
        $hit = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    throw "Packaged app missing — run npm run build or set E2E_APP_BINARY"
}

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    throw "venv/ missing — run one-time dev setup (docs/e2e-vm-setup.md)"
}

Write-Host "e2e-windows: test root=$($env:DRONMAKR_TEST_ROOT)"
Require-SurgeXt

$backend = $null
$app = $null
try {
    if ($env:E2E_BACKEND_ONLY -eq "1") {
        Write-Host "e2e-windows: backend-only mode"
        $backend = Start-Process -FilePath ".\venv\Scripts\python.exe" `
            -ArgumentList @("backend\backend_server.py", "--port", $env:E2E_PORT, "--debug", "--dev-frontend") `
            -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $LogDir "backend.stdout.log") `
            -RedirectStandardError (Join-Path $LogDir "backend.stderr.log")
    } else {
        $appPath = Find-WindowsApp
        Write-Host "e2e-windows: launching packaged app $appPath"
        $app = Start-Process -FilePath $appPath -PassThru `
            -RedirectStandardOutput (Join-Path $LogDir "app.stdout.log") `
            -RedirectStandardError (Join-Path $LogDir "app.stderr.log")
    }

    Wait-Backend

    if (-not (Test-Path "node_modules\@playwright\test")) {
        npm ci
    }
    if (-not (Test-Path "node_modules\.cache\ms-playwright")) {
        npx playwright install chromium
    }

    $env:PLAYWRIGHT_HTML_REPORT = Join-Path $ArtifactsDir "report"
    if ($PlaywrightArgs.Count -gt 0) {
        npx playwright test --config=playwright.config.ts @PlaywrightArgs
    } else {
        npx playwright test --config=playwright.config.ts
    }
    Write-Host "e2e-windows: PASS (artifacts: $ArtifactsDir)"
} finally {
    if ($backend -and -not $backend.HasExited) { Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue }
    if ($app -and -not $app.HasExited) { Stop-Process -Id $app.Id -Force -ErrorAction SilentlyContinue }
}
