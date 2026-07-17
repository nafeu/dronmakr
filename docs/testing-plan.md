# Dronmakr Local E2E Testing Plan

## Goals

- CI stays on builds, lint, packaging, basic validation.
- Full desktop E2E runs locally in VMs with real sidecar + (optionally) packaged app.
- Playwright drives UI; Surge XT required for auditionr smoke.

## Workflow (current)

**No SSH orchestration.** You set up each VM manually, clone repo inside guest, pull, run per-OS script.

```
Ubuntu VM:  cd ~/dronmakr && git pull && ./scripts/e2e/run-ubuntu-e2e.sh
Arch VM:    cd ~/dronmakr && git pull && ./scripts/e2e/run-arch-e2e.sh
Windows VM: cd ~/dronmakr; git pull; .\scripts\e2e\run-windows-e2e.ps1
macOS host: cd dronmakr && git pull && ./run-all-e2e
```

One-time VM setup: **[e2e-vm-setup.md](e2e-vm-setup.md)** (UTM + Ubuntu/Arch/Windows ARM on Mac mini).

## Test mode

`DRONMAKR_TEST=1` (`backend/dronmakr/core/test_mode.py`):

- Temp root via `DRONMAKR_TEST_ROOT`
- Isolated settings (never touches repo `config/`)
- Auto-seed `PLUGIN_PATHS`
- Updater disabled
- `/api/health` → `testMode: true`

## Playwright

```
tests/e2e/
    onboarding.spec.ts
    auditionr.spec.ts
    helpers/
```

Config: `playwright.config.ts` — longer timeouts on Linux.

## Initial test case

1. Launch app or `E2E_BACKEND_ONLY=1` backend
2. Complete onboarding
3. Detect Surge XT
4. Auditionr → load Surge → Generate
5. Validate WAV (`scripts/e2e/validate_audio.py`)

## Scripts

| Script | Where to run |
|--------|----------------|
| `run-all-e2e` | macOS/Linux host — current OS only |
| `scripts/e2e/run-macos-e2e.sh` | macOS |
| `scripts/e2e/run-ubuntu-e2e.sh` | Ubuntu VM |
| `scripts/e2e/run-arch-e2e.sh` | Arch VM |
| `scripts/e2e/run-linux-e2e.sh` | Any Linux (auto-detect) |
| `scripts/e2e/run-windows-e2e.ps1` | Windows VM |

Options (bash runners):

```bash
E2E_BACKEND_ONLY=1 ./scripts/e2e/run-ubuntu-e2e.sh   # fast dev
npm run build && ./scripts/e2e/run-ubuntu-e2e.sh     # packaged app
./run-all-e2e --build --backend-only                 # macOS fast
```

## ARM note

GitHub release Linux tarballs are **x64**. On ARM VMs, build inside the guest (`npm run build`) or use `E2E_BACKEND_ONLY=1`.

## Future tests

recorder, sample editor, batch generation, presets, sidecar restart, missing plugin handling, export validation.
