# Dronmakr Local E2E Testing Plan

## Goals

- Keep existing CI focused on builds, linting, packaging, and basic validation.
- Run comprehensive E2E tests locally from a macOS development machine.
- Use **Playwright** as the automation framework.
- Use **UTM** VMs for Windows and Linux testing.
- Run against the **real packaged Tauri app** with the **real Python sidecar**.

## High-Level Architecture

```
Mac mini
 ├── macOS native
 ├── UTM Ubuntu VM
 └── UTM Windows VM

Single command
    ↓
Build packaged app
    ↓
Boot VM
    ↓
Copy latest build
    ↓
Launch app
    ↓
Playwright drives UI
    ↓
Validate generated audio
    ↓
Collect logs/screenshots
```

## Why not CI?

CI will continue to verify:

- Builds
- Packaging
- Basic validation

Heavy desktop workflow testing remains local where plugins, audio drivers, and snapshots are deterministic.

## Test Mode

`DRONMAKR_TEST=1` is implemented in `backend/dronmakr/core/test_mode.py`.

When set:

- Uses temp folders under `DRONMAKR_TEST_ROOT` (auto-created if unset)
- Isolated `config/settings.json` (never touches repo `config/`)
- Seeds `PLUGIN_PATHS` from OS defaults (override with `DRONMAKR_TEST_PLUGIN_PATHS`)
- Disables auto-update checks
- Extra debug logging
- `/api/health` reports `testMode: true`

Optional: `DRONMAKR_TEST_AUTO_FILES_ROOT=1` skips onboarding UI by pre-setting `FILES_ROOT`.

## VM Strategy

Create three golden environments:

- macOS (native)
- Ubuntu (UTM)
- Windows (UTM)

Each snapshot contains:

- Python dependencies
- Surge XT installed
- identical sample library
- identical settings
- Playwright runtime

If Surge XT is missing, the test fails immediately (`scripts/e2e/run-linux-e2e.sh` + `requireSurgeXt` helper).

## Playwright

Framework: `@playwright/test` with config at `playwright.config.ts`.

Directory layout:

```
tests/e2e/
    onboarding.spec.ts
    auditionr.spec.ts
    helpers/
        app.ts
        audio.ts
```

Reusable helpers cover onboarding, navigation, plugin selection (Surge XT), and WAV validation via `scripts/e2e/validate_audio.py`.

Linux gets longer timeouts (plugin load / VST render).

## Initial Test Case

1. Launch packaged app (or `E2E_BACKEND_ONLY=1` dev backend).
2. Complete onboarding.
3. Verify Surge XT is detected.
4. Navigate to Auditionr.
5. Load Surge XT in instrument slot.
6. Press Generate.
7. Wait for render.
8. Verify:
   - file exists
   - duration > 0
   - RMS/amplitude above silence threshold
   - file decodes successfully
9. Pass.

## Audio Validation

No waveform snapshots. `scripts/e2e/validate_audio.py` checks:

- readable WAV (soundfile decode)
- expected sample rate (optional)
- duration > threshold
- non-zero peak/RMS
- decoder errors fail the run

## Automation

```bash
./run-all-e2e                    # default target = current OS
./run-all-e2e --build --macos    # build + macOS packaged app
./run-all-e2e --linux            # native Linux (UTM guest)
./run-all-e2e --utm-linux        # Mac orchestrator → SSH guest
./run-all-e2e --backend-only     # fast dev: Python sidecar only
```

Guest scripts:

- `scripts/e2e/run-macos-e2e.sh`
- `scripts/e2e/run-linux-e2e.sh` (Xvfb + Surge gate)
- `scripts/e2e/run-windows-e2e.ps1` (stub)

Artifacts: screenshots/video/trace under `E2E_ARTIFACTS_DIR` or `test-results/`.

## UTM Automation

`run-all-e2e` probes:

- `utmctl start <vm>` when available
- SSH readiness on `UTM_LINUX_SSH` (default `dronmakr@127.0.0.1 -p 2222`)
- `scp` Linux tarball → guest → `run-linux-e2e.sh`

Fallback: start VM manually in UTM.

Env:

| Variable | Default |
|----------|---------|
| `UTM_LINUX_VM` | `dronmakr-ubuntu-e2e` |
| `UTM_LINUX_SSH` | `dronmakr@127.0.0.1 -p 2222` |
| `E2E_PORT` | `3766` |
| `E2E_BACKEND_ONLY` | `0` |

## Future Tests

- recorder
- sample editor
- batch generation
- drag/drop into DAW
- presets
- settings persistence
- Python sidecar restart
- corrupted input files
- missing plugin handling
- export validation
