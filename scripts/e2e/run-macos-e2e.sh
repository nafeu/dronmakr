#!/usr/bin/env bash
# macOS native E2E runner.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export DRONMAKR_TEST=1
export DRONMAKR_TEST_ROOT="${DRONMAKR_TEST_ROOT:-$(mktemp -d /tmp/dronmakr-macos-e2e-XXXXXX)}"
export DRONMAKR_TEST_FILES_ROOT="${DRONMAKR_TEST_FILES_ROOT:-${DRONMAKR_TEST_ROOT}/files}"
export E2E_PORT="${E2E_PORT:-3766}"
export E2E_HOST="${E2E_HOST:-127.0.0.1}"
export E2E_BASE_URL="http://${E2E_HOST}:${E2E_PORT}"
export E2E_LOG_DIR="${E2E_LOG_DIR:-${DRONMAKR_TEST_ROOT}/logs}"
export E2E_ARTIFACTS_DIR="${E2E_ARTIFACTS_DIR:-${DRONMAKR_TEST_ROOT}/playwright}"
mkdir -p "$E2E_LOG_DIR" "$E2E_ARTIFACTS_DIR"

if [[ ! -d venv ]]; then
  echo "e2e-macos: venv/ missing" >&2
  exit 1
fi
source venv/bin/activate

cleanup() {
  if [[ -n "${APP_PID:-}" ]] && kill -0 "$APP_PID" 2>/dev/null; then
    kill "$APP_PID" 2>/dev/null || true
    wait "$APP_PID" 2>/dev/null || true
  fi
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${E2E_BACKEND_ONLY:-0}" == "1" ]]; then
  PYTHONPATH=backend python backend/backend_server.py \
    --host "$E2E_HOST" \
    --port "$E2E_PORT" \
    --debug \
    --dev-frontend \
    >"${E2E_LOG_DIR}/backend.stdout.log" 2>&1 &
  BACKEND_PID=$!
else
  APP="${E2E_MACOS_APP:-${ROOT_DIR}/src-tauri/target/release/bundle/macos/dronmakr.app}"
  if [[ ! -d "$APP" ]]; then
    echo "e2e-macos: app bundle missing at ${APP} — build or set E2E_MACOS_APP" >&2
    exit 1
  fi
  open -a "$APP" --args
  echo "e2e-macos: launched ${APP}"
fi

E2E_HEALTH_TIMEOUT="${E2E_HEALTH_TIMEOUT:-180}"
source scripts/e2e/wait-backend.sh
export E2E_BASE_URL="http://${E2E_HOST}:${E2E_PORT}"

if [[ ! -d node_modules/@playwright/test ]]; then
  npm ci
fi
if [[ ! -x "$(command -v playwright)" ]] && [[ ! -d node_modules/.cache/ms-playwright ]]; then
  npx playwright install chromium
fi

PLAYWRIGHT_HTML_REPORT="${E2E_ARTIFACTS_DIR}/report" \
  npx playwright test --config=playwright.config.ts "$@"

echo "e2e-macos: PASS (artifacts: ${E2E_ARTIFACTS_DIR})"
