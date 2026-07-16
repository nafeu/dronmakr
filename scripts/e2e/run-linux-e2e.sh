#!/usr/bin/env bash
# Linux guest E2E runner — headless display, Surge XT gate, Playwright smoke.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export DRONMAKR_TEST=1
export DRONMAKR_ASYNC_MODE="${DRONMAKR_ASYNC_MODE:-threading}"
export DRONMAKR_TEST_ROOT="${DRONMAKR_TEST_ROOT:-$(mktemp -d /tmp/dronmakr-linux-e2e-XXXXXX)}"
export DRONMAKR_TEST_FILES_ROOT="${DRONMAKR_TEST_FILES_ROOT:-${DRONMAKR_TEST_ROOT}/files}"
export E2E_PORT="${E2E_PORT:-3766}"
export E2E_HOST="${E2E_HOST:-127.0.0.1}"
export E2E_BASE_URL="http://${E2E_HOST}:${E2E_PORT}"
export E2E_LOG_DIR="${E2E_LOG_DIR:-${DRONMAKR_TEST_ROOT}/logs}"
export E2E_ARTIFACTS_DIR="${E2E_ARTIFACTS_DIR:-${DRONMAKR_TEST_ROOT}/playwright}"
mkdir -p "$E2E_LOG_DIR" "$E2E_ARTIFACTS_DIR"

if [[ ! -d venv ]]; then
  echo "e2e-linux: venv/ missing" >&2
  exit 1
fi
source venv/bin/activate

require_surge_xt() {
  python - <<'PY'
import glob
import os
import sys

needles = ("surge xt", "surge_xt", "surge-xt")
roots = []
for chunk in (os.environ.get("DRONMAKR_TEST_PLUGIN_PATHS") or "").split(","):
    chunk = chunk.strip()
    if chunk:
        roots.append(chunk)
home = os.path.expanduser("~")
roots.extend(
    [
        os.path.join(home, ".vst3"),
        os.path.join(home, ".vst"),
        os.path.join(home, "vst"),
        "/usr/lib/vst3",
        "/usr/local/lib/vst3",
    ]
)
seen = set()
for root in roots:
    if not root or root in seen or not os.path.isdir(root):
        continue
    seen.add(root)
    for pattern in ("*.vst3", "*.so", "*.vst"):
        for path in glob.glob(os.path.join(root, "**", pattern), recursive=True):
            name = os.path.basename(path).lower()
            if any(n in name for n in needles):
                print(path)
                raise SystemExit(0)
print("Surge XT not found under PLUGIN_PATHS or Linux default VST folders", file=sys.stderr)
raise SystemExit(1)
PY
}

echo "e2e-linux: checking Surge XT prerequisite..."
require_surge_xt

XVFB_PID=""
if [[ -z "${DISPLAY:-}" ]]; then
  if command -v Xvfb >/dev/null 2>&1; then
    Xvfb :99 -screen 0 1280x900x24 -ac +extension GLX +render -noreset \
      >"${E2E_LOG_DIR}/xvfb.log" 2>&1 &
    XVFB_PID=$!
    export DISPLAY=:99
    sleep 1
    echo "e2e-linux: started Xvfb on ${DISPLAY}"
  else
    echo "e2e-linux: DISPLAY unset and Xvfb missing — install xvfb" >&2
    exit 1
  fi
fi

cleanup() {
  if [[ -n "${APP_PID:-}" ]] && kill -0 "$APP_PID" 2>/dev/null; then
    kill "$APP_PID" 2>/dev/null || true
    wait "$APP_PID" 2>/dev/null || true
  fi
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "$XVFB_PID" ]] && kill -0 "$XVFB_PID" 2>/dev/null; then
    kill "$XVFB_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${E2E_BACKEND_ONLY:-0}" == "1" ]]; then
  echo "e2e-linux: backend-only mode"
  PYTHONPATH=backend python backend/backend_server.py \
    --host "$E2E_HOST" \
    --port "$E2E_PORT" \
    --debug \
    --dev-frontend \
    >"${E2E_LOG_DIR}/backend.stdout.log" 2>&1 &
  BACKEND_PID=$!
else
  APP_BIN="$(bash scripts/e2e/find-linux-app.sh)"
  echo "e2e-linux: launching packaged app ${APP_BIN}"
  "$APP_BIN" >"${E2E_LOG_DIR}/app.stdout.log" 2>&1 &
  APP_PID=$!
fi

source scripts/e2e/wait-backend.sh
export E2E_BASE_URL="http://${E2E_HOST}:${E2E_PORT}"

if [[ ! -d node_modules/@playwright/test ]]; then
  npm ci
fi
if [[ ! -d node_modules/.cache/ms-playwright ]]; then
  npx playwright install chromium --with-deps
fi

echo "e2e-linux: running Playwright..."
PLAYWRIGHT_HTML_REPORT="${E2E_ARTIFACTS_DIR}/report" \
  npx playwright test --config=playwright.config.ts "$@"

echo "e2e-linux: PASS (artifacts: ${E2E_ARTIFACTS_DIR})"
