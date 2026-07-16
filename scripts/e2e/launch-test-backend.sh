#!/usr/bin/env bash
# Start dev backend for E2E (real Python sidecar, live templates).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d venv ]]; then
  echo "e2e: venv/ missing — create with: python -m venv venv" >&2
  exit 1
fi

source venv/bin/activate

export DRONMAKR_TEST=1
export DRONMAKR_ASYNC_MODE="${DRONMAKR_ASYNC_MODE:-threading}"
export DRONMAKR_TEST_ROOT="${DRONMAKR_TEST_ROOT:-$(mktemp -d /tmp/dronmakr-test-XXXXXX)}"
export DRONMAKR_TEST_FILES_ROOT="${DRONMAKR_TEST_FILES_ROOT:-${DRONMAKR_TEST_ROOT}/files}"

PORT="${E2E_PORT:-3766}"
HOST="${E2E_HOST:-127.0.0.1}"
LOG_DIR="${E2E_LOG_DIR:-${DRONMAKR_TEST_ROOT}/logs}"
mkdir -p "$LOG_DIR"

echo "e2e: test root=${DRONMAKR_TEST_ROOT}"
echo "e2e: starting backend http://${HOST}:${PORT}"

PYTHONPATH=backend python backend/backend_server.py \
  --host "$HOST" \
  --port "$PORT" \
  --debug \
  --dev-frontend \
  >"${LOG_DIR}/backend.stdout.log" 2>&1 &
BACKEND_PID=$!
echo "$BACKEND_PID" >"${DRONMAKR_TEST_ROOT}/backend.pid"

cleanup() {
  if kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

E2E_PORT="$PORT" E2E_HOST="$HOST"
source scripts/e2e/wait-backend.sh

if (($#)); then
  "$@"
  status=$?
  exit "$status"
fi

wait "$BACKEND_PID"
