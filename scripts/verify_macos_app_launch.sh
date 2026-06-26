#!/usr/bin/env bash
# Verify the bundled macOS sidecar starts and serves /api/health within a timeout.
# Run after Tauri packaging in CI and locally before release.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

APP="${1:-src-tauri/target/release/bundle/macos/dronmakr.app}"
SIDECAR="${APP}/Contents/MacOS/dronmakr-backend"
PORT="${PORT:-3996}"
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "verify_macos_app_launch: skip (not macOS)"
  exit 0
fi

if [[ ! -x "$SIDECAR" ]]; then
  echo "error: sidecar missing at ${SIDECAR}" >&2
  exit 1
fi

echo "Launching sidecar from ${SIDECAR} on port ${PORT} (timeout ${TIMEOUT_SEC}s)..."
"$SIDECAR" --port "$PORT" &
PID=$!
cleanup() {
  kill "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
}
trap cleanup EXIT

deadline=$((SECONDS + TIMEOUT_SEC))
while (( SECONDS < deadline )); do
  if curl -sf "http://127.0.0.1:${PORT}/api/health" >/dev/null; then
    echo "verify_macos_app_launch: health OK on port ${PORT}"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "error: sidecar exited before becoming ready" >&2
    exit 1
  fi
  sleep 2
done

echo "error: backend did not respond within ${TIMEOUT_SEC}s" >&2
exit 1
