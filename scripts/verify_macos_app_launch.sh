#!/usr/bin/env bash
# Verify the bundled macOS sidecar starts and serves /api/health within a timeout.
# Uses a temporary HOME so first-launch settings creation runs (catches missing PyInstaller modules).
#
# Run locally before release via scripts/pre_release_checks.sh.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

APP="${1:-src-tauri/target/release/bundle/macos/dronmakr.app}"
SIDECAR="${APP}/Contents/Resources/resources/dronmakr-backend/dronmakr-backend"
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

TMPHOME="$(mktemp -d)"
SIDECAR_LOG="$(mktemp)"
cleanup() {
  rm -rf "$TMPHOME"
  rm -f "$SIDECAR_LOG"
  kill "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Launching sidecar from ${SIDECAR} on port ${PORT} (timeout ${TIMEOUT_SEC}s, fresh HOME)..."
HOME="$TMPHOME" "$SIDECAR" --port "$PORT" >"$SIDECAR_LOG" 2>&1 &
PID=$!

deadline=$((SECONDS + TIMEOUT_SEC))
while (( SECONDS < deadline )); do
  if curl -sf "http://127.0.0.1:${PORT}/api/health" >/dev/null; then
    echo "verify_macos_app_launch: health OK on port ${PORT}"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "error: sidecar exited before becoming ready" >&2
    echo "--- sidecar output ---" >&2
    cat "$SIDECAR_LOG" >&2 || true
    exit 1
  fi
  sleep 2
done

echo "error: backend did not respond within ${TIMEOUT_SEC}s" >&2
echo "--- sidecar output ---" >&2
cat "$SIDECAR_LOG" >&2 || true
exit 1
