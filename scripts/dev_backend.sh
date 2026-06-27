#!/usr/bin/env bash
# Start the Python backend for `npm run dev` with live templates from assets/.
#
# Tauri dev expects this process to stay running on port 3766. If a release-build
# sidecar (or other non-dev backend) already owns the port, we stop it so dev does
# not silently talk to stale frontend/dist HTML.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${DRONMAKR_DEV_PORT:-3766}"
HOST="${DRONMAKR_DEV_HOST:-127.0.0.1}"

probe_dev_backend() {
  curl -sf "http://${HOST}:${PORT}/dev/reload-check" 2>/dev/null \
    | grep -q '"version"'
}

listener_pids() {
  lsof -ti ":${PORT}" -sTCP:LISTEN 2>/dev/null || true
}

if listener_pids | grep -q .; then
  if probe_dev_backend; then
    echo "[dev-backend] live template server already on http://${HOST}:${PORT}"
    # Keep beforeDevCommand alive while Tauri dev runs (reuse existing server).
    while true; do sleep 86400; done
  fi

  echo "[dev-backend] port ${PORT} is in use by a non-dev backend (often a release dronmakr.app sidecar)." >&2
  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    cmd="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
    if [[ "${cmd}" == *dronmakr-backend* ]] || [[ "${cmd}" == *backend_server.py* ]]; then
      echo "[dev-backend] stopping stale backend pid ${pid}" >&2
      kill "${pid}" 2>/dev/null || true
    fi
  done < <(listener_pids)

  sleep 1
  if listener_pids | grep -q .; then
    echo "[dev-backend] could not free port ${PORT}. Quit dronmakr.app or run: lsof -ti :${PORT} | xargs kill" >&2
    exit 1
  fi
fi

cd "${ROOT}"
exec python backend/backend_server.py --host "${HOST}" --port "${PORT}" --dev-frontend "$@"
