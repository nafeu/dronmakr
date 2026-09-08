#!/usr/bin/env bash
# HTTPS dev server + WebXR shell for Quest passthrough testing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Regenerate self-signed cert so LAN IP is in SAN (Quest hits https://<lan-ip>:3767).
export DRONMAKR_DEV_TLS=1
export DRONMAKR_DEV_TLS_REGEN=1
export DRONMAKR_DEV_TLS_PORT="${DRONMAKR_DEV_TLS_PORT:-3767}"

echo "[dev-webxr] starting backend with sandbox TLS + WebXR shell"
echo "[dev-webxr] open https://<this-machine-lan-ip>:${DRONMAKR_DEV_TLS_PORT}/dev/webxr on Quest Browser"
echo "[dev-webxr] accept cert warning, tap Enter passthrough AR, allow WebXR when prompted"

exec bash "${ROOT}/scripts/dev_backend.sh" "$@"
