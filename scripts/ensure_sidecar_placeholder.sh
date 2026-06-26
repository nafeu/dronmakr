#!/usr/bin/env bash
# Install a dev placeholder sidecar when no PyInstaller build is present.
# Real one-file sidecars (~120MB) are gitignored and produced by build_sidecar.sh / CI.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET="$(rustc --print host-tuple)"
BIN_DIR="src-tauri/binaries"
DEST="${BIN_DIR}/dronmakr-backend-${TARGET}"
PLACEHOLDER="${BIN_DIR}/placeholders/dronmakr-backend-${TARGET}"

if [[ ! -f "$PLACEHOLDER" ]]; then
  echo "ensure_sidecar_placeholder: no placeholder for ${TARGET} (skip)" >&2
  exit 0
fi

if [[ -f "$DEST" ]]; then
  size="$(wc -c < "$DEST" | tr -d ' ')"
  # PyInstaller one-file sidecars are tens of MB; placeholders are tiny shell scripts.
  if [[ "$size" -gt 1048576 ]]; then
    exit 0
  fi
fi

mkdir -p "$BIN_DIR"
cp "$PLACEHOLDER" "$DEST"
chmod +x "$DEST"
echo "Installed sidecar placeholder: ${DEST}"
