#!/usr/bin/env bash
# Ad-hoc sign a Tauri macOS .app bundle so Gatekeeper does not report it as "damaged".
# CI builds are not notarized; users may still need Control-click → Open on first launch.
#
# Usage:
#   scripts/sign_mac_app.sh
#   scripts/sign_mac_app.sh path/to/dronmakr.app

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "sign_mac_app.sh: skip (not macOS)" >&2
  exit 0
fi

APP="${1:-src-tauri/target/release/bundle/macos/dronmakr.app}"
if [[ ! -d "$APP" ]]; then
  echo "error: ${APP} not found." >&2
  exit 1
fi

MACOS_DIR="${APP}/Contents/MacOS"

# Sign nested executables first, then the bundle (avoid codesign --deep when signing).
for bin in dronmakr-backend dronmakr; do
  if [[ -f "${MACOS_DIR}/${bin}" ]]; then
    codesign --force --sign - "${MACOS_DIR}/${bin}"
  fi
done

codesign --force --sign - "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"
echo "Signed and verified: ${APP}"
