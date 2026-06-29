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
BACKEND_DIR="${APP}/Contents/Resources/resources/dronmakr-backend"

if [[ -f "${BACKEND_DIR}/dronmakr-backend" ]]; then
  codesign --force --options runtime \
    --entitlements "${ROOT_DIR}/src-tauri/entitlements.plist" \
    --sign - "${BACKEND_DIR}/dronmakr-backend"
fi

INTERNAL="${BACKEND_DIR}/_internal"
if [[ -d "$INTERNAL" ]]; then
  while IFS= read -r -d '' f; do
    if file "$f" | grep -q 'Mach-O'; then
      codesign --force --options runtime \
        --entitlements "${ROOT_DIR}/src-tauri/entitlements.plist" \
        --sign - "$f" 2>/dev/null || true
    fi
  done < <(find "$INTERNAL" -type f \( -perm -111 -o -name '*.dylib' -o -name '*.so' \) -print0)
fi

if [[ -f "${MACOS_DIR}/dronmakr" ]]; then
  codesign --force --options runtime \
    --entitlements "${ROOT_DIR}/src-tauri/entitlements.plist" \
    --sign - "${MACOS_DIR}/dronmakr"
fi

codesign --force --sign - "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"
echo "Signed and verified: ${APP}"
