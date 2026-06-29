#!/usr/bin/env bash
# Install dev placeholders when no PyInstaller sidecar build is present.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RES_ONEDIR="${ROOT_DIR}/src-tauri/resources/dronmakr-backend"

_internal_is_real() {
  local dir="$1"
  [[ -d "${dir}/_internal" ]] || return 1
  local count
  count="$(find "${dir}/_internal" -type f 2>/dev/null | wc -l | tr -d ' ')"
  [[ "$count" -gt 5 ]]
}

if ! _internal_is_real "$RES_ONEDIR"; then
  rm -rf "$RES_ONEDIR"
  mkdir -p "$RES_ONEDIR/_internal"
  echo "PyInstaller onedir placeholder. Run scripts/build_sidecar.sh for release builds." \
    > "${RES_ONEDIR}/README.txt"
  echo "Installed sidecar resources placeholder: ${RES_ONEDIR}"
fi
