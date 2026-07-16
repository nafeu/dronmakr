#!/usr/bin/env bash
# Resolve packaged dronmakr binary on Linux.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUNDLE_DIR="${E2E_LINUX_BUNDLE_DIR:-${ROOT_DIR}/src-tauri/target/release/bundle}"

if [[ -n "${E2E_APP_BINARY:-}" && -x "${E2E_APP_BINARY}" ]]; then
  echo "${E2E_APP_BINARY}"
  exit 0
fi

candidates=(
  "${BUNDLE_DIR}/appimage/dronmakr"*.AppImage
  "${BUNDLE_DIR}/deb/dronmakr"*/usr/bin/dronmakr
  "${BUNDLE_DIR}/rpm/dronmakr"*/usr/bin/dronmakr
)

for pattern in "${candidates[@]}"; do
  for candidate in $pattern; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      exit 0
    fi
  done
done

echo "e2e: no Linux dronmakr binary under ${BUNDLE_DIR}" >&2
echo "e2e: build first (npm run build) or set E2E_APP_BINARY" >&2
exit 1
