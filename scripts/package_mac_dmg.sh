#!/usr/bin/env bash
# Build a drag-to-Applications DMG from the Tauri macOS .app bundle.
# Writes dist-artifacts/dronmakr-v<version>-macos-<arch>.dmg

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

APP="${APP:-src-tauri/target/release/bundle/macos/dronmakr.app}"
if [[ ! -d "$APP" ]]; then
  echo "error: ${APP} not found. Run the Tauri release build first." >&2
  exit 1
fi

VERSION="$(PYTHONPATH=backend python3 -c "from dronmakr.version import __version__; print(__version__)")"
case "$(uname -m)" in
  arm64|aarch64) ARCH_LABEL="arm64" ;;
  x86_64|amd64) ARCH_LABEL="x64" ;;
  *) ARCH_LABEL="$(uname -m)" ;;
esac

mkdir -p dist-artifacts
DMG_PATH="dist-artifacts/dronmakr-v${VERSION}-macos-${ARCH_LABEL}.dmg"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/dronmakr-dmg.XXXXXX")"
cleanup() {
  rm -rf "$STAGE"
}
trap cleanup EXIT

ditto "$APP" "${STAGE}/dronmakr.app"
bash "$ROOT_DIR/scripts/sign_mac_app.sh" "${STAGE}/dronmakr.app"
ln -sf /Applications "${STAGE}/Applications"

rm -f "$DMG_PATH"
hdiutil create \
  -volname "dronmakr ${VERSION}" \
  -srcfolder "$STAGE" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

echo "Built DMG: ${DMG_PATH}"
