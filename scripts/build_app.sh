#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source "venv/bin/activate"
npm run build

VERSION="$(python -c 'from version import __version__; print(__version__)')"
ARTIFACT_DIR="dist-artifacts"
mkdir -p "$ARTIFACT_DIR"

UNAME_S="$(uname -s | tr '[:upper:]' '[:lower:]')"
UNAME_M="$(uname -m | tr '[:upper:]' '[:lower:]')"
case "$UNAME_M" in
  arm64|aarch64) ARCH_LABEL="arm64" ;;
  x86_64|amd64) ARCH_LABEL="x64" ;;
  *) ARCH_LABEL="$UNAME_M" ;;
esac

if [[ "$UNAME_S" == "darwin" ]]; then
  APP="src-tauri/target/release/bundle/macos/dronmakr.app"
  if [[ ! -d "$APP" ]]; then
    echo "error: Tauri macOS bundle missing at $APP" >&2
    exit 1
  fi
  ARCHIVE_NAME="dronmakr-v${VERSION}-macos-${ARCH_LABEL}.tar.gz"
  tar -czf "${ARTIFACT_DIR}/${ARCHIVE_NAME}" -C "$(dirname "$APP")" "$(basename "$APP")"
  APP_PATH="$APP" bash scripts/package_mac_dmg.sh
elif [[ "$UNAME_S" == "linux" ]]; then
  ARCHIVE_NAME="dronmakr-v${VERSION}-linux-${ARCH_LABEL}.tar.gz"
  BUNDLE_DIR="src-tauri/target/release/bundle/appimage"
  if compgen -G "${BUNDLE_DIR}/*.AppImage" > /dev/null; then
    cp ${BUNDLE_DIR}/*.AppImage "${ARTIFACT_DIR}/"
  fi
  DEB_DIR="src-tauri/target/release/bundle/deb"
  if compgen -G "${BUNDLE_DIR}/.." > /dev/null; then
    tar -czf "${ARTIFACT_DIR}/${ARCHIVE_NAME}" -C "src-tauri/target/release/bundle" . || true
  fi
else
  echo "Use build_app.ps1 on Windows" >&2
  exit 1
fi

echo "Built artifacts under ${ARTIFACT_DIR}/"
ls -la "${ARTIFACT_DIR}"
