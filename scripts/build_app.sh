#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s | tr '[:upper:]' '[:lower:]')" == "linux" ]]; then
  exec bash scripts/build_linux.sh
fi

if [[ ! -d "venv" ]]; then
  echo "venv/ not found. Create it first (python -m venv venv)." >&2
  exit 1
fi

source "venv/bin/activate"
npm ci
npm run build

VERSION="$(PYTHONPATH=backend python -c 'from dronmakr.version import __version__; print(__version__)')"
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
  bash scripts/sign_mac_app.sh "$APP"
  tar -czf "${ARTIFACT_DIR}/${ARCHIVE_NAME}" -C "$(dirname "$APP")" "$(basename "$APP")"
  APP_PATH="$APP" bash scripts/package_mac_dmg.sh
elif [[ "$UNAME_S" == "linux" ]]; then
  echo "error: build_app.sh on Linux should have exec'd build_linux.sh" >&2
  exit 1
else
  echo "Use build_app.ps1 on Windows" >&2
  exit 1
fi

echo "Built artifacts under ${ARTIFACT_DIR}/"
ls -la "${ARTIFACT_DIR}"
