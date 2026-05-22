#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d "venv" ]]; then
  echo "venv/ not found. Create it first (python -m venv venv)."
  exit 1
fi

source "venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
python scripts/vendor_ffmpeg.py

UNAME_S="$(uname -s | tr '[:upper:]' '[:lower:]')"
if [[ "$UNAME_S" == "darwin" ]]; then
  if [[ ! -f "static/branding/macos/dronmakr.icns" ]]; then
    echo "error: static/branding/macos/dronmakr.icns is missing. Add the committed icon (see static/branding/macos/README.md)" >&2
    echo "  or run: bash scripts/build_mac_app_icns.sh" >&2
    exit 1
  fi
fi

if ! python -c "import tkinter" >/dev/null 2>&1; then
  echo "Warning: tkinter unavailable in this venv — Patchcraftr may be broken in the built app." >&2
  echo "  Fix: use Python from python.org, or brew install python-tk@MATCH (same major.minor as this venv's Python)." >&2
fi

pyinstaller --noconfirm --clean desktop.spec

# PySoundFile expects libsndfile_*.dylib next to frozen _soundfile_data; missing files give a useless cffi traceback.
if [[ "$UNAME_S" == "darwin" ]]; then
  SF_DIR="${ROOT_DIR}/dist/dronmakr.app/Contents/Frameworks/_soundfile_data"
  SF_COUNT="$(find "$SF_DIR" -maxdepth 1 -name 'libsndfile*.dylib' 2>/dev/null | wc -l | tr -d '[:space:]')"
  if [[ ! -d "$SF_DIR" || "${SF_COUNT:-0}" -lt 1 ]]; then
    echo "error: bundled libsndfile*.dylib missing under ${SF_DIR}; see build/desktop/warn-desktop.txt for sndfile hook warnings." >&2
    exit 1
  fi
fi

UNAME_M="$(uname -m | tr '[:upper:]' '[:lower:]')"
case "$UNAME_M" in
  arm64|aarch64) ARCH_LABEL="arm64" ;;
  x86_64|amd64) ARCH_LABEL="x64" ;;
  *) ARCH_LABEL="$UNAME_M" ;;
esac

VERSION="$(python -c 'from version import __version__; print(__version__)')"
ARTIFACT_DIR="dist-artifacts"
mkdir -p "$ARTIFACT_DIR"

# Names must include macos-arm64, macos-x64, linux-x64, or windows-x64 so the
# packaged app's updater (updater.py) can match GitHub release assets.
if [[ "$UNAME_S" == "darwin" ]]; then
  ARCHIVE_NAME="dronmakr-v${VERSION}-macos-${ARCH_LABEL}.tar.gz"
  if [[ ! -d "dist/dronmakr.app" ]]; then
    echo "error: dist/dronmakr.app missing after PyInstaller (desktop.spec should emit BUNDLE on macOS)." >&2
    exit 1
  fi
  tar -czf "${ARTIFACT_DIR}/${ARCHIVE_NAME}" -C "dist" "dronmakr.app"
  bash scripts/package_mac_dmg.sh
elif [[ "$UNAME_S" == "linux" ]]; then
  if [[ "$ARCH_LABEL" == "x64" ]]; then
    ARCHIVE_NAME="dronmakr-v${VERSION}-linux-x64.tar.gz"
  else
    ARCHIVE_NAME="dronmakr-v${VERSION}-linux-${ARCH_LABEL}.tar.gz"
  fi
  tar -czf "${ARTIFACT_DIR}/${ARCHIVE_NAME}" -C "dist" "dronmakr"
else
  echo "Unsupported OS for this script: ${UNAME_S} (use build_desktop.ps1 on Windows)"
  exit 1
fi

echo "Built artifact(s) under ${ARTIFACT_DIR}/"
ls -la "${ARTIFACT_DIR}"
