#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d "venv" ]]; then
  echo "venv/ not found. Create it first (python -m venv venv)." >&2
  exit 1
fi

source "venv/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt pyinstaller >/dev/null
python scripts/build_frontend.py
python scripts/vendor_ffmpeg.py
pyinstaller --noconfirm --clean backend/backend.spec

if [[ "$(uname -s)" == "Darwin" ]]; then
  python scripts/verify_frozen_soundfile_macos.py "dist/dronmakr-backend/dronmakr-backend"
fi

TARGET="$(rustc --print host-tuple)"
BIN_DIR="src-tauri/binaries"
mkdir -p "$BIN_DIR"

if [[ "$(uname -s)" == "Darwin" ]]; then
  SRC="dist/dronmakr-backend/dronmakr-backend"
else
  SRC="dist/dronmakr-backend/dronmakr-backend"
  if [[ "$(uname -s)" == "MINGW"* ]] || [[ "$(uname -s)" == "MSYS"* ]] || [[ -f "dist/dronmakr-backend/dronmakr-backend.exe" ]]; then
    SRC="dist/dronmakr-backend/dronmakr-backend.exe"
  fi
fi

if [[ ! -f "$SRC" ]]; then
  echo "error: PyInstaller output missing at $SRC" >&2
  exit 1
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  codesign --force --sign - "$SRC"
fi

DEST="$BIN_DIR/dronmakr-backend-${TARGET}"
cp "$SRC" "$DEST"
chmod +x "$DEST"
echo "Sidecar ready: $DEST"
