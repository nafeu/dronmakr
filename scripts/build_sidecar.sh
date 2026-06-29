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

ONEDIR_EXE="dist/dronmakr-backend/dronmakr-backend"
if [[ "$(uname -s)" == "Darwin" ]]; then
  python scripts/verify_frozen_soundfile_macos.py "$ONEDIR_EXE"
fi

python scripts/stage_sidecar_onedir_dist.py
echo "Sidecar onedir staged under src-tauri/resources/dronmakr-backend/ (gitignored)."
