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

pyinstaller --noconfirm --clean desktop.spec

PLATFORM="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m | tr '[:upper:]' '[:lower:]')"
VERSION="$(python -c 'from version import __version__; print(__version__)')"
ARTIFACT_DIR="dist-artifacts"
mkdir -p "$ARTIFACT_DIR"
ARCHIVE_NAME="dronmakr-v${VERSION}-${PLATFORM}-${ARCH}.tar.gz"

tar -czf "${ARTIFACT_DIR}/${ARCHIVE_NAME}" -C "dist" "dronmakr"
echo "Built artifact: ${ARTIFACT_DIR}/${ARCHIVE_NAME}"
