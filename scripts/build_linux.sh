#!/usr/bin/env bash
# Build dronmakr Linux x86_64 desktop bundles (deb + rpm + tar.gz).
# ARM64 Linux is not supported for release builds yet.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "error: build_linux.sh must run on Linux (use a VM or container for cross-arch builds)." >&2
  exit 1
fi

if [[ ! -d "venv" ]]; then
  echo "venv/ not found. Create it first: python3 -m venv venv" >&2
  exit 1
fi

UNAME_M="$(uname -m | tr '[:upper:]' '[:lower:]')"
case "$UNAME_M" in
  x86_64|amd64) ARCH_LABEL="x64" ;;
  aarch64|arm64)
    echo "error: Linux ARM64 release builds are not supported yet (x86_64 only)." >&2
    exit 1
    ;;
  *)
    echo "error: unsupported Linux machine type: ${UNAME_M}" >&2
    exit 1
    ;;
esac

source "venv/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt pyinstaller >/dev/null
npm ci
python scripts/vendor_ffmpeg.py
bash scripts/build_sidecar.sh
npm run tauri build -- --bundles deb,rpm

VERSION="$(PYTHONPATH=backend python -c 'from dronmakr.version import __version__; print(__version__)')"
BUNDLE_DIR="src-tauri/target/release/bundle"
if [[ ! -d "$BUNDLE_DIR" ]]; then
  echo "error: Tauri Linux bundle missing at ${BUNDLE_DIR}" >&2
  exit 1
fi

ARTIFACT_DIR="dist-artifacts"
mkdir -p "$ARTIFACT_DIR"
cp scripts/linux_release_readme.txt "${BUNDLE_DIR}/README-linux.txt"
ARCHIVE_NAME="dronmakr-v${VERSION}-linux-${ARCH_LABEL}-experimental.tar.gz"
tar -czf "${ARTIFACT_DIR}/${ARCHIVE_NAME}" -C "$BUNDLE_DIR" .

echo "Built ${ARCH_LABEL} Linux artifacts:"
echo "  archive: ${ARTIFACT_DIR}/${ARCHIVE_NAME}"
echo "  bundles: ${BUNDLE_DIR}/deb/ ${BUNDLE_DIR}/rpm/"
ls -la "${ARTIFACT_DIR}/${ARCHIVE_NAME}"
