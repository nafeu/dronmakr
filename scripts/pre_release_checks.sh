#!/usr/bin/env bash
# Pre-release build verification. macOS-only checks run when executed on Darwin.
#
# Usage:
#   ./scripts/pre_release_checks.sh              # verify existing release build
#   ./scripts/pre_release_checks.sh --build      # npm run build + sign, then verify
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

APP="${APP:-src-tauri/target/release/bundle/macos/dronmakr.app}"
DO_BUILD=false

usage() {
  echo "Usage: $(basename "$0") [--build]"
  echo "  --build   Run npm run build and sign_mac_app.sh before macOS verification"
}

while (($#)); do
  case "$1" in
    --build)
      DO_BUILD=true
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "error: Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "pre_release_checks: skip macOS verification (not on macOS)"
  exit 0
fi

if [[ ! -d "venv" ]]; then
  echo "error: venv/ not found. Create it first (python -m venv venv)." >&2
  exit 1
fi

source "venv/bin/activate"

if [[ "$DO_BUILD" == true ]]; then
  echo "pre_release_checks: building release (frontend + sidecar + Tauri)..."
  npm run build
  bash scripts/sign_mac_app.sh "$APP"
elif [[ ! -d "$APP" ]]; then
  echo "error: macOS app bundle missing at ${APP}" >&2
  echo "Run with --build or build locally first (npm run build && scripts/sign_mac_app.sh)." >&2
  exit 1
fi

echo "pre_release_checks: verify frozen soundfile in bundled sidecar..."
python scripts/verify_frozen_soundfile_macos.py "$APP"

echo "pre_release_checks: verify macOS app backend launch (first-install HOME)..."
bash scripts/verify_macos_app_launch.sh "$APP"

echo "pre_release_checks: verify macOS app signature..."
codesign --verify --deep --strict --verbose=2 "$APP"

echo "pre_release_checks: macOS OK"
