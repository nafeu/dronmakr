#!/usr/bin/env bash
# Build packaging/macos/dronmakr.icns from PWA/app icon PNGs (macOS PyInstaller BUNDLE).
# Requires macOS (sips + iconutil). Safe to re-run; output is gitignored.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SRC="${ROOT_DIR}/static/branding/android-chrome-512x512.png"
OUT_DIR="${ROOT_DIR}/packaging/macos"
OUT_ICNS="${OUT_DIR}/dronmakr.icns"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: build_mac_app_icns.sh only runs on macOS (need sips + iconutil)." >&2
  exit 1
fi

if [[ ! -f "$SRC" ]]; then
  echo "error: missing source icon: $SRC" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
ICONSET="${OUT_ICNS}.iconset"
rm -rf "$ICONSET"
mkdir "$ICONSET"

# Standard macOS .iconset layout (see `man iconutil`).
sips -z 16 16 "$SRC" --out "$ICONSET/icon_16x16.png" >/dev/null
sips -z 32 32 "$SRC" --out "$ICONSET/icon_16x16@2x.png" >/dev/null
sips -z 32 32 "$SRC" --out "$ICONSET/icon_32x32.png" >/dev/null
sips -z 64 64 "$SRC" --out "$ICONSET/icon_32x32@2x.png" >/dev/null
sips -z 128 128 "$SRC" --out "$ICONSET/icon_128x128.png" >/dev/null
sips -z 256 256 "$SRC" --out "$ICONSET/icon_128x128@2x.png" >/dev/null
sips -z 256 256 "$SRC" --out "$ICONSET/icon_256x256.png" >/dev/null
sips -z 512 512 "$SRC" --out "$ICONSET/icon_256x256@2x.png" >/dev/null
sips -z 512 512 "$SRC" --out "$ICONSET/icon_512x512.png" >/dev/null
sips -z 1024 1024 "$SRC" --out "$ICONSET/icon_512x512@2x.png" >/dev/null

rm -f "$OUT_ICNS"
iconutil -c icns "$ICONSET" -o "$OUT_ICNS"
rm -rf "$ICONSET"

echo "Wrote $OUT_ICNS"
