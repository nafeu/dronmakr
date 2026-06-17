#!/usr/bin/env bash
# Compose a macOS-style 1024² icon layer (Pillow) and build OS / Tauri bundle icons only.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SRC="${ROOT_DIR}/assets/static/branding/logo.png"
OUT_DIR="${ROOT_DIR}/assets/static/branding/macos"
OUT_ICNS="${OUT_DIR}/dronmakr.icns"
TAURI_ICONS="${ROOT_DIR}/src-tauri/icons"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: build_mac_app_icns.sh only runs on macOS (need sips + iconutil)." >&2
  exit 1
fi

if [[ ! -f "$SRC" ]]; then
  echo "error: missing source icon: $SRC" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$TAURI_ICONS"

LAYER_1024="$(mktemp "${TMPDIR:-/tmp}/dronmakr-icns-layer.XXXXXX")"
cleanup() {
  rm -f "$LAYER_1024"
}
trap cleanup EXIT

echo "Composing macOS icon layer (Pillow)…"
python3 "${ROOT_DIR}/scripts/compose_mac_app_icon_layer.py" --source "$SRC" --out "$LAYER_1024"

ICONSET="${OUT_ICNS}.iconset"
rm -rf "$ICONSET"
mkdir "$ICONSET"

# Standard macOS .iconset layout (see `man iconutil`).
sips -z 16 16 "$LAYER_1024" --out "$ICONSET/icon_16x16.png" >/dev/null
sips -z 32 32 "$LAYER_1024" --out "$ICONSET/icon_16x16@2x.png" >/dev/null
sips -z 32 32 "$LAYER_1024" --out "$ICONSET/icon_32x32.png" >/dev/null
sips -z 64 64 "$LAYER_1024" --out "$ICONSET/icon_32x32@2x.png" >/dev/null
sips -z 128 128 "$LAYER_1024" --out "$ICONSET/icon_128x128.png" >/dev/null
sips -z 256 256 "$LAYER_1024" --out "$ICONSET/icon_128x128@2x.png" >/dev/null
sips -z 256 256 "$LAYER_1024" --out "$ICONSET/icon_256x256.png" >/dev/null
sips -z 512 512 "$LAYER_1024" --out "$ICONSET/icon_256x256@2x.png" >/dev/null
sips -z 512 512 "$LAYER_1024" --out "$ICONSET/icon_512x512.png" >/dev/null
sips -z 1024 1024 "$LAYER_1024" --out "$ICONSET/icon_512x512@2x.png" >/dev/null

rm -f "$OUT_ICNS"
iconutil -c icns "$ICONSET" -o "$OUT_ICNS"
rm -rf "$ICONSET"

cp "$OUT_ICNS" "${TAURI_ICONS}/icon.icns"
cp "$LAYER_1024" "${TAURI_ICONS}/icon.png"
sips -z 32 32 "$LAYER_1024" --out "${TAURI_ICONS}/32x32.png" >/dev/null
sips -z 128 128 "$LAYER_1024" --out "${TAURI_ICONS}/128x128.png" >/dev/null
sips -z 256 256 "$LAYER_1024" --out "${TAURI_ICONS}/128x128@2x.png" >/dev/null

python3 - "$LAYER_1024" "$TAURI_ICONS" <<'PY'
import sys
from pathlib import Path
from PIL import Image

layer_path = Path(sys.argv[1])
tauri_icons = Path(sys.argv[2])
layer = Image.open(layer_path).convert("RGBA")

ico_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
ico_images = [layer.resize(size, Image.Resampling.LANCZOS) for size in ico_sizes]
ico_images[0].save(tauri_icons / "icon.ico", format="ICO", sizes=[im.size for im in ico_images])
print(f"Wrote {tauri_icons / 'icon.ico'}")
PY

echo "Wrote:"
echo "  $OUT_ICNS"
echo "  ${TAURI_ICONS}/icon.icns"
echo "  ${TAURI_ICONS}/icon.png"
echo "  ${TAURI_ICONS}/32x32.png"
echo "  ${TAURI_ICONS}/128x128.png"
echo "  ${TAURI_ICONS}/128x128@2x.png"
echo "  ${TAURI_ICONS}/icon.ico"
echo ""
echo "Rebuild the Tauri binary so macOS picks up the new Dock icon:"
echo "  cd src-tauri && cargo build"
echo "Or restart: npm run dev (after the rebuild completes once)."
