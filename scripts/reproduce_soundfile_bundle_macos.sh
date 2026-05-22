#!/usr/bin/env bash
# Diagnose frozen PySoundFile / libsndfile layout on macOS, and optionally produce a broken
# copy of dronmakr.app that crashes the same way as a missing-wheel dylib in /Applications.
#
# Usage:
#   bash scripts/reproduce_soundfile_bundle_macos.sh diagnose [path/to/dronmakr.app]
#   bash scripts/reproduce_soundfile_bundle_macos.sh break-copy [path/to/dronmakr.app]
#
# Default path: dist/dronmakr.app (repo root-relative if not absolute).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cmd="${1:-}"
if [[ "$cmd" != "diagnose" && "$cmd" != "break-copy" ]]; then
  cat >&2 <<'EOF'
Usage:
  bash scripts/reproduce_soundfile_bundle_macos.sh diagnose   [path/to/dronmakr.app]
  bash scripts/reproduce_soundfile_bundle_macos.sh break-copy [path/to/dronmakr.app]

  diagnose    — print arch, expected packaged dylib name, and ls of Frameworks/_soundfile_data/
  break-copy  — ditto-copy the .app into $TMPDIR, delete libsndfile*.dylib, print run command

Examples (from repo root, after pyinstaller):

  bash scripts/reproduce_soundfile_bundle_macos.sh diagnose dist/dronmakr.app
  bash scripts/reproduce_soundfile_bundle_macos.sh diagnose /Applications/dronmakr.app

  bash scripts/reproduce_soundfile_bundle_macos.sh break-copy dist/dronmakr.app
  "$TMPDIR/dronmakr-soundfile-repro.app/Contents/MacOS/dronmakr"

Why this matches production errors:

  • PySoundFile first dlopen()'s Frameworks/_soundfile_data/libsndfile_<platform.machine>.dylib
    (Apple Silicon → libsndfile_arm64.dylib).
  • If that file is missing or unloadable, it falls through to ctypes find_library('sndfile') and
    then tries libsndfile.dylib — your traceback ends with Frameworks/_soundfile_data/libsndfile.dylib
    (that generic name usually is NOT shipped in the wheel; the arch-named file is).
EOF
  exit 1
fi

APP="${2:-}"
if [[ -z "$APP" ]]; then
  APP="${ROOT_DIR}/dist/dronmakr.app"
else
  if [[ "$APP" != /* ]]; then
    APP="${ROOT_DIR}/${APP}"
  fi
fi

if [[ ! -d "$APP" ]]; then
  echo "error: not a directory: ${APP}" >&2
  exit 1
fi

SF_DIR="${APP}/Contents/Frameworks/_soundfile_data"
MACHINE="$(uname -m)"
case "$MACHINE" in
  arm64) EXPECTED="libsndfile_arm64.dylib" ;;
  x86_64) EXPECTED="libsndfile_x86_64.dylib" ;;
  *) EXPECTED="libsndfile_${MACHINE}.dylib" ;;
esac

if [[ "$cmd" == "diagnose" ]]; then
  echo "App bundle:     $APP"
  echo "Host uname -m:  $MACHINE"
  echo "Packaged file:  $EXPECTED  (first choice for PySoundFile on this Mac)"
  echo ""
  echo "Contents of:    $SF_DIR"
  if [[ ! -d "$SF_DIR" ]]; then
    echo "  (directory missing — import soundfile in the frozen app will fail.)"
    exit 1
  fi
  ls -la "$SF_DIR"
  echo ""
  if [[ -f "${SF_DIR}/${EXPECTED}" ]]; then
    echo "OK: ${EXPECTED} is present."
  else
    echo "PROBLEM: ${EXPECTED} is missing — this host’s Python would not load the wheel copy."
    echo "         (Intel Mac needs an x86_64 build + libsndfile_x86_64.dylib; arm64 needs arm64.)"
  fi
  exit 0
fi

# break-copy
TMP_ROOT="${TMPDIR:-/tmp}"
DEST="${TMP_ROOT}/dronmakr-soundfile-repro.app"
rm -rf "$DEST"
ditto "$APP" "$DEST"
BROKEN_SF="${DEST}/Contents/Frameworks/_soundfile_data"
if [[ ! -d "$BROKEN_SF" ]]; then
  echo "error: copy has no ${BROKEN_SF}; cannot strip dylibs." >&2
  exit 1
fi
shopt -s nullglob
for f in "${BROKEN_SF}"/libsndfile*.dylib; do
  rm -f "$f"
done
shopt -u nullglob

echo "Broken bundle (libsndfile*.dylib removed from _soundfile_data):"
echo "  ${DEST}"
echo ""
echo "Run from Terminal (stderr visible):"
echo "  \"${DEST}/Contents/MacOS/dronmakr\""
echo ""
echo "Expect a traceback ending with dlopen(.../libsndfile.dylib) once PySoundFile exhausts fallbacks."
