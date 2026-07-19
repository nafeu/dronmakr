#!/usr/bin/env bash
# Apt packages for Tauri Linux desktop builds (deb/rpm). Run on Debian/Ubuntu.
set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
  echo "linux_build_deps.sh: apt-get not found (Debian/Ubuntu only)." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  curl \
  file \
  libayatana-appindicator3-dev \
  librsvg2-dev \
  libsndfile1 \
  libssl-dev \
  libwebkit2gtk-4.1-dev \
  libxdo-dev \
  patchelf \
  pkg-config \
  wget
