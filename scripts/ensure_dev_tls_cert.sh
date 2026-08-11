#!/usr/bin/env bash
# Self-signed cert for local HTTPS dev (Quest browser mic needs secure context).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CERT_DIR="${ROOT}/.dev-certs"
CRT="${CERT_DIR}/dev.crt"
KEY="${CERT_DIR}/dev.key"

if [[ -f "${CRT}" && -f "${KEY}" ]]; then
  exit 0
fi

if ! command -v openssl >/dev/null 2>&1; then
  echo "[dev-tls] openssl required to generate ${CERT_DIR}/" >&2
  exit 1
fi

mkdir -p "${CERT_DIR}"
if openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout "${KEY}" -out "${CRT}" -days 825 \
  -subj "/CN=dronmakr-dev" \
  -addext "subjectAltName=DNS:localhost,DNS:dronmakr-dev,IP:127.0.0.1" 2>/dev/null; then
  :
else
  openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "${KEY}" -out "${CRT}" -days 825 \
    -subj "/CN=dronmakr-dev"
fi

echo "[dev-tls] wrote ${CRT} (self-signed — accept cert warning in Quest browser)"
