#!/usr/bin/env bash
# Self-signed cert for local HTTPS dev sandbox (getUserMedia needs secure context).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CERT_DIR="${ROOT}/.dev-certs"
CRT="${CERT_DIR}/dev.crt"
KEY="${CERT_DIR}/dev.key"

if [[ "${DRONMAKR_DEV_TLS_REGEN:-}" == "1" && -f "${CRT}" ]]; then
  rm -f "${CRT}" "${KEY}"
fi

if [[ -f "${CRT}" && -f "${KEY}" ]]; then
  exit 0
fi

if ! command -v openssl >/dev/null 2>&1; then
  echo "[dev-tls] openssl required to generate ${CERT_DIR}/" >&2
  exit 1
fi

LAN_IP=""
if command -v ipconfig >/dev/null 2>&1; then
  LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)"
fi

SAN="DNS:localhost,DNS:dronmakr-dev,IP:127.0.0.1"
if [[ -n "${LAN_IP}" ]]; then
  SAN="${SAN},IP:${LAN_IP}"
fi

mkdir -p "${CERT_DIR}"
if openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout "${KEY}" -out "${CRT}" -days 825 \
  -subj "/CN=dronmakr-dev" \
  -addext "subjectAltName=${SAN}" 2>/dev/null; then
  :
else
  openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "${KEY}" -out "${CRT}" -days 825 \
    -subj "/CN=dronmakr-dev"
fi

echo "[dev-tls] wrote ${CRT} (self-signed — accept cert warning in sandbox browser)"
