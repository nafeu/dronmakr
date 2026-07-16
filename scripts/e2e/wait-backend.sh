#!/usr/bin/env bash
# Wait until dronmakr backend answers /api/health.
set -euo pipefail

HOST="${E2E_HOST:-127.0.0.1}"
TIMEOUT_S="${E2E_HEALTH_TIMEOUT:-180}"
PORTS="${E2E_PORTS:-3766 3767 3768 3769}"

deadline=$((SECONDS + TIMEOUT_S))
while ((SECONDS < deadline)); do
  for port in $PORTS; do
    url="http://${HOST}:${port}/api/health"
    if curl -fsS "$url" >/dev/null 2>&1; then
      export E2E_PORT="$port"
      export E2E_BASE_URL="http://${HOST}:${port}"
      echo "e2e: backend ready at ${url}"
      exit 0
    fi
  done
  sleep 1
done

echo "e2e: backend not ready after ${TIMEOUT_S}s (ports: ${PORTS})" >&2
exit 1
