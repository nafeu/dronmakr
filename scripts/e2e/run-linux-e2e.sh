#!/usr/bin/env bash
# Generic Linux E2E — auto-labels ubuntu/arch when detectable.
set -euo pipefail

if [[ -z "${E2E_LINUX_LABEL:-}" ]]; then
  if [[ -f /etc/arch-release ]]; then
    export E2E_LINUX_LABEL=arch
  elif [[ -f /etc/os-release ]] && grep -qi '^ID=ubuntu' /etc/os-release; then
    export E2E_LINUX_LABEL=ubuntu
  else
    export E2E_LINUX_LABEL=linux
  fi
fi

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_linux-e2e-core.sh" "$@"
