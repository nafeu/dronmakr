#!/usr/bin/env bash
# Arch Linux VM E2E — run inside guest after one-time setup (docs/e2e-vm-setup.md).
set -euo pipefail
export E2E_LINUX_LABEL=arch
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_linux-e2e-core.sh" "$@"
