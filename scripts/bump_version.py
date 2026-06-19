#!/usr/bin/env python3
"""Bump app semver via scripts/bump_version.sh (post-Tauri layout)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "bump_version.sh"


def main() -> int:
    if not SCRIPT.is_file():
        print(f"error: {SCRIPT} not found", file=sys.stderr)
        return 1
    try:
        subprocess.run([str(SCRIPT), *sys.argv[1:]], cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
