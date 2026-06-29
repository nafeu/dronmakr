#!/usr/bin/env python3
"""
Verify a frozen dronmakr sidecar or .app can import soundfile (macOS).

Run after PyInstaller sidecar build and again after Tauri packaging so the
executable inside the bundle is checked, not only the pre-bundle dist/ copy.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def _run_smoke_imports(exe: Path, work_dir: Path) -> int:
    try:
        proc = subprocess.run(
            [str(exe), "--smoke-imports"],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except OSError as e:
        _err(f"could not run {exe}: {e}")
        return 1

    if proc.returncode != 0:
        _err("--smoke-imports failed")
        sys.stderr.write(proc.stdout or "")
        sys.stderr.write(proc.stderr or "")
        return 1

    print(proc.stdout.strip() or "verify_frozen_soundfile_macos: OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        type=Path,
        help="Path to dronmakr.app or the dronmakr-backend sidecar executable",
    )
    args = parser.parse_args()

    if sys.platform != "darwin":
        print("verify_frozen_soundfile_macos: skip (not macOS)")
        return 0

    target = args.target.expanduser().resolve()
    if target.suffix == ".app":
        if not target.is_dir():
            _err(f"not a bundle: {target}")
            return 1
        exe = target / "Contents" / "Resources" / "resources" / "dronmakr-backend" / "dronmakr-backend"
        work_dir = exe.parent
    else:
        exe = target
        work_dir = exe.parent

    if not exe.is_file():
        _err(f"missing executable: {exe}")
        return 1

    return _run_smoke_imports(exe, work_dir)


if __name__ == "__main__":
    raise SystemExit(main())
