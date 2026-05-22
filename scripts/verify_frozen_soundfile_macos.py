#!/usr/bin/env python3
"""
Verify a frozen dronmakr.app bundles libsndfile where PySoundFile expects it (macOS).

Run after PyInstaller, before zipping / DMG. Used by build_desktop.sh and release-desktop.yml.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path


def _err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "app_bundle",
        type=Path,
        help="Path to dronmakr.app",
    )
    args = parser.parse_args()

    if sys.platform != "darwin":
        print("verify_frozen_soundfile_macos: skip (not macOS)")
        return 0

    app = args.app_bundle.expanduser().resolve()
    fw = app / "Contents" / "Frameworks" / "_soundfile_data"
    res = app / "Contents" / "Resources" / "_soundfile_data"
    exe = app / "Contents" / "MacOS" / "dronmakr"

    failed = False
    if not app.is_dir():
        _err(f"not a bundle: {app}")
        return 1
    if not exe.is_file():
        _err(f"missing executable: {exe}")
        failed = True
    if not fw.is_dir():
        _err(f"missing Frameworks dir: {fw}")
        failed = True
    if failed:
        return 1

    arch = platform.machine().strip().lower()
    arm = fw / "libsndfile_arm64.dylib"
    x86 = fw / "libsndfile_x86_64.dylib"
    generic = fw / "libsndfile.dylib"

    if arch in {"arm64", "aarch64"}:
        if not arm.is_file() and not (res / "libsndfile_arm64.dylib").is_file():
            _err(f"expected libsndfile_arm64.dylib under {fw} or Resources symlink")
            failed = True
    elif arch in {"x86_64", "amd64"}:
        if not x86.is_file() and not (res / "libsndfile_x86_64.dylib").is_file():
            _err(f"expected libsndfile_x86_64.dylib under {fw} or Resources symlink")
            failed = True

    # Always require generic name: frozen PySoundFile may use this when platform.machine() is ''.
    if not generic.is_file() and not (res / "libsndfile.dylib").is_file():
        _err(
            "missing libsndfile.dylib (desktop.spec must ship a copy for empty platform.machine()); "
            f"checked {fw} and {res}"
        )
        failed = True

    if failed:
        print(f"Listing {fw}:", file=sys.stderr)
        subprocess.run(["ls", "-la", str(fw)], check=False)
        return 1

    try:
        proc = subprocess.run(
            [str(exe), "--smoke-imports"],
            cwd=str(app.parent),
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


if __name__ == "__main__":
    raise SystemExit(main())
