#!/usr/bin/env python3
"""Stage PyInstaller onedir output for Tauri bundle resources."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _onedir_dir(root: Path) -> Path:
    return root / "dist" / "dronmakr-backend"


def _sidecar_exe(root: Path) -> Path:
    onedir = _onedir_dir(root)
    if sys.platform == "win32":
        return onedir / "dronmakr-backend.exe"
    return onedir / "dronmakr-backend"


def stage_sidecar_onedir(root: Path | None = None) -> Path:
    root = root or _repo_root()
    onedir = _onedir_dir(root)
    exe = _sidecar_exe(root)
    internal = onedir / "_internal"
    if not exe.is_file():
        raise FileNotFoundError(f"PyInstaller onedir executable missing: {exe}")
    if not internal.is_dir():
        raise FileNotFoundError(f"PyInstaller onedir _internal/ missing: {internal}")

    dest = root / "src-tauri" / "resources" / "dronmakr-backend"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(onedir, dest)

    license_src = root / "LICENSE"
    if license_src.is_file():
        shutil.copy2(license_src, dest / "LICENSE")

    return dest


def main() -> int:
    dest = stage_sidecar_onedir()
    print(f"Sidecar onedir ready: {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
