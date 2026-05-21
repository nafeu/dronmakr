"""Application bundle filesystem layout (frozen PyInstaller + dev source tree)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def get_bundle_app_root() -> Path:
    """
    Root directory holding templates/static/resources beside the runnable app.

    Mirrors desktop_app._bundle_root: PyInstaller uses sys._MEIPASS; otherwise repo root.
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", "") or ""
        if meipass:
            return Path(meipass)
    return Path(__file__).resolve().parent


def resolve_ffmpeg_executable() -> Path | None:
    """Bundled FFmpeg (desktop) wins; then DRONMAKR_FFMPEG_PATH; then ffmpeg on PATH."""
    override = (os.environ.get("DRONMAKR_FFMPEG_PATH") or "").strip()
    if override:
        p = Path(os.path.expanduser(override)).resolve()
        if p.is_file():
            return p

    root = get_bundle_app_root()
    exe_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    bundled = root / "resources" / "ffmpeg" / exe_name
    if bundled.is_file():
        return bundled

    which = shutil.which("ffmpeg")
    if which:
        return Path(which).resolve()

    return None
