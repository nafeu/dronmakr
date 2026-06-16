"""Application bundle filesystem layout (frozen PyInstaller + dev source tree)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from dronmakr._repo import ASSETS_ROOT, FRONTEND_ROOT, REPO_ROOT, RESOURCES_ROOT


def get_bundle_app_root() -> Path:
    """
    Root directory holding bundled assets beside the runnable app.

    PyInstaller uses sys._MEIPASS; otherwise the repository root.
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", "") or ""
        if meipass:
            return Path(meipass)
    return REPO_ROOT


def get_frontend_dist_dir() -> Path:
    """Pre-built HTML pages served by the Flask UI."""
    if getattr(sys, "frozen", False):
        return get_bundle_app_root() / "frontend" / "dist"
    return FRONTEND_ROOT / "dist"


def get_static_dir() -> Path:
    """Web static assets (branding, vendor files not inlined in HTML)."""
    if getattr(sys, "frozen", False):
        return get_bundle_app_root() / "static"
    return ASSETS_ROOT / "static"


def bundled_asset_path(*relative_segments: str) -> Path:
    """
    Resolved path under resources/ in the bundle or repo (not CWD-relative).
    """
    if relative_segments and relative_segments[0] == "resources":
        segments = relative_segments[1:]
    else:
        segments = relative_segments
    if getattr(sys, "frozen", False):
        base = get_bundle_app_root() / "resources"
    else:
        base = RESOURCES_ROOT
    return base.joinpath(*segments)


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
