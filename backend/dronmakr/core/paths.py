from __future__ import annotations

import os
from pathlib import Path

from dronmakr.core.settings import ensure_managed_files_root, get_files_root, has_configured_files_root


def get_files_root_path(*, ensure: bool = False) -> Path:
    """Resolved FILES_ROOT path. Never falls back to ~/dronmakr-files."""
    root = get_files_root(allow_default=False)
    if not root:
        raise ValueError("FILES_ROOT is not configured")
    abs_root = os.path.abspath(root)
    if ensure:
        ensure_managed_files_root(abs_root)
    return Path(abs_root)


def get_managed_dir(name: str) -> str:
    root = get_files_root(allow_default=False)
    if not root:
        return ""
    return os.path.join(os.path.abspath(root), name)


def get_managed_file(*parts: str) -> str:
    root = get_files_root(allow_default=False)
    if not root:
        return ""
    return os.path.join(os.path.abspath(root), *parts)


def normalize_path_basename(path: str) -> str:
    """Final path segment for URLs and API tokens (Windows backslashes safe)."""
    if not path:
        return ""
    cleaned = str(path).strip().replace("\\", "/").rstrip("/")
    if not cleaned:
        return ""
    return os.path.basename(cleaned) or cleaned.split("/")[-1]


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def files_root_is_configured() -> bool:
    return has_configured_files_root()
