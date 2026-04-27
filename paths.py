from __future__ import annotations

import os
from pathlib import Path

from settings import ensure_managed_files_root, get_files_root


def get_files_root_path() -> Path:
    return Path(ensure_managed_files_root(get_files_root(allow_default=True)))


def get_managed_dir(name: str) -> str:
    return str(get_files_root_path() / name)


def get_managed_file(*parts: str) -> str:
    return str(get_files_root_path().joinpath(*parts))


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
