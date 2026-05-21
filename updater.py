from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass

import requests

from version import __version__

GITHUB_RELEASES_API = "https://api.github.com/repos/nafeu/dronmakr/releases/latest"


@dataclass
class UpdateInfo:
    tag: str
    notes: str
    asset_name: str
    asset_url: str


def _normalize_version(value: str) -> tuple[int, ...]:
    cleaned = value.strip().lstrip("v")
    parts = []
    for chunk in cleaned.split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            break
    return tuple(parts)


def _platform_asset_hint() -> str:
    machine = platform.machine().lower()
    system = platform.system().lower()
    if "darwin" in system:
        return "macos-arm64" if "arm" in machine else "macos-x64"
    if "windows" in system:
        return "windows-x64"
    return "linux-x64"


def _asset_download_priority(asset_name: str) -> int:
    """
    Pick a sensible default when multiple release assets match the platform hint
    (e.g. macOS ships both .dmg and .tar.gz). Lower value = higher priority.
    """
    system = platform.system().lower()
    n = asset_name.lower()
    if "darwin" in system:
        if n.endswith(".dmg"):
            return 0
        if n.endswith(".tar.gz") or n.endswith(".tgz"):
            return 1
        if n.endswith(".zip"):
            return 2
        return 10
    if "windows" in system:
        if n.endswith(".zip"):
            return 0
        return 10
    if n.endswith(".tar.gz") or n.endswith(".tgz"):
        return 0
    return 10


def check_for_update(timeout: int = 5) -> UpdateInfo | None:
    resp = requests.get(GITHUB_RELEASES_API, timeout=timeout)
    if resp.status_code != 200:
        return None
    data = resp.json()
    tag = str(data.get("tag_name", "")).strip()
    if not tag:
        return None
    if _normalize_version(tag) <= _normalize_version(__version__):
        return None
    hint = _platform_asset_hint()
    assets = data.get("assets", []) if isinstance(data.get("assets"), list) else []
    notes = str(data.get("body", ""))
    matching: list[UpdateInfo] = []
    for asset in assets:
        name = str(asset.get("name", ""))
        url = str(asset.get("browser_download_url", ""))
        if hint in name.lower() and url:
            matching.append(
                UpdateInfo(tag=tag, notes=notes, asset_name=name, asset_url=url)
            )
    if not matching:
        return None
    matching.sort(
        key=lambda info: (_asset_download_priority(info.asset_name), info.asset_name)
    )
    return matching[0]


_UPDATE_CHECK_MIN_INTERVAL_S = 3600.0
_cached_update_info: UpdateInfo | None = None
_cached_update_check_monotonic: float = 0.0


def peek_cached_update_info() -> UpdateInfo | None:
    """Last result from :func:`fetch_update_info_throttled` without hitting the network."""
    return _cached_update_info


def fetch_update_info_throttled(
    force: bool = False,
    min_interval_s: float = _UPDATE_CHECK_MIN_INTERVAL_S,
    timeout: int = 5,
) -> UpdateInfo | None:
    """
    Query GitHub Releases at most once per ``min_interval_s`` unless ``force`` is True.
    Caches the latest result (including \"no update\") for tray menu state.
    """
    global _cached_update_info, _cached_update_check_monotonic
    now = time.monotonic()
    if not force and (now - _cached_update_check_monotonic) < min_interval_s:
        return _cached_update_info
    _cached_update_info = check_for_update(timeout=timeout)
    _cached_update_check_monotonic = now
    return _cached_update_info


def download_update(info: UpdateInfo, destination_dir: str) -> str:
    os.makedirs(destination_dir, exist_ok=True)
    target = os.path.join(destination_dir, info.asset_name)
    with requests.get(info.asset_url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        with open(target, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=1024 * 128):
                if chunk:
                    handle.write(chunk)
    return target


def reveal_file(path: str) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", path], check=False)
    elif sys.platform == "win32":
        subprocess.run(["explorer", f"/select,{path}"], check=False)
    else:
        subprocess.run(["xdg-open", os.path.dirname(path)], check=False)
