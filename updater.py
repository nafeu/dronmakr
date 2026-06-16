from __future__ import annotations

import platform
import time
from dataclasses import dataclass

import requests

from version import __version__

GITHUB_RELEASES_API = "https://api.github.com/repos/nafeu/dronmakr/releases/latest"
GITHUB_RELEASES_LATEST_PAGE = "https://github.com/nafeu/dronmakr/releases/latest"


@dataclass
class UpdateInfo:
    tag: str
    notes: str
    asset_name: str
    release_url: str


def release_page_url(tag: str | None = None) -> str:
    if tag:
        cleaned = tag.strip().lstrip("v")
        return f"https://github.com/nafeu/dronmakr/releases/tag/v{cleaned}"
    return GITHUB_RELEASES_LATEST_PAGE


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
    html_url = str(data.get("html_url", "")).strip() or release_page_url(tag)
    matching: list[UpdateInfo] = []
    for asset in assets:
        name = str(asset.get("name", ""))
        if hint in name.lower():
            matching.append(
                UpdateInfo(
                    tag=tag,
                    notes=notes,
                    asset_name=name,
                    release_url=html_url,
                )
            )
    if not matching:
        return UpdateInfo(tag=tag, notes=notes, asset_name="", release_url=html_url)
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
    global _cached_update_info, _cached_update_check_monotonic
    now = time.monotonic()
    if not force and (now - _cached_update_check_monotonic) < min_interval_s:
        return _cached_update_info
    _cached_update_info = check_for_update(timeout=timeout)
    _cached_update_check_monotonic = now
    return _cached_update_info
