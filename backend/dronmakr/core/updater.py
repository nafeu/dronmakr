from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from dronmakr.version import __version__

GITHUB_RELEASES_API = "https://api.github.com/repos/nafeu/dronmakr/releases/latest"
GITHUB_RELEASES_LATEST_PAGE = "https://github.com/nafeu/dronmakr/releases/latest"


@dataclass
class UpdateInfo:
    tag: str
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


def check_for_update(timeout: int = 5) -> UpdateInfo | None:
    """Return release metadata when GitHub has a newer semver than this build."""
    resp = requests.get(GITHUB_RELEASES_API, timeout=timeout)
    if resp.status_code != 200:
        return None
    data = resp.json()
    tag = str(data.get("tag_name", "")).strip()
    if not tag:
        return None
    if _normalize_version(tag) <= _normalize_version(__version__):
        return None
    html_url = str(data.get("html_url", "")).strip() or release_page_url(tag)
    return UpdateInfo(tag=tag, release_url=html_url)


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
