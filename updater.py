from __future__ import annotations

import os
import platform
import subprocess
import sys
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
    for asset in assets:
        name = str(asset.get("name", ""))
        url = str(asset.get("browser_download_url", ""))
        if hint in name.lower() and url:
            return UpdateInfo(
                tag=tag,
                notes=str(data.get("body", "")),
                asset_name=name,
                asset_url=url,
            )
    return None


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
