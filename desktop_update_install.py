"""Download / extract desktop release artifacts (shared by tray + Tk updater UI)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from updater import UpdateInfo, download_update, reveal_file


def _find_executable_in_tree(root: str) -> str:
    target_name = "dronmakr.exe" if sys.platform == "win32" else "dronmakr"
    for dirpath, _dirnames, filenames in os.walk(root):
        if target_name in filenames:
            return os.path.join(dirpath, target_name)
    return ""


def _find_mac_app_bundle(root: str) -> str:
    """Return path to dronmakr.app if present under extracted staging."""
    for dirpath, dirnames, _filenames in os.walk(root):
        if "dronmakr.app" in dirnames:
            candidate = os.path.join(dirpath, "dronmakr.app")
            if os.path.isdir(candidate):
                return candidate
    return ""


def launch_updated_build(path: str) -> None:
    """Open a frozen dronmakr binary or macOS app bundle."""
    if sys.platform == "win32":
        subprocess.Popen(
            [path],
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # type: ignore[attr-defined]
        )
    elif sys.platform == "darwin":
        if path.endswith(".app"):
            subprocess.Popen(["open", "-n", "-a", path])
        else:
            subprocess.Popen([path])
    else:
        subprocess.Popen([path])


def extract_archive_to_staged(downloaded: str, staged: str) -> None:
    os.makedirs(staged, exist_ok=True)
    shutil.rmtree(staged, ignore_errors=True)
    os.makedirs(staged, exist_ok=True)
    if downloaded.endswith(".zip"):
        with zipfile.ZipFile(downloaded, "r") as zf:
            zf.extractall(staged)
    elif downloaded.endswith(".tar.gz") or downloaded.endswith(".tgz"):
        with tarfile.open(downloaded, "r:gz") as tf:
            tf.extractall(staged)
    else:
        raise ValueError(f"unsupported archive format: {downloaded}")


def apply_download_and_launch(
    update: UpdateInfo,
    download_root: str | None = None,
) -> tuple[str, bool]:
    """
    Download GitHub asset, unpack if needed, launch .app/exe when found.

    Returns (message_for_user, launched_new_build).
    """
    base = download_root or os.path.join(str(Path.home()), "Downloads", "dronmakr-updates")
    os.makedirs(base, exist_ok=True)

    try:
        downloaded = download_update(update, base)
    except OSError as e:
        return f"Download failed: {e}", False

    if downloaded.lower().endswith(".dmg"):
        subprocess.Popen(["open", downloaded], start_new_session=True)
        return (
            "Opened the update disk image. Drag dronmakr.app into Applications (replace the old copy), "
            "eject the volume, then relaunch dronmakr from Applications.",
            False,
        )

    staged = os.path.join(base, "staged")
    try:
        extract_archive_to_staged(downloaded, staged)
    except (OSError, ValueError, zipfile.BadZipFile, tarfile.TarError) as e:
        reveal_file(downloaded)
        return f"Could not extract automatically ({e}). The package was revealed in Finder/Explorer.", False

    if sys.platform == "darwin":
        app_b = _find_mac_app_bundle(staged)
        if app_b:
            launch_updated_build(app_b)
            return (
                "Launched the downloaded dronmakr.app.\nQuit this older menu bar instance when you’re done migrating.",
                True,
            )

    exe = _find_executable_in_tree(staged)
    if exe:
        launch_updated_build(exe)
        return (
            "Launched the downloaded build.\nQuit this older instance when finished.",
            True,
        )

    reveal_file(downloaded)
    return (
        "Update downloaded but no dronmakr executable was found in the archive. "
        "The file was revealed so you can install manually.",
        False,
    )
