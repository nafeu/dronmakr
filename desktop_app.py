from __future__ import annotations

import os
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path

import webview

from settings import has_configured_files_root
from updater import check_for_update, download_update, reveal_file
from webui import start_server


def _find_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def main() -> None:
    port = _find_open_port()
    start_server(debug=False, port=port, host="127.0.0.1")
    time.sleep(1.0)

    window = webview.create_window(
        "dronmakr",
        f"http://127.0.0.1:{port}" if has_configured_files_root() else f"http://127.0.0.1:{port}/onboarding",
        width=1300,
        height=880,
    )

    def _startup_tasks() -> None:
        update = check_for_update()
        if not update:
            return
        should_update = window.create_confirmation_dialog(
            "Update available",
            f"A newer version ({update.tag}) is available. Download now?",
        )
        if not should_update:
            return
        target_dir = os.path.join(str(Path.home()), "Downloads", "dronmakr-updates")
        try:
            downloaded = download_update(update, target_dir)
            staged = os.path.join(target_dir, "staged")
            os.makedirs(staged, exist_ok=True)
            extracted_root = ""
            if downloaded.endswith(".zip"):
                with zipfile.ZipFile(downloaded, "r") as zf:
                    zf.extractall(staged)
                extracted_root = staged
            elif downloaded.endswith(".tar.gz"):
                with tarfile.open(downloaded, "r:gz") as tf:
                    tf.extractall(staged)
                extracted_root = staged
            executable = _find_staged_executable(extracted_root)
            if executable:
                _launch_executable(executable)
                os._exit(0)
            reveal_file(downloaded)
            window.create_alert_dialog("Update downloaded", "Package downloaded. Install manually, then relaunch.")
        except Exception as e:  # noqa: BLE001
            window.create_alert_dialog("Update failed", str(e))

    webview.start(_startup_tasks, storage_path=os.path.join(tempfile.gettempdir(), "dronmakr-webview"))


def _find_staged_executable(root: str) -> str:
    if not root or not os.path.exists(root):
        return ""
    target_name = "dronmakr.exe" if sys.platform == "win32" else "dronmakr"
    for dirpath, _dirnames, filenames in os.walk(root):
        if target_name in filenames:
            return os.path.join(dirpath, target_name)
    return ""


def _launch_executable(path: str) -> None:
    if sys.platform == "win32":
        subprocess.Popen([path], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path] if path.endswith(".app") else [path])
    else:
        subprocess.Popen([path])


if __name__ == "__main__":
    main()
