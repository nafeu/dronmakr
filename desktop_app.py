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


def _find_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def main() -> None:
    print("[desktop] launcher: startup begin", flush=True)
    port = _find_open_port()
    print(f"[desktop] launcher: selected port {port}", flush=True)
    server_process = _start_backend_subprocess(port)
    base_url = f"http://127.0.0.1:{port}"
    launch_url = f"{base_url}" if has_configured_files_root() else f"{base_url}/onboarding"
    print("[desktop] launcher: waiting for backend readiness", flush=True)
    ready = _wait_for_port("127.0.0.1", port, timeout_s=30.0)

    if not ready or server_process.poll() is not None:
        print("[desktop] launcher: backend not ready, opening diagnostic window", flush=True)
        window = webview.create_window(
            "dronmakr",
            html=(
                "<html><body style='background:#111;color:#ff9505;font-family:monospace;padding:16px'>"
                "<h3>dronmakr backend did not start</h3>"
                "<p>Check terminal logs above for backend startup errors.</p>"
                f"<p>Backend process running: {server_process.poll() is None}</p>"
                "</body></html>"
            ),
            width=980,
            height=520,
        )
        webview.start(storage_path=os.path.join(tempfile.gettempdir(), "dronmakr-webview"))
        _stop_backend_subprocess(server_process)
        return

    print(f"[desktop] launcher: creating window at {launch_url}", flush=True)
    window = webview.create_window("dronmakr", launch_url, width=1300, height=880)

    def _startup_tasks() -> None:
        if not getattr(sys, "frozen", False):
            return
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

    try:
        webview.start(
            _startup_tasks,
            storage_path=os.path.join(tempfile.gettempdir(), "dronmakr-webview"),
        )
    finally:
        _stop_backend_subprocess(server_process)


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


def _start_backend_subprocess(port: int) -> subprocess.Popen:
    repo_root = str(Path(__file__).resolve().parent)
    command = [
        sys.executable,
        "dronmakr.py",
        "webui",
        "--port",
        str(int(port)),
        "--host",
        "127.0.0.1",
        "--no-open-browser",
    ]
    print(f"[desktop] launcher: spawning backend {' '.join(command)}", flush=True)
    return subprocess.Popen(command, cwd=repo_root)


def _wait_for_port(host: str, port: int, timeout_s: float = 20.0) -> bool:
    deadline = time.time() + max(1.0, timeout_s)
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            with socket.create_connection((host, int(port)), timeout=1.0):
                print(f"[desktop] launcher: backend ready on attempt {attempt}", flush=True)
                return True
        except OSError:
            if attempt % 10 == 0:
                print(f"[desktop] launcher: waiting... attempt {attempt}", flush=True)
            time.sleep(0.2)
    return False


def _stop_backend_subprocess(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    print("[desktop] launcher: stopping backend process", flush=True)
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


if __name__ == "__main__":
    main()
