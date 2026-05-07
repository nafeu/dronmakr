from __future__ import annotations

import os
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image
from pystray import Icon, Menu, MenuItem

from settings import has_configured_files_root
from updater import UpdateInfo, check_for_update, download_update, reveal_file
from webui import open_webui_in_browser, start_server

_HEALTH_PATH = "/api/health"
_ICON_REL = Path("static") / "branding" / "favicon-32x32.png"


def _find_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _bundle_root() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            return Path(meipass)
    return Path(__file__).resolve().parent


def _load_tray_image() -> Image.Image:
    path = _bundle_root() / _ICON_REL
    if path.is_file():
        with Image.open(path) as im:
            return im.copy()
    img = Image.new("RGBA", (64, 64), (255, 149, 5, 255))
    return img


def _wait_for_health(base_url: str, timeout_s: float = 30.0) -> bool:
    health = base_url.rstrip("/") + _HEALTH_PATH
    deadline = time.time() + max(1.0, timeout_s)
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            with urllib.request.urlopen(health, timeout=2.0) as resp:
                if resp.getcode() == 200:
                    print(f"[desktop] launcher: health OK on attempt {attempt}", flush=True)
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            if attempt % 10 == 0:
                print(f"[desktop] launcher: waiting for health... attempt {attempt}", flush=True)
            time.sleep(0.2)
    return False


def _tk_root():
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    return root


def _messagebox_error(title: str, message: str) -> None:
    try:
        from tkinter import messagebox

        root = _tk_root()
        try:
            messagebox.showerror(title, message)
        finally:
            root.destroy()
    except Exception:  # noqa: BLE001
        print(f"[desktop] {title}: {message}", flush=True)


def _messagebox_askyesno(title: str, message: str) -> bool:
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    try:
        return bool(messagebox.askyesno(title, message))
    finally:
        root.destroy()


def _messagebox_showinfo(title: str, message: str) -> None:
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    try:
        messagebox.showinfo(title, message)
    finally:
        root.destroy()


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


def _install_after_download_confirm(update: UpdateInfo) -> None:
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
        _messagebox_showinfo(
            "Update downloaded",
            "Package downloaded. Install manually, then relaunch.",
        )
    except Exception as e:  # noqa: BLE001
        _messagebox_error("Update failed", str(e))


def _run_update_download_flow() -> None:
    update = check_for_update()
    if not update:
        _messagebox_showinfo("dronmakr", "You are on the latest release.")
        return
    if not _messagebox_askyesno(
        "Update available",
        f"A newer version ({update.tag}) is available. Download now?",
    ):
        return
    _install_after_download_confirm(update)


def _maybe_offer_update_on_startup() -> None:
    if not getattr(sys, "frozen", False):
        return
    update = check_for_update()
    if not update:
        return
    if not _messagebox_askyesno(
        "Update available",
        f"A newer version ({update.tag}) is available. Download now?",
    ):
        return
    _install_after_download_confirm(update)


def main() -> None:
    print("[desktop] launcher: startup begin", flush=True)
    port = _find_open_port()
    print(f"[desktop] launcher: selected port {port}", flush=True)
    base_url = f"http://127.0.0.1:{port}"
    launch_url = base_url if has_configured_files_root() else f"{base_url}/onboarding"
    settings_url = f"{base_url}/settings"

    start_server(debug=False, port=port, host="127.0.0.1", build_sample_cache=True)
    print("[desktop] launcher: waiting for backend readiness", flush=True)
    ready = _wait_for_health(base_url, timeout_s=30.0)

    if not ready:
        _messagebox_error(
            "dronmakr",
            "The dronmakr server did not become ready in time.\n"
            "Check the terminal or log output for startup errors.",
        )
        sys.exit(1)

    _maybe_offer_update_on_startup()

    image = _load_tray_image()

    def open_home(icon_: Icon, item_: object) -> None:
        open_webui_in_browser(launch_url)

    def open_settings(icon_: Icon, item_: object) -> None:
        open_webui_in_browser(settings_url)

    def check_updates(icon_: Icon, item_: object) -> None:
        if not getattr(sys, "frozen", False):
            _messagebox_showinfo(
                "dronmakr",
                "Update checks apply to the packaged desktop app only.",
            )
            return
        _run_update_download_flow()

    def on_quit(icon_: Icon, item_: object) -> None:
        icon_.stop()

    items: list[MenuItem] = [
        MenuItem("Open in browser", open_home),
        MenuItem("Settings", open_settings),
    ]
    if getattr(sys, "frozen", False):
        items.append(MenuItem("Check for updates…", check_updates))
    items.append(Menu.SEPARATOR)
    items.append(MenuItem("Quit", on_quit))

    menu = Menu(*items)
    icon = Icon(
        "dronmakr",
        icon=image,
        title="dronmakr",
        menu=menu,
    )
    print("[desktop] launcher: tray/menu bar icon running", flush=True)
    try:
        icon.run()
    finally:
        # Eventlet/Flask shutdown is best-effort; exiting the process stops the daemon server thread.
        print("[desktop] launcher: exiting", flush=True)


if __name__ == "__main__":
    main()
