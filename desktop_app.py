from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run-patchcraftr":
        from settings import ensure_settings

        ensure_settings()
        from patchcraftr_gui import main as _patchcraftr_main

        _patchcraftr_main()
        raise SystemExit(0)
    if "--pedalboard-worker" in sys.argv:
        # Child process: real OS main thread for AU/VST (Flask runs off-thread in tray mode).
        sys.argv = [a for a in sys.argv if a != "--pedalboard-worker"]
        from pedalboard_isolated_runner import run_stdio_worker

        run_stdio_worker()
        raise SystemExit(0)
    os.environ["DRONMAKR_ASYNC_MODE"] = "threading"

import socket
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image
from pystray import Icon, Menu, MenuItem

from bundle_paths import get_bundle_app_root
from settings import get_files_root, has_configured_files_root
from updater import (
    UpdateInfo,
    download_update,
    fetch_update_info_throttled,
    peek_cached_update_info,
    reveal_file,
)
from webui import open_webui_in_browser, start_server

_HEALTH_PATH = "/api/health"
_SOURCE_ICON = Path("static") / "branding" / "favicon-32x32.png"
_TRAY_ICON_LIGHT_MODE = Path("static") / "branding" / "tray-icon-light-mode.png"
_TRAY_ICON_DARK_MODE = Path("static") / "branding" / "tray-icon-dark-mode.png"

# Tried in order so the browser sees a stable origin (incl. port) for permissions.
_DESKTOP_PREFERRED_PORTS: tuple[int, ...] = (3766, 3767, 3768, 3769)


def _find_open_port() -> int:
    """Ask the OS for any free port (last resort)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _can_bind_desktop_port(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, int(port)))
    except OSError:
        return False
    return True


def _select_desktop_listen_port(host: str = "127.0.0.1") -> int:
    for p in _DESKTOP_PREFERRED_PORTS:
        if _can_bind_desktop_port(host, p):
            return p
    return _find_open_port()


def _bundle_root() -> Path:
    return get_bundle_app_root()


def _patchcraftr_work_dir() -> Path:
    """Directory so ``resources/`` and managed paths resolve like the repo root."""
    if getattr(sys, "frozen", False):
        return _bundle_root()
    return Path(__file__).resolve().parent


def _launch_patchcraftr_gui() -> None:
    work_dir = _patchcraftr_work_dir()
    if getattr(sys, "frozen", False):
        argv = [sys.executable, "--run-patchcraftr"]
    else:
        script = work_dir / "patchcraftr_gui.py"
        if not script.is_file():
            raise RuntimeError(f"Missing patchcraftr_gui.py at {script}")
        argv = [sys.executable, str(script)]
    subprocess.Popen(argv, cwd=str(work_dir), start_new_session=True)


def _macos_menu_bar_prefers_white_icon() -> bool:
    """Dark menu bar (macOS dark / dark-aqua chrome) needs a white glyph."""
    if sys.platform != "darwin":
        return False
    try:
        from Foundation import NSUserDefaults  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        style = NSUserDefaults.standardUserDefaults().stringForKey_("AppleInterfaceStyle")
        return style == "Dark"
    except Exception:
        return False


def _tray_icon_path() -> Path:
    root = _bundle_root()
    if sys.platform == "darwin":
        if _macos_menu_bar_prefers_white_icon():
            return root / _TRAY_ICON_LIGHT_MODE
        return root / _TRAY_ICON_DARK_MODE
    # Windows/Linux: typical systray background is light; use black glyph.
    return root / _TRAY_ICON_DARK_MODE


def _load_tray_image() -> Image.Image:
    path = _tray_icon_path()
    if path.is_file():
        with Image.open(path) as im:
            return im.copy()
    fallback = _bundle_root() / _SOURCE_ICON
    if fallback.is_file():
        with Image.open(fallback) as im:
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
    update = fetch_update_info_throttled(force=True)
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
    update = fetch_update_info_throttled(force=True)
    if not update:
        return
    if not _messagebox_askyesno(
        "Update available",
        f"A newer version ({update.tag}) is available. Download now?",
    ):
        return
    _install_after_download_confirm(update)


def main(debug: bool = False) -> None:
    print("[desktop] launcher: startup begin", flush=True)
    port = _select_desktop_listen_port("127.0.0.1")
    print(f"[desktop] launcher: selected port {port}", flush=True)
    base_url = f"http://127.0.0.1:{port}"
    launch_url = base_url if has_configured_files_root() else f"{base_url}/onboarding"
    settings_url = f"{base_url}/settings"
    about_url = f"{base_url}/about"

    start_server(
        debug=debug,
        port=port,
        host="127.0.0.1",
        build_sample_cache=True,
    )
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

    health_probe = base_url.rstrip("/") + _HEALTH_PATH

    def tray_status_label(_item: MenuItem) -> str:
        try:
            with urllib.request.urlopen(health_probe, timeout=1.2) as resp:
                if resp.getcode() == 200:
                    return f"Server running · {base_url}"
        except Exception:
            pass
        return f"Server not responding · {base_url}"

    def open_home(icon_: Icon, item_: object) -> None:
        open_webui_in_browser(launch_url)

    def open_settings(icon_: Icon, item_: object) -> None:
        open_webui_in_browser(settings_url)

    def open_about(icon_: Icon, item_: object) -> None:
        open_webui_in_browser(about_url)

    def browse_files(icon_: Icon, item_: object) -> None:
        root = get_files_root(allow_default=False)
        if not root or not os.path.isdir(root):
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", root])
        elif sys.platform == "win32":
            os.startfile(root)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", root])

    def browse_files_enabled(_item: MenuItem) -> bool:
        path = get_files_root(allow_default=False)
        return bool(path and os.path.isdir(path))

    def launch_patchcraftr(_icon: Icon, _item: object) -> None:
        try:
            _launch_patchcraftr_gui()
        except Exception as e:  # noqa: BLE001
            _messagebox_error("patchcraftr", str(e))

    def check_updates(icon_: Icon, item_: object) -> None:
        if not getattr(sys, "frozen", False):
            _messagebox_showinfo(
                "dronmakr",
                "Update checks apply to the packaged desktop app only.",
            )
            return
        _run_update_download_flow()

    def download_update_now(icon_: Icon, item_: object) -> None:
        info = peek_cached_update_info()
        if not info:
            return
        if not _messagebox_askyesno(
            "Update available",
            f"Download {info.tag} now?",
        ):
            return
        _install_after_download_confirm(info)

    def download_update_visible(_item: MenuItem) -> bool:
        return peek_cached_update_info() is not None

    def download_update_label(_item: MenuItem) -> str:
        info = peek_cached_update_info()
        return f"Download {info.tag}…" if info else "Download update…"

    stop_menu_poll = threading.Event()

    def on_quit(icon_: Icon, item_: object) -> None:
        stop_menu_poll.set()
        icon_.stop()

    def poll_menu_for_refresh(tray_icon: Icon) -> None:
        while not stop_menu_poll.wait(4.0):
            try:
                if getattr(sys, "frozen", False):
                    fetch_update_info_throttled(force=False)
                tray_icon.update_menu()
            except Exception:
                break

    items: list[MenuItem] = [
        MenuItem(tray_status_label, None, enabled=False),
        Menu.SEPARATOR,
        MenuItem("Open dronmakr in browser", open_home),
        MenuItem(
            "Browse files",
            browse_files,
            enabled=browse_files_enabled,
        ),
        MenuItem("Launch patchcraftr", launch_patchcraftr),
        MenuItem("Settings", open_settings),
        MenuItem("About", open_about),
    ]
    if getattr(sys, "frozen", False):
        items.extend(
            [
                MenuItem(
                    download_update_label,
                    download_update_now,
                    visible=download_update_visible,
                ),
                MenuItem("Check for updates…", check_updates),
            ]
        )
    items.append(Menu.SEPARATOR)
    items.append(MenuItem("Quit", on_quit))

    menu = Menu(*items)
    icon = Icon(
        "dronmakr",
        icon=image,
        title="dronmakr",
        menu=menu,
    )
    threading.Thread(target=poll_menu_for_refresh, args=(icon,), daemon=True).start()
    print("[desktop] launcher: tray/menu bar icon running", flush=True)
    try:
        icon.run()
    finally:
        stop_menu_poll.set()
        # Eventlet/Flask shutdown is best-effort; exiting the process stops the daemon server thread.
        print("[desktop] launcher: exiting", flush=True)


if __name__ == "__main__":
    main()
