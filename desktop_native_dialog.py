"""Native dialogs callable from tray / auxiliary threads without Tkinter on macOS/Windows."""

from __future__ import annotations

import subprocess
import sys
from shutil import which


def _run_osascript(source: str) -> str:
    proc = subprocess.run(
        ["osascript", "-e", source],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return (proc.stdout or "").strip()


def _darwin_yes_no(
    title: str,
    message: str,
    *,
    yes_label: str,
    no_label: str,
    default_no: bool,
) -> bool:
    t = title.replace('"', "")
    m = message.replace('"', "").replace("\r", "").replace("\n", " ")
    yn = yes_label.replace('"', "")
    nn = no_label.replace('"', "")
    default_btn = nn if default_no else yn
    source = (
        'tell application "System Events"\n'
        "  activate\n"
        "end tell\n"
        f'set Ans to button returned of (display alert "{t}" message "{m}" '
        f'buttons {{"{nn}","{yn}"}} default button "{default_btn}")\n'
        f'if Ans is "{yn}" then\n'
        '  return "yes"\n'
        "else\n"
        '  return "no"\n'
        "end if\n"
    )
    out = _run_osascript(source).lower()
    return out == "yes"


def _darwin_alert_generic(title: str, message: str, style: str) -> None:
    t = title.replace('"', "")
    m = message.replace('"', "").replace("\r", "").replace("\n", " ")
    subprocess.run(
        [
            "osascript",
            "-e",
            f'display alert "{t}" message "{m}" as {style} buttons {{"OK"}} default button "OK"',
        ],
        capture_output=False,
        check=False,
    )


def tray_show_info(title: str, message: str) -> None:
    msg = message.strip()
    if sys.platform == "darwin":
        _darwin_alert_generic(title, msg, "informational")
        return
    if sys.platform == "win32":
        _win_show(title, msg, style="info")
        return
    xz = which("zenity")
    if xz:
        subprocess.run([xz, "--info", "--width=460", "--no-wrap", "--title", title, "--text", msg], check=False)
        return
    print(f"[desktop] {title}: {msg}", flush=True)


def tray_show_error(title: str, message: str) -> None:
    msg = message.strip()
    if sys.platform == "darwin":
        _darwin_alert_generic(title, msg, "critical")
        return
    if sys.platform == "win32":
        _win_show(title, msg, style="error")
        return
    xz = which("zenity")
    if xz:
        subprocess.run([xz, "--error", "--title", title, "--text", msg, "--width=460"], check=False)
        return
    print(f"[desktop] ERROR {title}: {msg}", flush=True)


def tray_ask_yes_no(
    title: str,
    message: str,
    *,
    yes_label: str,
    no_label: str,
    default_no: bool = True,
) -> bool:
    """Return True for yes / affirmative (first yes_label semantics)."""
    msg = message.replace('"', "").replace("\r", "").replace("\n", " ")
    if sys.platform == "darwin":
        return _darwin_yes_no(title, msg, yes_label=yes_label, no_label=no_label, default_no=default_no)
    if sys.platform == "win32":
        return _win_yes_no(title, msg, default_no=default_no)

    xz = which("zenity")
    if xz:
        proc = subprocess.run(
            [
                xz,
                "--question",
                "--width=460",
                "--no-wrap",
                "--title",
                title,
                "--text",
                msg,
                "--ok-label",
                yes_label,
                "--cancel-label",
                no_label,
            ],
            check=False,
        )
        return proc.returncode == 0

    print(f"[desktop] {title} — {msg} ({yes_label}/{no_label})", flush=True)
    return False


def _win_show(title: str, message: str, style: str) -> None:
    import ctypes  # noqa: PLC0415

    icon = {"info": 0x40, "error": 0x10, "warn": 0x30}.get(style, 0x40)
    MB_OK = 0x00000000
    MB_TOPMOST = 0x00040000
    ctypes.windll.user32.MessageBoxW(0, message[:2048], title[:512], MB_OK | icon | MB_TOPMOST)


def _win_yes_no(title: str, message: str, *, default_no: bool) -> bool:
    import ctypes  # noqa: PLC0415

    MB_YESNO = 0x00000004
    MB_ICONQUESTION = 0x00000020
    MB_DEFBUTTON2 = 0x00000100 if default_no else 0
    MB_TOPMOST = 0x00040000
    IDYES = 6
    rv = ctypes.windll.user32.MessageBoxW(
        0,
        message[:2048],
        title[:512],
        MB_YESNO | MB_ICONQUESTION | MB_DEFBUTTON2 | MB_TOPMOST,
    )
    return int(rv) == IDYES
