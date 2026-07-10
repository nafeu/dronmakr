"""Native folder pickers safe to call from worker threads (e.g. Flask request handlers).

Do not use Tkinter here: macOS requires GUI on the main thread; the web server runs on a daemon thread.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class FolderPickResult:
    """Result of a native folder chooser subprocess."""

    path: str = ""
    """Absolute or normalized path when ``status`` is ``ok``."""

    status: str = "unavailable"
    """One of: ``ok``, ``cancelled``, ``unavailable``."""


_TIMEOUT_S = 300


def _pick_folder_darwin() -> FolderPickResult:
    script = 'tell application "Finder" to activate\nreturn POSIX path of (choose folder)'
    try:
        proc = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
        )
    except FileNotFoundError:
        return FolderPickResult(status="unavailable")
    except subprocess.TimeoutExpired:
        return FolderPickResult(status="unavailable")
    out = (proc.stdout or "").strip()
    if proc.returncode == 0 and out:
        return FolderPickResult(path=out, status="ok")
    if proc.returncode != 0:
        return FolderPickResult(status="cancelled")
    return FolderPickResult(status="cancelled")


def _pick_folder_linux() -> FolderPickResult:
    for cmd in (
        ["zenity", "--file-selection", "--directory", "--modal"],
        ["kdialog", "--getexistingdirectory", os.path.expanduser("~")],
        ["yad", "--file", "--directory", "--title=Select folder"],
    ):
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
            )
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            return FolderPickResult(status="unavailable")
        out = (proc.stdout or "").strip()
        if proc.returncode == 0 and out:
            return FolderPickResult(path=out, status="ok")
        return FolderPickResult(status="cancelled")
    return FolderPickResult(status="unavailable")


def _pick_folder_win32() -> FolderPickResult:
    ps_script = """
Add-Type -AssemblyName System.Windows.Forms
$dlg = New-Object System.Windows.Forms.FolderBrowserDialog
$dlg.Description = 'Select folder'
$dlg.ShowNewFolderButton = $true
$null = $dlg.ShowDialog()
if ($dlg.SelectedPath) { $dlg.SelectedPath } else { '' }
""".strip()
    kw: dict[str, object] = {}
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    for exe in ("powershell.exe", "pwsh.exe"):
        try:
            proc = subprocess.run(
                [exe, "-NoProfile", "-STA", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
                **kw,
            )
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            return FolderPickResult(status="unavailable")
        out = (proc.stdout or "").strip()
        if proc.returncode != 0:
            continue
        if out:
            return FolderPickResult(path=out, status="ok")
        return FolderPickResult(status="cancelled")
    return FolderPickResult(status="unavailable")


def pick_folder_subprocess() -> FolderPickResult:
    """Spawn the platform folder dialog; suitable for Flask worker threads."""
    if sys.platform == "darwin":
        return _pick_folder_darwin()
    if sys.platform.startswith("linux"):
        return _pick_folder_linux()
    if sys.platform == "win32":
        return _pick_folder_win32()
    return FolderPickResult(status="unavailable")
