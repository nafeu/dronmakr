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


def _looks_like_windows_path(path: str) -> bool:
    return len(path) >= 2 and path[0].isalpha() and path[1] == ":"


def _normalize_initial_dir(path: str | None) -> str:
    """Return an existing directory to open the picker in, or empty string."""
    if not isinstance(path, str):
        return ""
    cleaned = path.strip().strip('"').strip("'")
    if not cleaned:
        return ""
    expanded = os.path.expanduser(cleaned)
    if _looks_like_windows_path(expanded):
        candidate = expanded.replace("/", "\\")
    else:
        candidate = os.path.abspath(expanded)
    if os.path.isdir(candidate):
        return candidate
    parent = os.path.dirname(candidate)
    while parent and parent != candidate:
        if os.path.isdir(parent):
            return parent
        next_parent = os.path.dirname(parent)
        if next_parent == parent:
            break
        parent = next_parent
    return ""


def _pick_folder_darwin(initial_dir: str | None = None) -> FolderPickResult:
    start_dir = _normalize_initial_dir(initial_dir)
    if start_dir:
        escaped = start_dir.replace("\\", "\\\\").replace('"', '\\"')
        script = (
            'tell application "Finder" to activate\n'
            f'set defaultLocation to POSIX file "{escaped}"\n'
            "return POSIX path of (choose folder default location defaultLocation)"
        )
    else:
        script = (
            'tell application "Finder" to activate\n'
            "return POSIX path of (choose folder)"
        )
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


def _pick_folder_linux(initial_dir: str | None = None) -> FolderPickResult:
    start_dir = _normalize_initial_dir(initial_dir) or os.path.expanduser("~")
    for cmd in (
        [
            "zenity",
            "--file-selection",
            "--directory",
            "--modal",
            f"--filename={start_dir}{os.sep}",
        ],
        ["kdialog", "--getexistingdirectory", start_dir],
        ["yad", "--file", "--directory", "--title=Select folder", f"--filename={start_dir}"],
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


def _pick_folder_win32(initial_dir: str | None = None) -> FolderPickResult:
    start_dir = _normalize_initial_dir(initial_dir)
    ps_script = """
Add-Type -AssemblyName System.Windows.Forms
$dlg = New-Object System.Windows.Forms.FolderBrowserDialog
$dlg.Description = 'Select folder'
$dlg.ShowNewFolderButton = $true
$initial = $env:DRONMAKR_PICKER_INITIAL
if ($initial -and (Test-Path -LiteralPath $initial -PathType Container)) {
  $dlg.SelectedPath = $initial
}
$null = $dlg.ShowDialog()
if ($dlg.SelectedPath) { $dlg.SelectedPath } else { '' }
""".strip()
    kw: dict[str, object] = {}
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    env = os.environ.copy()
    if start_dir:
        env["DRONMAKR_PICKER_INITIAL"] = start_dir
    else:
        env.pop("DRONMAKR_PICKER_INITIAL", None)
    for exe in ("powershell.exe", "pwsh.exe"):
        try:
            proc = subprocess.run(
                [exe, "-NoProfile", "-STA", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
                env=env,
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


def pick_folder_subprocess(initial_dir: str | None = None) -> FolderPickResult:
    """Spawn the platform folder dialog; suitable for Flask worker threads."""
    if sys.platform == "darwin":
        return _pick_folder_darwin(initial_dir)
    if sys.platform.startswith("linux"):
        return _pick_folder_linux(initial_dir)
    if sys.platform == "win32":
        return _pick_folder_win32(initial_dir)
    return FolderPickResult(status="unavailable")
