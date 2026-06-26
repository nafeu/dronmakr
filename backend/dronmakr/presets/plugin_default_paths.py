"""OS-specific default PLUGIN_PATHS (comma-separated) for VST/AU plugin scan roots."""

from __future__ import annotations

import os
import sys


def _abs(parts: tuple[str, ...]) -> str:
    return os.path.abspath(os.path.join(*parts))


def default_plugin_paths_csv() -> str:
    """
    Return comma-separated canonical plugin folders for this OS.

    Paths are normalized with expanduser / expandvars; missing directories are still included
    (plugin scans skip non-existent directories).
    """
    sep = ","
    if sys.platform == "darwin":
        home = os.path.expanduser("~")
        # VST3 + Audio Unit paths — omit legacy macOS VST2 (.vst) from defaults to avoid broken pickers.
        rows = [
            "/Library/Audio/Plug-Ins/Components",
            "/Library/Audio/Plug-Ins/VST3",
            _abs((home, "Library", "Audio", "Plug-Ins", "Components")),
            _abs((home, "Library", "Audio", "Plug-Ins", "VST3")),
        ]
        return sep.join(rows)

    if sys.platform == "win32":
        pf = os.environ.get("ProgramFiles") or ""
        pfx86 = os.environ.get("ProgramFiles(x86)") or pf
        candidate_dirs: list[str] = []
        if pf:
            candidate_dirs.extend(
                [
                    os.path.join(pf, "Common Files", "VST3"),
                    os.path.join(pf, "VSTPlugins"),
                    os.path.join(pf, "Steinberg", "VSTPlugins"),
                ]
            )
        if pfx86 and pfx86 != pf:
            candidate_dirs.extend(
                [
                    os.path.join(pfx86, "Common Files", "VST3"),
                    os.path.join(pfx86, "VSTPlugins"),
                    os.path.join(pfx86, "Steinberg", "VSTPlugins"),
                ]
            )
        lap = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Common\VST3")
        if lap and "%" not in lap:
            candidate_dirs.append(lap)
        return sep.join(os.path.abspath(p) for p in candidate_dirs if p)

    # Linux / other POSIX
    home = os.path.expanduser("~")
    rows = [
        _abs((home, ".vst")),
        _abs((home, ".vst3")),
        _abs((home, "vst")),
        "/usr/local/lib/vst",
        "/usr/lib/vst",
        "/usr/lib/lxvst",
        "/usr/lib/vst3",
        "/usr/local/lib/vst3",
    ]
    return sep.join(os.path.abspath(p) for p in rows)
