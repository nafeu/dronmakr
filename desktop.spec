# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Resolve version for macOS Info.plist (SPECPATH is defined by PyInstaller when this file runs).
_spec_root = Path(SPECPATH).resolve().parent
if str(_spec_root) not in sys.path:
    sys.path.insert(0, str(_spec_root))
from version import __version__ as _DESKTOP_APP_VERSION  # noqa: E402

hiddenimports_base = collect_submodules("eventlet") + collect_submodules("pystray")
tk_datas, tk_bins, tk_hidden = collect_all("tkinter")

a = Analysis(
    ["desktop_app.py"],
    pathex=[],
    binaries=tk_bins,
    datas=[
        ("templates", "templates"),
        ("static", "static"),
        ("resources", "resources"),
        ("patchcraftr_gui.py", "."),
    ]
    + tk_datas,
    hiddenimports=hiddenimports_base
    + [
        "preset_authoring",
        "patchcraftr_gui",
        "patchcraftr_live_monitor",
        "pedalboard_isolated_runner",
        "tkinter",
        "_tkinter",
    ]
    + tk_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["patchcraftr_rth_tk.py"],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="dronmakr",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="dronmakr",
)

# macOS: wrap the onedir COLLECT output in a proper .app so Finder, Spotlight, and
# drag-to-/Applications installs work like a normal desktop app.
if sys.platform == "darwin":
    from PyInstaller.building.osx import BUNDLE as _OSXBundle

    app = _OSXBundle(
        coll,
        name="dronmakr.app",
        bundle_identifier="io.github.nafeu.dronmakr",
        info_plist={
            "CFBundleName": "dronmakr",
            "CFBundleDisplayName": "dronmakr",
            "CFBundleShortVersionString": str(_DESKTOP_APP_VERSION),
            "CFBundleVersion": str(_DESKTOP_APP_VERSION),
            "NSHighResolutionCapable": True,
        },
    )
