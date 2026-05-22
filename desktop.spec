# -*- mode: python ; coding: utf-8 -*-

import importlib.util
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Resolve version for macOS Info.plist (SPECPATH is defined by PyInstaller when this file runs).
# PyInstaller sets SPECPATH to the *directory* that contains desktop.spec—not the path to the file itself.
_spec_root = Path(SPECPATH).resolve()
if str(_spec_root) not in sys.path:
    sys.path.insert(0, str(_spec_root))
from version import __version__ as _DESKTOP_APP_VERSION  # noqa: E402

_brand_ico = _spec_root / "static" / "branding" / "favicon.ico"
_mac_icns = _spec_root / "static" / "branding" / "macos" / "dronmakr.icns"
# Linux wheels usually omit bundled portaudio binaries; hooks fall back to the system library.
_has_sounddevice_data = importlib.util.find_spec("_sounddevice_data") is not None

_sounddevice_data_hiddenimports = ["_sounddevice_data"] if _has_sounddevice_data else []

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
        "_cffi_backend",
        "cffi",
        "_sounddevice",
        "_soundfile",
        "_soundfile_data",
        "sounddevice",
        "soundfile",
        "preset_authoring",
        "patchcraftr_gui",
        "patchcraftr_live_monitor",
        "pedalboard_isolated_runner",
        "tkinter",
        "_tkinter",
    ]
    + _sounddevice_data_hiddenimports
    + tk_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["patchcraftr_rth_tk.py"],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

# Finder-launched macOS bundles: UPX-packed Mach-O/dylibs can fail to load; console=True leaves no stderr.
_console = sys.platform != "darwin"
_upx = sys.platform != "darwin"

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="dronmakr",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=_upx,
    console=_console,
    icon=str(_brand_ico) if _brand_ico.is_file() else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=_upx,
    upx_exclude=[],
    name="dronmakr",
)

# macOS: wrap the onedir COLLECT output in a proper .app so Finder, Spotlight, and
# drag-to-/Applications installs work like a normal desktop app.
if sys.platform == "darwin":
    from PyInstaller.building.osx import BUNDLE as _OSXBundle

    if not _mac_icns.is_file():
        raise FileNotFoundError(
            f"Missing {_mac_icns}. Commit a Finder icon there (see static/branding/macos/README.md) "
            "or run scripts/build_mac_app_icns.sh locally to generate it."
        )

    app = _OSXBundle(
        coll,
        name="dronmakr.app",
        bundle_identifier="io.github.nafeu.dronmakr",
        icon=str(_mac_icns),
        info_plist={
            "CFBundleName": "dronmakr",
            "CFBundleDisplayName": "dronmakr",
            "CFBundleShortVersionString": str(_DESKTOP_APP_VERSION),
            "CFBundleVersion": str(_DESKTOP_APP_VERSION),
            "NSHighResolutionCapable": True,
        },
    )
