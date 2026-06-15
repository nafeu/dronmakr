# -*- mode: python ; coding: utf-8 -*-

import importlib.util
import shutil
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, get_module_file_attribute

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


def _soundfile_packaged_binaries():
    """Libs shipped in the PyPI soundfile wheel — same pairing as contrib hook-soundfile.

    Also bundles libsndfile.dylib as a copy of the arch-specific wheel dylib: PySoundFile builds
    the packaged path as 'libsndfile_' + platform.machine() + '.dylib'. Some frozen/macOS combos
    report an empty machine string, which makes tier-1 load 'libsndfile.dylib' (not shipped by
    the wheel) and crash with a bogus Frameworks/... path before system fallbacks run.
    """
    module_dir = Path(get_module_file_attribute("soundfile")).parent
    data_dir = module_dir / "_soundfile_data"
    if not data_dir.is_dir():
        return []
    destdir = str(data_dir.relative_to(module_dir))
    binaries = [(str(p.resolve()), destdir) for p in sorted(data_dir.glob("libsndfile*.*"))]

    if sys.platform == "darwin":
        arm = data_dir / "libsndfile_arm64.dylib"
        x86 = data_dir / "libsndfile_x86_64.dylib"
        import os as _os
        import platform as _plat

        pm = (_plat.machine() or "").strip().lower()
        if not pm and hasattr(_os, "uname"):
            pm = (_os.uname()[4] or "").strip().lower()

        chosen = None
        if pm in ("arm64", "aarch64") and arm.is_file():
            chosen = arm
        elif pm in ("x86_64", "amd64") and x86.is_file():
            chosen = x86
        elif arm.is_file():
            chosen = arm
        elif x86.is_file():
            chosen = x86
        else:
            _dylibs = sorted(data_dir.glob("libsndfile*.dylib"))
            chosen = _dylibs[0] if _dylibs else None
        if chosen is not None:
            alias_dir = _spec_root / "build" / "pyi_soundfile_dylib_alias"
            alias_dir.mkdir(parents=True, exist_ok=True)
            alias_dylib = alias_dir / "libsndfile.dylib"
            shutil.copyfile(chosen, alias_dylib)
            binaries.append((str(alias_dylib.resolve()), destdir))

    return binaries


_soundfile_bins = _soundfile_packaged_binaries()

# Fail fast if the macOS wheel is broken (e.g. pip --no-binary / bad env). Otherwise the app
# crashes at runtime with a confusing cffi path to libsndfile.dylib.
if sys.platform == "darwin":
    _sf_data = Path(get_module_file_attribute("soundfile")).parent / "_soundfile_data"
    if not _sf_data.is_dir() or not any(_sf_data.glob("libsndfile*.*")):
        raise FileNotFoundError(
            "soundfile macOS wheel is missing _soundfile_data/libsndfile*.dylib. "
            "Reinstall: pip uninstall -y soundfile && pip install soundfile "
            "(do not use --no-binary=soundfile)."
        )

# Frozen desktop_app sets DRONMAKR_ASYNC_MODE=threading; engineio.async_drivers.threading
# imports simple_websocket → wsproto. PyInstaller does not trace that chain — without these,
# SocketIO(..., async_mode="threading") raises ValueError: Invalid async_mode specified.
_hidden_socketio_threading = (
    ["engineio.async_drivers.threading", "engineio.async_drivers._websocket_wsgi"]
    + collect_submodules("simple_websocket")
    + collect_submodules("wsproto")
)

hiddenimports_base = (
    collect_submodules("eventlet")
    + collect_submodules("pystray")
    + _hidden_socketio_threading
)
tk_datas, tk_bins, tk_hidden = collect_all("tkinter")
_sd_collect_datas, _sd_collect_bins, _sd_collect_hidden = collect_all("sounddevice")
_dd_collect_datas, _dd_collect_bins, _dd_collect_hidden = collect_all("dawdreamer")
_dawdreamer_submodules = collect_submodules("dawdreamer")

a = Analysis(
    ["desktop_app.py"],
    pathex=[],
    binaries=tk_bins + _soundfile_bins + _sd_collect_bins + _dd_collect_bins,
    datas=[
        ("templates", "templates"),
        ("static", "static"),
        ("resources", "resources"),
        ("patchcraftr_gui.py", "."),
    ]
    + tk_datas
    + _sd_collect_datas
    + _dd_collect_datas,
    hiddenimports=hiddenimports_base
    + [
        "_cffi_backend",
        "cffi",
        "_sounddevice",
        "_soundfile",
        "_soundfile_data",
        "sounddevice",
        "soundfile",
        "dawdreamer",
        "audio_host",
        "audio_worker",
        "preset_authoring",
        "patchcraftr_gui",
        "patchcraftr_live_monitor",
        "pedalboard_isolated_runner",
        "generate_sample",
        "generate_midi",
        "desktop_update_ui",
        "desktop_update_install",
        "desktop_native_dialog",
        "plugin_default_paths",
        "native_folder_picker",
        "server_error_logging",
        "tkinter",
        "_tkinter",
    ]
    + _sounddevice_data_hiddenimports
    + tk_hidden
    + list(_sd_collect_hidden)
    + list(_dd_collect_hidden)
    + _dawdreamer_submodules,
    hookspath=["hooks"],
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
