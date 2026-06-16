# -*- mode: python ; coding: utf-8 -*-

import shutil
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, get_module_file_attribute

_spec_root = Path(SPECPATH).resolve()
if str(_spec_root) not in sys.path:
    sys.path.insert(0, str(_spec_root))

_brand_ico = _spec_root / "static" / "branding" / "favicon.ico"


def _soundfile_packaged_binaries():
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

if sys.platform == "darwin":
    _sf_data = Path(get_module_file_attribute("soundfile")).parent / "_soundfile_data"
    if not _sf_data.is_dir() or not any(_sf_data.glob("libsndfile*.*")):
        raise FileNotFoundError(
            "soundfile macOS wheel is missing _soundfile_data/libsndfile*.dylib. "
            "Reinstall: pip uninstall -y soundfile && pip install soundfile "
            "(do not use --no-binary=soundfile)."
        )

_hidden_socketio_threading = (
    ["engineio.async_drivers.threading", "engineio.async_drivers._websocket_wsgi"]
    + collect_submodules("simple_websocket")
    + collect_submodules("wsproto")
)

_dd_collect_datas, _dd_collect_bins, _dd_collect_hidden = collect_all("dawdreamer")
_dawdreamer_submodules = collect_submodules("dawdreamer")

a = Analysis(
    ["backend_server.py"],
    pathex=[],
    binaries=_soundfile_bins + _dd_collect_bins,
    datas=[
        ("frontend/dist", "frontend/dist"),
        ("static", "static"),
        ("resources", "resources"),
    ]
    + _dd_collect_datas,
    hiddenimports=_hidden_socketio_threading
    + [
        "_cffi_backend",
        "cffi",
        "_soundfile",
        "_soundfile_data",
        "soundfile",
        "dawdreamer",
        "audio_host",
        "audio_worker",
        "preset_authoring",
        "pedalboard_isolated_runner",
        "generate_sample",
        "generate_midi",
        "plugin_default_paths",
        "native_folder_picker",
        "server_error_logging",
    ]
    + list(_dd_collect_hidden)
    + _dawdreamer_submodules,
    hookspath=["hooks"],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "pystray"],
    noarchive=False,
)
pyz = PYZ(a.pure)

_console = True
_upx = sys.platform != "darwin"

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="dronmakr-backend",
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
    name="dronmakr-backend",
)
