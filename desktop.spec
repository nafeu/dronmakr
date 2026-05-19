# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_submodules

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
