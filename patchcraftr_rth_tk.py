# PyInstaller runtime hook — Tcl/Tk live under sys._MEIPASS as tcl8.x / tk8.x trees.
import os
import sys


def _patchcraftr_apply_tk_env() -> None:
    base = getattr(sys, "_MEIPASS", "") or ""
    if not base or not getattr(sys, "frozen", False):
        return
    try:
        entries = os.listdir(base)
    except OSError:
        return
    for name in sorted(entries):
        path = os.path.join(base, name)
        if not os.path.isdir(path):
            continue
        if name.startswith("tcl"):
            os.environ.setdefault("TCL_LIBRARY", path)
        elif name.startswith("tk"):
            os.environ.setdefault("TK_LIBRARY", path)


_patchcraftr_apply_tk_env()
