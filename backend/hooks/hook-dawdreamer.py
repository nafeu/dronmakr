"""PyInstaller hook — bundle DawDreamer for offline VST rendering."""

from PyInstaller.utils.hooks import collect_all, collect_submodules

hiddenimports = collect_submodules("dawdreamer")
datas, binaries, more_hidden = collect_all("dawdreamer")
hiddenimports += more_hidden
