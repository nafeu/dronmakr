"""PyInstaller hook — Pedalboard pulls ``midi_utils`` in at runtime for MIDI rendering."""

from PyInstaller.utils.hooks import collect_all, collect_submodules

hiddenimports = collect_submodules("pedalboard")
datas, binaries, more_hidden = collect_all("pedalboard")
hiddenimports += more_hidden
