# macOS packaging artifacts

[`dronmakr.icns`](dronmakr.icns) is **generated** (gitignored) from [`../../static/branding/android-chrome-512x512.png`](../../static/branding/android-chrome-512x512.png) by [`scripts/build_mac_app_icns.sh`](../../scripts/build_mac_app_icns.sh).

It stays **outside** [`resources/`](../resources/) so the PyInstaller datas tree does not ship a redundant copy of the icon inside the frozen bundle—only **`dronmakr.app`** embedding uses it via `desktop.spec`.
