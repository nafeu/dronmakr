# macOS packaging artifacts

[`dronmakr.icns`](dronmakr.icns) is **generated** (gitignored) by [`scripts/build_mac_app_icns.sh`](../../scripts/build_mac_app_icns.sh).

The shell script builds an intermediate **`icns-layer-1024.png`** (gitignored) with [`scripts/compose_mac_app_icon_layer.py`](../../scripts/compose_mac_app_icon_layer.py): **`--theme-a`** from [`templates/_app_css_root.html`](../../templates/_app_css_root.html) is painted edge-to-edge (opaque) behind the centred transparent logo [`static/branding/android-chrome-512x512.png`](../../static/branding/android-chrome-512x512.png), so translucent PNG margins never pick up Finder’s muddy white halo. macOS draws the dock/Spotlight silhouette on top automatically.

Artifacts stay **outside** [`resources/`](../resources/) so PyInstaller **`datas=("resources")`** doesn’t duplicate the packaged icon blobs.
