# macOS Finder / Dock icon (committed)

[`dronmakr.icns`](dronmakr.icns) is **tracked in git** and used by **`desktop.spec`** for the **`dronmakr.app`** bundle.

- **GitHub Actions** release builds **do not** generate this file (no Pillow / icon pipeline on the runner).
- After changing artwork or theme colours, regenerate locally on a Mac and **commit** the updated `.icns`:

  ```sh
  bash scripts/build_mac_app_icns.sh
  git add static/branding/macos/dronmakr.icns
  ```

  That script uses [`scripts/compose_mac_app_icon_layer.py`](../../../scripts/compose_mac_app_icon_layer.py) (reads **`--theme-a`** from [`templates/_app_css_root.html`](../../../templates/_app_css_root.html)), then **`sips` + `iconutil`**. Requires a venv with **Pillow** from **`requirements.txt`**.

You can instead place any valid **`dronmakr.icns`** you built with another tool in this folder (same filename).
