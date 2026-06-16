# macOS Finder / Dock icon (committed)

[`dronmakr.icns`](dronmakr.icns) is **tracked in git** and used by the **Tauri** desktop bundle (`src-tauri/icons/icon.icns`).

- **GitHub Actions** release builds **do not** generate this file (no Pillow / icon pipeline on the runner).
- After changing artwork or theme colours, regenerate locally on a Mac and **commit** the updated `.icns`:

  ```sh
  bash scripts/build_mac_app_icns.sh
  git add static/branding/macos/dronmakr.icns
  ```

  That script uses [`scripts/compose_mac_app_icon_layer.py`](../../../scripts/compose_mac_app_icon_layer.py) (reads **`--theme-a`** from [`assets/templates/_app_css_root.html`](../../../assets/templates/_app_css_root.html)), then **`sips` + `iconutil`**. Requires Pillow (install locally for icon builds).

You can instead place any valid **`dronmakr.icns`** you built with another tool in this folder (same filename).
