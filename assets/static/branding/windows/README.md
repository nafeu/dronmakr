# Windows desktop app icon (committed)

[`dronmakr.ico`](dronmakr.ico) is **tracked in git** and copied to the Tauri bundle (`src-tauri/icons/icon.ico`).

- **GitHub Actions** Windows release builds use this committed file (no icon pipeline on the runner).
- After changing artwork or theme colours, regenerate locally on any OS with Pillow and **commit** the updated `.ico`:

  ```sh
  npm run icons:windows
  git add assets/static/branding/windows/dronmakr.ico src-tauri/icons/icon.ico
  ```

  Uses [`scripts/compose_windows_app_icon_layer.py`](../../../scripts/compose_windows_app_icon_layer.py) (reads **`--theme-a`** from [`assets/templates/_app_css_root.html`](../../../assets/templates/_app_css_root.html)) and embeds 16–256 px sizes for crisp shell rendering.

This is separate from the macOS `.icns` pipeline (`npm run icons`); regenerating macOS icons does **not** update the Windows icon.
