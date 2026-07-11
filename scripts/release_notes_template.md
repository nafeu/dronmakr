<p align="center">
  <a href="https://discord.gg/BysAyRje57"><img src="https://img.shields.io/discord/1358944581873307871?label=discord&logo=discord&style=for-the-badge" alt="Discord" /></a>
  <a href="https://www.patreon.com/phrakture"><img src="https://img.shields.io/badge/Patreon-F96854?style=for-the-badge&logo=patreon&logoColor=white" alt="Patreon" /></a>
  <a href="https://buymeacoffee.com/nafeunasir"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me a Coffee" /></a>
</p>

Thanks for checking `dronmakr` out! Please report bugs in our [Discord bug reports channel](https://discord.gg/BysAyRje57) or [open an issue](https://github.com/nafeu/dronmakr/issues) right here on GitHub and I'll try and help you out as soon as I can. You can support the project by signing up as a [Patreon member](https://www.patreon.com/phrakture) or donating through [Buy Me a Coffee](https://buymeacoffee.com/nafeunasir), anything helps!

## Changelog

{{CHANGELOG}}

## Instructions

Pick the download for your OS below, install, launch **dronmakr**, then choose where to store your **`dronmakr-files`** folder on first run (generated audio, presets, config). Point **Settings** at your VST3/AU plug-in paths when you're ready to generate.

### macOS (Apple Silicon)

1. Download **`dronmakr-v{{VERSION}}-macos-arm64.dmg`** (or the `.tar.gz`).
2. Open the DMG and install **dronmakr.app** (drag to Applications or run from the bundle).
3. If macOS blocks launch: **System Settings → Privacy & Security → Open Anyway**, or **Control-click the app → Open** once. Release builds are not Apple-notarized.
4. Use **VST3** and/or **AU** plug-ins (configure paths in Settings).

### Windows (x86_64)

1. Download **`dronmakr-v{{VERSION}}-windows-x64.zip`**.
2. Unzip, open the **`nsis`** folder, and run **`dronmakr_*_x64-setup.exe`**.
3. Launch from the Start menu. See **`README-windows.txt`** in the zip if you need more detail.
4. Use **VST3** plug-ins (configure paths in Settings).

### Linux (x86_64, experimental)

> **Unstable:** Linux builds are published for testing but are **not fully supported yet**. Expect distro-specific UI issues, optional folder-picker deps (`zenity` / `kdialog` / `yad`), and less mature VST3 hosting than on macOS/Windows. Read **`README-linux.txt`** inside the archive before installing, and please [report issues](https://github.com/nafeu/dronmakr/issues) with `errors.log` from `~/.local/share/dronmakr/logs/` when something breaks.

1. Download **`dronmakr-v{{VERSION}}-linux-x64.tar.gz`** (`.deb` / `.rpm` are also in the archive).
2. Extract or install the package for your distro, then launch **dronmakr**.
3. **VST3 only** on Linux (no AU). Configure plug-in paths in Settings.

**Updates:** download a newer release and replace your existing install. The in-app toolbar also links to GitHub Releases when an update is available.
