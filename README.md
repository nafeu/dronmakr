<p align="center">
  <img src="static/branding/logo.png" alt="dronmakr" width="100" />
</p>

<p align="center"><em>pronounced “drone maker”</em></p>

<p align="center">Python-based suite of sample generation, editing, and packaging tools with browser-ui for auditioning, beatbuilding, collections, and more.</p>

<p align="center">
  <a href="https://discord.gg/BysAyRje57"><img src="https://img.shields.io/discord/1358944581873307871?label=discord&logo=discord&style=for-the-badge" alt="Discord" /></a>
  <a href="https://www.patreon.com/phrakture"><img src="https://img.shields.io/badge/Patreon-F96854?style=for-the-badge&logo=patreon&logoColor=white" alt="Patreon" /></a>
</p>

![Auditionr Preview](preview-auditionr.png)

![Beatbuildr Preview](preview-beatbuildr.png)

![Collections Preview](preview-collections.png)

## Made Using `dronmakr`

- [Ember Proxima - Ambient Drone Pack](https://www.youtube.com/watch?v=DcgXYEDiIHc)
- Parts of the [Primordialis OST](https://store.steampowered.com/app/3011360/Primordialis/)

## Installation

### Desktop (GitHub Releases)

1. Open **[Releases](https://github.com/nafeu/dronmakr/releases/latest)** and download the archive for your platform:
   - **macOS (Apple Silicon):** `dronmakr-v*-macos-arm64.tar.gz`
   - **macOS (Intel):** `dronmakr-v*-macos-x64.tar.gz`
   - **Linux (x86_64):** `dronmakr-v*-linux-x64.tar.gz`
   - **Windows:** `dronmakr-v*-windows-x64.zip`
2. Extract the archive. Inside you will find a `dronmakr` folder containing the app bundle from PyInstaller.
3. Run the executable:
   - **macOS / Linux:** `dronmakr` (inside `dronmakr/`)
   - **Windows:** `dronmakr.exe`
4. A **console window** may stay open (useful for logs). The **menu bar** (macOS) or **system tray** (Windows) icon lets you open the app in your browser, browse your `dronmakr-files` folder, settings, and about page. On first launch, choose where to store `dronmakr-files`.

Packaged desktop builds also check **GitHub Releases** for updates: use **Check for updates…** or **Download v…** in the tray when a newer version is available (menu checks the API at most once per hour).

Desktop releases include a **vendored FFmpeg** binary (used when **Folysplitr** uploads browser recordings); you do not need a separate system FFmpeg install for that path. Notices: `resources/ffmpeg/THIRD_PARTY_FFMPEG.txt` in the bundle — see [resources/ffmpeg/LICENSE.third_party.ffmpeg](https://github.com/nafeu/dronmakr/blob/main/resources/ffmpeg/LICENSE.third_party.ffmpeg).

**Requirements**

- Python **3.10+** (the project was developed on Python 3.10.16)
- macOS (may work on Windows or Linux; contributors welcome)
- A VST3 or AU library with a few working instruments and effects (for preset capture workflows—see [CLI.md](https://github.com/nafeu/dronmakr/blob/main/CLI.md))

```sh
git clone https://github.com/nafeu/dronmakr.git
cd dronmakr
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env-sample .env
```

Edit `.env` with paths and plugins, for example:

```env
PLUGIN_PATHS="/Library/Audio/Plug-Ins/Components,/Library/Audio/Plug-Ins/VST,/Library/Audio/Plug-Ins/VST3"
ASSERT_INSTRUMENT="Reaktor 6"
IGNORE_PLUGINS=""
CUSTOM_PLUGINS=""
```

### Usage

**Open the web UI** by clicking on the app tray icon and clicking `open dronmakr in browser` or if manually running server use:

```sh
python dronmakr.py
```

Then open the URL shown in the terminal (defaults to the unified web UI on port **3766**).

For **advanced command-line usage** (batch generators, packaging presets, local PyInstaller builds), see **[CLI.md on GitHub](https://github.com/nafeu/dronmakr/blob/main/CLI.md)**.

## Project Limitations

The VST/AU running functionality of this project is built ontop of [pedalboard.io](https://spotify.github.io/pedalboard/reference/pedalboard.io.html) which is a python wrapper on the [JUCE framework](https://juce.com/). There are [known compatibility issues](https://spotify.github.io/pedalboard/reference/pedalboard.html#pedalboard.VST3Plugin). Some of the ones that I've been able to get working are as follows:

_* All testing was done on `macOS Sequoia 15.1` on an `Apple M4 Pro` machine._

| Plugin Name | VST3 Works | AU Works |
| --- | --- | --- |
| Massive | Yes | ? |
| FM8 | Yes | ? |
| Reaktor 6 | Yes | ? |
| Replika | Yes | ? |
| Raum | Yes | ? |
| Vital | Yes | ? |
| Phasis | Yes | ? |
| Saltygrain | No | Yes |

Terminal tips for preset capture (e.g. spacebar preview) are described in [CLI.md](https://github.com/nafeu/dronmakr/blob/main/CLI.md).

## FAQ

Join the [Phrakture Discord Community](https://discord.gg/BysAyRje57) for better support.

> Where are my samples stored once generated?

Initially, all audio is stored in the `exports` folder and all generated MIDI is stored in `midi`. When using the auditioner, you can move samples into the `saved` or `trash` folders.

> "Reaktor 6" is being recognized as an effect instead of an instrument, what do I do?

You can use the `ASSERT_INSTRUMENT` env var to list any plugins that you want to launch strictly as an *instrument*.

## Contributing

- Contributors welcome! Open PRs or Github Issues

**Maintainers:** publishing a **GitHub Release** (not only a tag) runs [`.github/workflows/release-desktop.yml`](.github/workflows/release-desktop.yml) on macOS, Windows, and Linux and uploads the matching archives to that release.

On **Linux**, the tray icon may require GTK AppIndicator / `libappindicator` (or compatible) for `pystray`.

### Manual setup (from this repository)

Use this path if you want to run from source or contribute.

## License

[MIT](https://choosealicense.com/licenses/mit/)
