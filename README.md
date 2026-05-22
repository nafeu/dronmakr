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

1. Open **[Releases](https://github.com/nafeu/dronmakr/releases/latest)** and download the build for your platform:
   - **macOS (Apple Silicon):** Prefer **`dronmakr-v*-macos-arm64.dmg`** (drag **dronmakr.app** into **Applications**). A **`dronmakr-v*-macos-arm64.tar.gz`** is also available if you prefer unpacking manually.
   - **macOS (Intel):** **`dronmakr-v*-macos-x64.tar.gz`** from a maintainer-built artifact (Intel builds are not produced by CI today).
   - **Linux (x86_64):** `dronmakr-v*-linux-x64.tar.gz`
   - **Windows:** `dronmakr-v*-windows-x64.zip`
2. **macOS (DMG):** Open the `.dmg`, drag **`dronmakr.app`** into **Applications**, then eject the disk image. Launch **dronmakr** from Launchpad / Spotlight (**⌘Space**, type **dronmakr**).
3. **macOS / Linux (tar.gz)** or **Windows (zip):** Extract the archive. On macOS you get **`dronmakr.app`**; on Linux / Windows you get a **`dronmakr`** folder with the launcher inside.
4. Run:
   - **macOS:** open **`dronmakr.app`** (from `/Applications` or the extracted folder).
   - **Linux:** `./dronmakr/dronmakr`
   - **Windows:** `dronmakr.exe`
5. Runtime behavior: builds use **no separate terminal window** on **macOS** (launcher is Finder-only). Watch the **menu bar** (near the Wi‑Fi / clock area) for the **dronmakr tray icon**. Open **Open dronmakr in browser** from its menu—the local server listens on `127.0.0.1`. On **Windows / Linux**, a **console window** may remain open (logs). **On first launch** you choose where to store **`dronmakr-files`** (often via onboarding in the browser).

GitHub CI builds are **not Apple-notarized**. After downloading, macOS may block the embedded Python libraries (`library load disallowed by system policy`). Typical workarounds:

- **Preferred:** Open **System Settings → Privacy & Security** after a blocked launch and choose **Open Anyway**, **or** Control-click **`dronmakr.app` → Open** and confirm once.
- **Alternative:** clear quarantine flags after you trust the build (copied apps inherit them until cleared):

  ```sh
  xattr -dr com.apple.quarantine /Applications/dronmakr.app
  ```

For distribution without these prompts you need your own **Apple Developer Program** subscription, **Developer ID Application** signing, **`codesign`**, **notarization** (`notarytool`), and **stapling**. That pipeline is outside this repo’s automated CI today.

**If launching from Spotlight / Finder “does nothing”:** the packaged app waits for startup to finish—check the **menu bar icon** before assuming it failed (see step **5**). To surface errors, open **Terminal** and run:

```sh
/Applications/dronmakr.app/Contents/MacOS/dronmakr
```

On recent builds (or after a silent failure), **`~/Library/Application Support/dronmakr/last-startup-error.txt`** may contain Python tracebacks captured from the launcher.

Packaged desktop builds also poll **GitHub Releases**: when an update exists, native tray prompts can offer **Open updater** on startup; otherwise use **Check for updates…** or **Updater (<tag>)** in the tray. The updater opens a Tk window in a separate process (download progress, install / DMG reveal). Clearing `PLUGIN_PATHS` after first run stays empty—defaults apply only when `settings.json` is first created.

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

**Version bump:** from the repo root, run [`scripts/bump_version.sh`](scripts/bump_version.sh) with **`major`**, **`minor`**, or **`patch`** ([SemVer](https://semver.org/) bump applied to [`version.py`](version.py)). It edits `version.py`, commits **`Bump version to v…`**, creates an annotated tag **`v*.*.*`**, and **`git push`es** branch + tag to **`origin`** (use **`--dry-run`** to preview). Example patch release:

```sh
./scripts/bump_version.sh patch
```

On **Linux**, the tray icon may require GTK AppIndicator / `libappindicator` (or compatible) for `pystray`.

### Manual setup (from this repository)

Use this path if you want to run from source or contribute.

## License

[MIT](https://choosealicense.com/licenses/mit/)
