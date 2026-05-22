# dronmakr — Command-line interface

End-user desktop installs (prebuilt binaries) are documented in the [README](https://github.com/nafeu/dronmakr/blob/main/README.md).

---

## Verify the CLI

```sh
python dronmakr.py --version
```

```
dronmakr ■ v_._._
  github.com/nafeu/dronmakr (phrakturemusic@proton.me)
```

```sh
python dronmakr.py --help
```

```
Usage: dronmakr.py [OPTIONS] COMMAND [ARGS]...

  CLI entrypoint. With no subcommand, launches the unified web UI.

Options:
  -v, --version         Show the version and exit.
  --install-completion  Install completion for the current shell.
  --show-completion     Show completion for the current shell, to copy it or
                        customize the installation.
  --help                Show this message and exit.

Commands:
  generate-drone       Generate n iterations of samples (.wav) with...
  generate-beat        Generate n iterations of drum loops from...
  list                 List all available presets
  pack                 Rename all samples inside of saved folder for...
  webui                Run the unified web UI (auditionr + beatbuildr on...
  reset                Delete all files within the exports, trash and...
  generate-bass        Generate bass loops.
  generate-transition  Generate transition sounds (sweeps, risers, etc.).
```

## Patchcraftr (instrument & FX chains)

Author presets from the **desktop app tray**: **Launch patchcraftr** opens the Patchcraftr window (Pedalboard). Outputs go to your managed files (`config/presets.json` plus `.vstpreset` files under `presets/`).

Use **`python dronmakr.py list`** to print instrument and FX chain names for **`generate-drone`** (`--instrument` / `--effect`).

**macOS — Homebrew Python without Tk:** If Patchcraftr fails with `No module named '_tkinter'`, install bindings for your Python version (same major.minor), for example:

```sh
brew install python-tk@3.10
```

The official **python.org** macOS installers include Tk by default.

While a plug‑in editor is open, Patchcraftr streams a **simple live preview**: the same plug‑in instance is rendered in a background loop (`sounddevice`), so **parameter changes in the editor are heard on the next blocks**. Instruments use **built‑in MIDI** (a repeating middle‑C-style preview) instead of an external keyboard or MIDI port. FX slots hear a faint noise bed.

**Desktop build (PyInstaller):** The distributed app bundles Tcl/Tk via `collect_all('tkinter')` and does **not** require `brew install python-tk` on the machine where you **run** the app. Install `python-tk` only on the **build** host if its Python lacks `_tkinter` (so PyInstaller can copy the libraries into the bundle).

## Run the unified web UI

Start the app:

```sh
python dronmakr.py webui
```

Or with no command (same result):

```sh
python dronmakr.py
```

Open `http://0.0.0.0:3766` in a browser (or the URL printed in the terminal).

On **Auditionr → Generate Samples**, the dropdown lists **beat** and **drone** by default. Extra generators aligned with `generate-bass` / `generate-transition` (`reese`, `donk`, sweeps, etc.) are experimental: append **`?flags=extended_generators`** to the URL (comma-separated flags are supported), or **`?flags=all_cli_options`** for the same behaviour.

## Desktop runtime (menu bar / tray, from source)

The desktop launcher runs the Flask server in the background and shows a **menu bar** icon (macOS) or **system tray** icon (Windows). The UI opens in your default browser.

```sh
python dronmakr.py desktop
```

Optional verbose server logging:

```sh
python dronmakr.py desktop --debug
```

On first run you choose a `dronmakr-files` location. The app creates and manages `presets/`, `midi/`, `exports/`, `archive/`, `saved/`, `recordings/`, `splits/`, `trash/`, `packages/`, `history/`, `temp/`, `vst-preset-files/`, `config/`. Change it later under **Settings** (`FILES_ROOT`).

On **Linux**, the tray icon may require GTK AppIndicator / `libappindicator` (or compatible) for `pystray`.

## Local PyInstaller builds

Build outputs go to `dist/`; archives to `dist-artifacts/`. Release asset names must include `macos-arm64`, `macos-x64`, `linux-x64`, or `windows-x64` so the packaged app’s updater can find them.

**macOS icon:** GitHub Actions uses the **committed** file [`static/branding/macos/dronmakr.icns`](static/branding/macos/dronmakr.icns) (see [`static/branding/macos/README.md`](static/branding/macos/README.md)). Regenerate on a Mac with [`scripts/build_mac_app_icns.sh`](scripts/build_mac_app_icns.sh) (Pillow + `sips`/`iconutil`) after theme or artwork changes, then commit the `.icns`. `build_desktop.sh` only checks that the file exists.

**macOS:** PyInstaller emits **`dist/dronmakr.app`** (Finder bundle). **`build_desktop.sh`** produces **`dist-artifacts/dronmakr-v*-macos-*.tar.gz`** containing that `.app`, plus **`dist-artifacts/dronmakr-v*-macos-*.dmg`** (drag-to-Applications layout). CI publishes both formats for Apple Silicon builds.

Desktop builds populate **FFmpeg** under `resources/ffmpeg/` via [`scripts/vendor_ffmpeg.py`](scripts/vendor_ffmpeg.py) automatically when you run the build scripts (`build_desktop.sh` / `.ps1`/CI invoke it before PyInstaller). The packaged binary is used for **Folysplitr** browser recording uploads so end users don’t need a separate system FFmpeg install. Third-party FFmpeg notices ship as `resources/ffmpeg/THIRD_PARTY_FFMPEG.txt` inside the bundle; see [`resources/ffmpeg/LICENSE.third_party.ffmpeg`](resources/ffmpeg/LICENSE.third_party.ffmpeg) for redistribution notes.

**PySoundFile / PortAudio bundled libs:** Do **not** paste all of **`_soundfile_data`** into **`datas`** — macOS `.app` freezes need **`libsndfile*`** delivered as **`binaries`** ( **`pyinstaller-hooks-contrib`** `hook-soundfile`; **`libsndfile*`** beside **`soundfile.py`** on macOS/Windows wheels). **`desktop.spec`** only adds **`hiddenimports`** for **`soundfile`** / **`_soundfile*`** so analysis picks up that hook; Linux falls back to a system **`sndfile`** when wheels ship no bundled `.so`.

When running **`python webui.py` / tray from source**, Folysplitr still falls back to `ffmpeg` on your **`PATH`** (or **`DRONMAKR_FFMPEG_PATH`**) unless you ran the vendor script locally.

**macOS / Linux:**

```sh
./build_desktop.sh
```

**Windows (PowerShell):**

```powershell
.\build_desktop.ps1
```

## CLI command examples

### `generate-drone`

Options **`--length`** / **`--bars`** set how many **musical** bars the MIDI pattern spans (allowed: **4**, **8**, **16** default, **32**, **64**). **`--padded-silence`** appends extra **silent** bars at the end of the MIDI file (**0** default, or **4** / **8** / **16** / **32** / **64**), which lengthens Pedalboard/offline renders without changing the played pattern.

Post-processing uses legacy tokens and/or bracket-parameter steps. Separate steps with commas or semicolons.

**Default:** every synthesized export is **peak-normalized to −1 dBFS** after generation (same as the “Normalize” action). Steps you pass via `--post-processing` run **before** that final normalize.

**Bracket syntax:** `type:[key=value][key2=value2]`  
Examples:

- `fade:[style=in][duration_ms=2000]` — fade in over 2 seconds  
- `fade:[style=out][duration_ms=5000]` — fade out over 5 seconds  
- `filter:[kind=lpf][cutoff_hz=800]` — low-pass at 800 Hz  
- `filter:[kind=lpf][cutoff_hz=3200][resonance=0.4][steepness=0.85]` — sharper slope + resonance bump (~0–1 each)  
- `filter:[kind=bpf][low_hz=400][high_hz=5000][steepness=0]` — mild legacy-style band-pass (omit `steepness` in the spec for the same gentle default)  
- `delay:[time_mode=sync][bpm=128][division=1/8][feedback=0.55][mix=0.4][ping_pong=true][stereo_width=1][input_crossfeed=0.15][feedback_lowpass_hz=9000][feedback_highpass_hz=120]` — tempo-sync delay with ping-pong, filtering, and stereo widening (`time_mode=ms` uses `delay_ms=` instead of BPM/division). In Auditionr, use the **wand** next to BPM to fill tempo from the sample name when it contains tokens like `140bpm` or `140 BPM`.
- Chain: `fade:[style=out][duration_ms=5000];filter:[kind=lpf][cutoff_hz=800]`

Legacy tokens (e.g. `fade:in 2s`, `eq:lows +5db`) still work.

```sh
python dronmakr.py generate-drone \
  --name "my_drone" \
  --instrument "Reaktor 6" \
  --chart-name "minor" \
  --length 16 \
  --padded-silence 8 \
  --iterations 2 \
  --post-processing "normalize:[]"
```

```sh
python dronmakr.py generate-drone \
  --name "my_drone" \
  --post-processing "fade:[style=in][duration_ms=2000],eq:[band=lows][db=5]"
```

```sh
python dronmakr.py generate-drone --help
```

### `generate-beat`

```sh
python dronmakr.py generate-beat \
  --tempo 140 \
  --loops 4 \
  --pattern "default" \
  --kit "funky-dance" \
  --variants "1/2,3/4" \
  --swing 0.2 \
  --iterations 2
```

```sh
python dronmakr.py generate-beat --help
```

### `generate-bass` (`reese`, `donk`)

```sh
python dronmakr.py generate-bass reese \
  --tempo 170 \
  --bars 4 \
  --sound "sub;neuro;wave_a:saw;wave_b:tri" \
  --movement "filter_cutoff_low:180;filter_cutoff_high:3800" \
  --distortion "drive_soft:0.3;hard_mix:0.2" \
  --fx "stereo_width:0.75;chorus_mix:0.25"
```

```sh
python dronmakr.py generate-bass donk \
  --tempo 140 \
  --bars 2 \
  --sound "wave:sine;base_freq:52;pitch_start_semitones:18;sat_drive:0.45" \
  --iterations 3
```

```sh
python dronmakr.py generate-bass --help
python dronmakr.py generate-bass reese --help
python dronmakr.py generate-bass donk --help
```

### `generate-transition` (`sweep`, `closh`, `kickboom`, `longcrash`, `riser`, `drop`)

```sh
python dronmakr.py generate-transition sweep \
  --tempo 128 \
  --bars 8 \
  --sound "voice:noise;type:white" \
  --curve "shape:ease_in;peak_pos:0.9" \
  --filter "type:hpf;cutoff_low:250;cutoff_high:11000" \
  --iterations 2
```

```sh
python dronmakr.py generate-transition riser \
  --tempo 140 \
  --bars 4 \
  --peak-pos 1.0 \
  --build-shape ease_in \
  --longcrash-level 0.45 \
  --sweep-level 0.65
```

```sh
python dronmakr.py generate-transition drop \
  --tempo 140 \
  --bars 4 \
  --synth "voice:saw;freq_high:2600;freq_low:90;level:0.7" \
  --riser-level 0.4 \
  --synth-level 0.6
```

```sh
python dronmakr.py generate-transition --help
python dronmakr.py generate-transition sweep --help
python dronmakr.py generate-transition closh --help
python dronmakr.py generate-transition kickboom --help
python dronmakr.py generate-transition longcrash --help
python dronmakr.py generate-transition riser --help
python dronmakr.py generate-transition drop --help
```

### Utility commands (`list`, `pack`, `reset`, `webui`)

```sh
python dronmakr.py list --show-patterns
python dronmakr.py list --show-chain-plugins
```

```sh
python dronmakr.py pack --name "night textures" --artist "phrakture" --dry-run
python dronmakr.py reset --force
python dronmakr.py webui --port 3766 --open-browser
```

```sh
python dronmakr.py list --help
python dronmakr.py pack --help
python dronmakr.py reset --help
python dronmakr.py webui --help
```