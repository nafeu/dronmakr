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

## Build presets from your VST/AU library

```sh
python build_preset.py
```

When a plugin editor is open, you can press **spacebar** in the terminal to export and play a preview sample in the background (preset workflow tip).

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

```sh
python dronmakr.py generate-drone \
  --name "my_drone" \
  --instrument "Reaktor 6" \
  --chart-name "minor" \
  --iterations 2 \
  --post-processing "normalize,fade"
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