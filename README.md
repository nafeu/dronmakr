# ┌ dronmakr ┐

> pronounced "drone maker"

Python-based music sample generator powered by [pedalboard.io](https://spotify.github.io/pedalboard/index.html), [paulstretch](https://github.com/canyondust/paulstretch_python) and your own **VST Instruments & FX** library.

## Installation & Setup

_Built in Python `3.10.16`_

```sh
git clone https://github.com/nafeu/dronmakr.git
cd dronmakr
```

Setup Virtual Environment

```sh
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

Install Dependencies

```sh
pip install -r requirements.txt
```

## Usage

`python dronmakr.py --help`

```
Usage: python dronmakr.py [OPTIONS] COMMAND [ARGS]...

  Default to 'generate' if no command is given.

Commands:
  generate  Generate n iterations of samples (.wav) with parameters
  preset    Use interactive preset builder
  list      List all available presets
  pack      Rename all samples inside of saved folder for packaging
  server    Run auditioner web server
```

`python dronmakr.py generate --help`

```
Usage: dronmakr.py generate [OPTIONS]

  Generate n iterations of samples (.wav) with parameters

Options:
  -n, --name TEXT           Name for the generated sample.
  -c, --chart-name TEXT     Chart name to filter chords/scales.
  -i, --instrument TEXT     Name of the instrument.
  -e, --effect TEXT         Name of the effect or chain.
  -t, --tags TEXT           Comma delimited list of tags to filter
                            chords/scales.
  -r, --roots TEXT          Comma delimited list of roots to filter
                            chords/scales.
  -y, --chart-type TEXT     Type of chart used for midi, either 'chord' or
                            'scale'.
  -s, --style TEXT          Style of sample. One of "chaotic_arpeggio",
                            "chord", "split_chord", "quantized_arpeggio".
  -I, --iterations INTEGER  Number of times to generate samples (default: 1).
                            [default: 1]
  -O, --shift-octave-down   Shift all notes one octave down.
  -R, --shift-root-note     Shift root note one octave down.
  -d, --dry-run             Verify CLI options
  -v, --log-server          Run logs as server mode
  --help                    Show this message and exit.
```

## License

[MIT](https://choosealicense.com/licenses/mit/)