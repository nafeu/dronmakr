import fnmatch
import time
import os
import sys
import json
import random
import builtins
import subprocess

import typer

from settings import ensure_settings
from build_preset import list_presets
from webui import run as run_webui
from generate_midi import generate_drone_midi, get_pattern_config
from generate_sample import generate_drone_sample, generate_beat_sample
from generate_transition import generate_sweep_sample
from process_sample import process_drone_sample
from utils import (
    format_name,
    generate_beat_header,
    generate_beat_name,
    generate_drone_name,
    generate_id,
    generate_transition_header,
    get_cli_version,
    get_version,
    RED,
    rename_samples,
    RESET,
    with_main_prompt as with_prompt,
    with_generate_beat_prompt,
    delete_all_files,
    EXPORTS_DIR,
    MIDI_DIR,
    TRASH_DIR,
)
from version import __version__

GENERATED_LABEL = f"{RED}...{RESET}"

cli = typer.Typer(invoke_without_command=True)


def open_files_with_default_player(file_paths):
    """Open one or more files with the system default application."""
    if not file_paths:
        return

    try:
        if sys.platform.startswith("darwin"):
            subprocess.run(["open"] + file_paths)
        elif sys.platform.startswith("win"):
            for file_path in file_paths:
                os.startfile(file_path)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open"] + file_paths)
    except Exception as e:
        print(with_prompt(f"Failed to open files: {e}"))


def version_callback(ctx: typer.Context, value: bool):
    if value:
        print(get_cli_version())
        raise typer.Exit()


@cli.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
):
    """CLI entrypoint. With no subcommand, launches the unified web UI."""
    ensure_settings()
    if ctx.invoked_subcommand is None:
        ctx.invoke(webui, debug=False, port=3766, open_browser=True)


@cli.command(name="generate-drone")
def generate_drone(
    name: str = typer.Option(
        None, "--name", "-n", help="Name for the generated sample."
    ),
    notes: str = typer.Option(
        None,
        "--notes",
        "-N",
        help="Comma separated list of notes with octave numbers (e.g., C2,D#3,F#3). Overrides other MIDI generation options.",
    ),
    chart_name: str = typer.Option(
        None, "--chart-name", "-c", help="Chart name to filter chords/scales."
    ),
    instrument: str = typer.Option(
        None, "--instrument", "-i", help="Name of the instrument."
    ),
    effect: str = typer.Option(
        None, "--effect", "-e", help="Name of the effect or chain."
    ),
    tags: str = typer.Option(
        None,
        "--tags",
        "-t",
        help="Comma delimited list of tags to filter chords/scales.",
    ),
    roots: str = typer.Option(
        None,
        "--roots",
        "-r",
        help="Comma delimited list of roots to filter chords/scales.",
    ),
    chart_type: str = typer.Option(
        None,
        "--chart-type",
        "-y",
        help="Type of chart used for midi, either 'chord' or 'scale'.",
    ),
    pattern: str = typer.Option(
        None,
        "--pattern",
        "-s",
        help="Name of midi pattern used to play virtual instrument.",
    ),
    iterations: int = typer.Option(
        1,
        "--iterations",
        "-I",
        help="Number of times to generate samples (default: 1).",
    ),
    shift_octave_down: bool = typer.Option(
        None, "--shift-octave-down", "-O", help="Shift all notes one octave down."
    ),
    shift_root_note: bool = typer.Option(
        None, "--shift-root-note", "-R", help="Shift root note one octave down."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="Verify CLI options"),
    log_server: bool = typer.Option(
        False, "--log-server", "-v", help="Run logs as server mode"
    ),
    play: bool = typer.Option(
        False,
        "--play",
        help="Open all generated files with the system's default player",
    ),
):
    """Generate n iterations of samples (.wav) with parameters"""
    start_time = time.time()

    if not log_server:
        print(get_version())

    if not os.path.exists("presets/presets.json"):
        print(
            with_prompt(
                "'presets/presets.json' does not exist, please run 'build_preset.py'"
            )
        )
        sys.exit(1)

    print(with_prompt(f"sample name          {name if name else GENERATED_LABEL}"))
    print(with_prompt(f"sound design"))
    print(
        with_prompt(
            f"  instrument         {instrument if instrument else GENERATED_LABEL}"
        )
    )
    print(with_prompt(f"  effect             {effect if effect else GENERATED_LABEL}"))
    if notes:
        print(with_prompt("notes                " + notes))
    else:
        print(with_prompt(f"thematics"))
        print(
            with_prompt(
                f"  chart name         {chart_name if chart_name else GENERATED_LABEL}"
            )
        )
        print(with_prompt(f"  tags               {tags if tags else GENERATED_LABEL}"))
        print(
            with_prompt(f"  roots              {roots if roots else GENERATED_LABEL}")
        )
        print(
            with_prompt(
                f"  chart type         {chart_type if chart_type else GENERATED_LABEL}"
            )
        )
        print(with_prompt(f"midi customization"))
        print(
            with_prompt(
                f"  pattern            {pattern if pattern else GENERATED_LABEL}"
            )
        )
        print(
            with_prompt(
                f"  shift octave down  {shift_octave_down if shift_octave_down else GENERATED_LABEL}"
            )
        )
        print(
            with_prompt(
                f"  shift root note    {shift_root_note if shift_root_note else GENERATED_LABEL}"
            )
        )

    print(
        with_prompt(
            f"iterations           {iterations if iterations else GENERATED_LABEL}"
        )
    )
    print(with_prompt(f"play when done        {play}"))
    print(f"{RED}│{RESET}")

    if dry_run:
        print(f"{RED}■ dry run completed{RESET}")
        return ["midi/dry_run_example.mid", "exports/dry_run_export.wav"]

    filters = {}

    if tags:
        filters["tags"] = tags.split(",")
    if roots:
        filters["roots"] = roots.split(",")
    if chart_type:
        filters["type"] = chart_type
    if chart_name:
        filters["name"] = chart_name

    results = []

    for iteration in range(iterations):
        if iterations > 1:
            print(f"{RED}■ preparing")
            print(f"{RED}│{RESET}   iteration {iteration + 1} of {iterations}")
            print(f"{RED}│{RESET}")

        midi_file, selected_chart = generate_drone_midi(
            pattern=pattern,
            shift_octave_down=shift_octave_down,
            shift_root_note=shift_root_note,
            filters=filters,
            notes=notes.split(",") if notes else None,
        )
        sample_name = format_name(
            f"{name or generate_drone_name()}_-_{selected_chart}_-_{generate_id()}"
        )
        output_path = f"{EXPORTS_DIR}/{sample_name}"
        generated_sample = generate_drone_sample(
            input_path=midi_file,
            output_path=f"{output_path}.wav",
            instrument=instrument,
            effect=effect,
        )
        (
            generated_sample_stretched,
            generated_sample_stretched_reverberated,
            generated_sample_stretched_reverberated_transposed,
        ) = process_drone_sample(input_path=generated_sample)
        results.append(midi_file)
        results.append(generated_sample)
        results.append(generated_sample_stretched)
        results.append(generated_sample_stretched_reverberated)
        results.append(generated_sample_stretched_reverberated_transposed)

    end_time = time.time()
    time_elapsed = round(end_time - start_time)
    print(f"{RED}■ completed in {time_elapsed}s{RESET}")

    for index, result in enumerate(results):
        if index == 0:
            print(with_prompt(f"generated: {result}"))
        else:
            print(with_prompt(f"           {result}"))

    # Open all generated .wav files if play is enabled
    if play and results:
        wav_files = [f for f in results if f.endswith(".wav")]
        if wav_files:
            open_files_with_default_player(wav_files)

    return results


def _load_drum_kits_for_cli():
    """Load drum kits from config/drum-kits.json for CLI use."""
    path = "config/drum-kits.json"
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


@cli.command(name="generate-beat")
def generate_beat(
    bpm: int = typer.Option(
        None, "--bpm", "-t", help="Beats per minute (random 80-180 if not specified)"
    ),
    loops: int = typer.Option(1, "--loops", "-l", help="Number of bars per pattern loop"),
    pattern: str = typer.Option(
        None,
        "--pattern",
        "-p",
        help="Drum pattern style (random from config if not specified)",
    ),
    kit: str = typer.Option(
        None,
        "--kit",
        "-k",
        help="Drum kit name from config/drum-kits.json (uses env sample folders if not specified)",
    ),
    randomize_tempo: bool = typer.Option(
        False,
        "--randomize-tempo",
        "-r",
        help="Ignore pattern's saved tempo and use random BPM (ignored if --bpm is set)",
    ),
    swing: float = typer.Option(
        0.0,
        "--swing",
        "-w",
        min=0.0,
        max=1.0,
        help="Rhythmic swing amount between 0 (straight) and 1 (strong swing).",
    ),
    humanize: bool = typer.Option(True, help="Apply humanization (velocity + timing)"),
    play: bool = typer.Option(
        False,
        help="Open the exported file with the system's default WAV player",
    ),
    iterations: int = typer.Option(
        1,
        "--iterations",
        "-I",
        help="Number of times to generate beats (default: 1).",
    ),
):
    """Generate n iterations of drum loops from env-configured sample folders."""
    start_time = time.time()

    # Load available patterns from config
    try:
        with open("config/beat-patterns.json", "r") as f:
            beat_patterns_data = json.load(f)
            available_patterns = (
                builtins.list(beat_patterns_data.keys()) if beat_patterns_data else []
            )
            if not available_patterns:
                print(
                    with_prompt(
                        f"Error: No patterns found in config/beat-patterns.json"
                    )
                )
                sys.exit(1)
    except FileNotFoundError:
        print(with_prompt("Error: config/beat-patterns.json not found"))
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(with_prompt(f"Error: Invalid JSON in config/beat-patterns.json: {e}"))
        sys.exit(1)

    # Resolve pattern: exact match, wildcard match (*), or random if not specified
    matching_patterns = None
    if pattern is not None and pattern:
        if "*" in pattern:
            matching_patterns = [
                p for p in available_patterns if fnmatch.fnmatch(p, pattern)
            ]
            if not matching_patterns:
                print(with_prompt(f"Error: No patterns match '{pattern}'"))
                sys.exit(1)
        else:
            if pattern not in available_patterns:
                print(
                    with_prompt(
                        f"Error: Pattern '{pattern}' not found in config/beat-patterns.json"
                    )
                )
                sys.exit(1)
            matching_patterns = [pattern]

    # Resolve kit if --kit specified
    kit_paths = None
    kit_name_for_file = ""
    if kit is not None and kit:
        drum_kits = _load_drum_kits_for_cli()
        if kit not in drum_kits:
            print(with_prompt(f"Error: Kit '{kit}' not found in config/drum-kits.json"))
            sys.exit(1)
        kit_paths = drum_kits[kit]
        if not isinstance(kit_paths, dict):
            kit_paths = {}
        kit_name_for_file = format_name(kit)

    print(get_version())
    print(with_prompt(f"tempo"))
    print(with_prompt(f"  bpm                 {bpm if bpm else GENERATED_LABEL}"))
    print(with_prompt(f"  randomize-tempo     {randomize_tempo}"))
    print(with_prompt(f"  loops               {loops}"))
    print(
        with_prompt(f"pattern               {pattern if pattern else GENERATED_LABEL}")
    )
    print(with_prompt(f"kit                   {kit if kit else GENERATED_LABEL}"))
    print(with_prompt(f"swing                 {swing}"))
    print(with_prompt(f"humanize              {humanize}"))
    print(with_prompt(f"play when done        {play}"))
    print(
        with_prompt(
            f"iterations            {iterations if iterations else GENERATED_LABEL}"
        )
    )
    print(f"{RED}│{RESET}")

    results = []

    for iteration in range(iterations):
        if iterations > 1:
            print(f"{RED}■ preparing")
            print(f"{RED}│{RESET}   iteration {iteration + 1} of {iterations}")
            print(f"{RED}│{RESET}")

        # Determine pattern for this iteration
        if matching_patterns is not None:
            current_pattern = random.choice(matching_patterns)
        else:
            current_pattern = random.choice(available_patterns)

        raw = beat_patterns_data.get(current_pattern, {})
        gs, ts, ln, meta_tempo, meta_swing = (None, None, None, None, None)
        if isinstance(raw, dict) and raw:
            gs, ts, ln, meta_tempo, meta_swing = get_pattern_config(raw)

        pattern_config = None
        if gs is not None:
            pattern_config = {"gridSize": gs, "timeSignature": ts, "length": ln}

        # Determine BPM: --bpm overrides all; else --randomize-tempo ignores _meta tempo; else use _meta tempo or random
        if bpm is not None:
            current_bpm = bpm
        elif randomize_tempo:
            current_bpm = random.randint(80, 180)
        elif meta_tempo is not None:
            current_bpm = int(meta_tempo)
        else:
            current_bpm = random.randint(80, 180)

        # Use pattern's swing from _meta when available, otherwise CLI swing
        current_swing = meta_swing if meta_swing is not None else swing

        # Generate beat name
        beat_name = generate_beat_name()

        # Generate output filename: drumpattern___beatname___pattern___kit?___bpm___id
        name_parts = ["drumpattern", beat_name, current_pattern]
        if kit_name_for_file:
            name_parts.append(kit_name_for_file)
        name_parts.append(f"{current_bpm}bpm")
        name_parts.append(generate_id())
        sample_name = format_name("___".join(name_parts))
        output_path = f"{EXPORTS_DIR}/{sample_name}.wav"

        print(generate_beat_header())
        print(with_generate_beat_prompt(f"bpm: {current_bpm}"))
        print(with_generate_beat_prompt(f"pattern: {current_pattern}"))

        generate_beat_sample(
            bpm=current_bpm,
            bars=loops,
            output=output_path,
            humanize=humanize,
            style=current_pattern,
            swing=current_swing,
            play=False,  # Never play during generation
            pattern_config=pattern_config,
            kit_paths=kit_paths,
        )

        results.append(output_path)

    end_time = time.time()
    time_elapsed = round(end_time - start_time)
    print(f"{RED}■ completed in {time_elapsed}s{RESET}")

    for index, result in enumerate(results):
        if index == 0:
            print(with_prompt(f"generated: {result}"))
        else:
            print(with_prompt(f"           {result}"))

    # Open all files at once if play is enabled
    if play and results:
        open_files_with_default_player(results)

    return results


# ---------------------------------------------------------------------------
# generate-transition (group with sub-commands)
# ---------------------------------------------------------------------------

transition_app = typer.Typer(help="Generate transition sounds (sweeps, risers, etc.).")


@transition_app.command("sweep")
def transition_sweep(
    tempo: int = typer.Option(
        120,
        "--tempo",
        "-t",
        help="Tempo in BPM. With --bars, determines sweep length (default: 120).",
    ),
    bars: int = typer.Option(
        2,
        "--bars",
        "-b",
        help="Length in bars (default: 2). At 120 BPM, 2 bars ≈ 4 seconds.",
    ),
    cutoff_low: int | None = typer.Option(
        None,
        "--cutoff-low",
        "-l",
        help="Filter cutoff (Hz) at sweep start. Lower = darker/muffled start. Random 300–700 if omitted.",
    ),
    cutoff_high: int | None = typer.Option(
        None,
        "--cutoff-high",
        "-H",
        help="Filter cutoff (Hz) at peak. Higher = brighter sweep. Random 10k–18k if omitted.",
    ),
    decay: float | None = typer.Option(
        None,
        "--decay",
        "-d",
        help="Decay rate after peak (1–10). Higher = faster fade-out. Random 2–6 if omitted.",
    ),
    peak_pos: float | None = typer.Option(
        None,
        "--peak-pos",
        "-P",
        help="Where the peak lands, 0.0–1.0 (0.5 = middle). Random 0.4–0.6 if omitted.",
    ),
    noise_level: float | None = typer.Option(
        None,
        "--noise-level",
        "-n",
        help="Noise amplitude (0.2–1.0). Higher = louder sweep. Random 0.45–0.8 if omitted.",
    ),
    noise_type: str | None = typer.Option(
        None,
        "--noise-type",
        "-N",
        help="Noise spectrum: white (flat), pink (1/f, warmer), brown (1/f², rumbly), blue (bright/hissy). Random if omitted.",
    ),
    filter_order: int | None = typer.Option(
        None,
        "--filter-order",
        "-f",
        help="Lowpass filter order: 2, 4, or 6. Higher = steeper rolloff. Random if omitted.",
    ),
    build_shape: str | None = typer.Option(
        None,
        "--build-shape",
        "-s",
        help="Build curve: ease_in (slow start), linear, ease_out (slow end). Random if omitted.",
    ),
    tremolo_depth: float | None = typer.Option(
        None,
        "--tremolo-depth",
        help="Gain LFO depth (0–1). 0=off, 0.9=volume dips to 10% at trough. Random 0.4–0.9 if omitted.",
    ),
    tremolo_rate_min: float | None = typer.Option(
        None,
        "--tremolo-rate-min",
        help="Tremolo LFO rate (Hz) at sweep start. Oscillations speed up toward peak. Random 1–4 if omitted.",
    ),
    tremolo_rate_max: float | None = typer.Option(
        None,
        "--tremolo-rate-max",
        help="Tremolo LFO rate (Hz) at peak. Random 10–25 if omitted.",
    ),
    phaser: bool | None = typer.Option(
        None,
        "--phaser/--no-phaser",
        help="Add phaser effect. Random if neither specified.",
    ),
    chorus: bool | None = typer.Option(
        None,
        "--chorus/--no-chorus",
        help="Add chorus effect. Random if neither specified.",
    ),
    flanger: bool | None = typer.Option(
        None,
        "--flanger/--no-flanger",
        help="Add flanger effect (Chorus with short delay + feedback). Random if neither specified.",
    ),
    play: bool = typer.Option(
        False,
        "--play",
        "-p",
        help="Open the exported file with the system's default WAV player.",
    ),
):
    """Generate a noise riser with LFO-modulated filter cutoff (techno/trance/house).

    Use --noise-type to choose white, pink, brown, or blue noise. Builds from
    muffled to bright, peaks, then smooth decay. Omit sound-design options to
    randomize them for variation.
    """
    start_time = time.time()

    # Resolve build_shape for DSP
    valid_shapes = ("ease_in", "linear", "ease_out")
    build_shape_typed = None
    if build_shape is not None:
        if build_shape not in valid_shapes:
            print(with_prompt(f"Error: --build-shape must be one of {valid_shapes}"))
            raise typer.Exit(1)
        build_shape_typed = build_shape

    # Resolve noise_type for DSP
    valid_noise = ("white", "pink", "brown", "blue")
    noise_type_typed = None
    if noise_type is not None:
        nt = noise_type.lower()
        if nt not in valid_noise:
            print(with_prompt(f"Error: --noise-type must be one of {valid_noise}"))
            raise typer.Exit(1)
        noise_type_typed = nt

    print(get_version())
    print(generate_transition_header())
    print(with_prompt(f"sweep"))
    print(with_prompt(f"  tempo               {tempo}"))
    print(with_prompt(f"  bars                {bars}"))
    for label, val in [
        ("cutoff_low", cutoff_low),
        ("cutoff_high", cutoff_high),
        ("decay", decay),
        ("peak_pos", peak_pos),
        ("noise_level", noise_level),
        ("noise_type", noise_type),
        ("filter_order", filter_order),
        ("build_shape", build_shape),
        ("tremolo_depth", tremolo_depth),
        ("tremolo_rate_min", tremolo_rate_min),
        ("tremolo_rate_max", tremolo_rate_max),
        ("phaser", phaser),
        ("chorus", chorus),
        ("flanger", flanger),
    ]:
        print(with_prompt(f"  {label:<18} {val if val is not None else GENERATED_LABEL}"))
    print(with_prompt(f"  play when done      {play}"))
    print(f"{RED}│{RESET}")

    beat_name = generate_beat_name()
    name_parts = ["transition_sweep", beat_name, f"{tempo}bpm", f"{bars}bars", generate_id()]
    sample_name = format_name("___".join(name_parts))
    output_path = f"{EXPORTS_DIR}/{sample_name}.wav"

    output_path, params_used = generate_sweep_sample(
        tempo=tempo,
        bars=bars,
        output=output_path,
        cutoff_low=cutoff_low,
        cutoff_high=cutoff_high,
        decay_rate=decay,
        peak_pos=peak_pos,
        noise_level=noise_level,
        noise_type=noise_type_typed,
        filter_order=filter_order,
        build_shape=build_shape_typed,
        tremolo_depth=tremolo_depth,
        tremolo_rate_min=tremolo_rate_min,
        tremolo_rate_max=tremolo_rate_max,
        phaser=phaser,
        chorus=chorus,
        flanger=flanger,
    )

    end_time = time.time()
    time_elapsed = round(end_time - start_time)
    print(f"{RED}■ completed in {time_elapsed}s{RESET}")
    print(with_prompt(f"generated: {output_path}"))
    t = params_used
    mod_str = ", ".join(m for m in ["phaser", "chorus", "flanger"] if t.get(m))
    fx_str = f", fx=[{mod_str}]" if mod_str else ""
    print(with_prompt(f"  used: {t['noise_type']} noise, cutoff {t['cutoff_low']}–{t['cutoff_high']}Hz, tremolo depth={t['tremolo_depth']:.2f} rate={t['tremolo_rate_min']:.1f}–{t['tremolo_rate_max']:.1f}Hz{fx_str}"))

    if play:
        open_files_with_default_player([output_path])

    return [output_path]


cli.add_typer(transition_app, name="generate-transition")


@cli.command()
def list(
    show_chain_plugins: bool = typer.Option(
        False,
        "--show-chain-plugins",
        "-p",
        help="List all the plugins used within an effect chain",
    ),
    show_patterns: bool = typer.Option(
        False,
        "--show-patterns",
        "-t",
        help="List all available midi patterns and descriptions",
    ),
):
    """List all available presets"""
    list_presets(show_chain_plugins=show_chain_plugins, show_patterns=show_patterns)


@cli.command()
def pack(
    pack_name: str = typer.Option(
        None, "--name", "-n", help="Name for the sample pack"
    ),
    artist_name: str = typer.Option(
        None, "--artist", "-a", help="Artist name for sample pack"
    ),
    affix: bool = typer.Option(
        False, "--affix", "-f", help="Simply attach the meta to the end of the filename"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-d", help="Check updated filenames"
    ),
    delimiter: str = typer.Option(
        "^", "--delimiter", "-l", help="Original filename delimiter"
    ),
):
    """Rename all samples inside of saved folder for packaging"""
    rename_samples(
        pack_name=pack_name,
        artist_name=artist_name,
        dry_run=dry_run,
        affix=affix,
        delimiter=delimiter,
    )


@cli.command()
def webui(
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Enable debug logs in the web server"
    ),
    port: int = typer.Option(3766, "--port", "-p", help="Port for the unified web UI"),
    open_browser: bool = typer.Option(
        True,
        "--open-browser/--no-open-browser",
        help="Open the web UI in the default browser on start",
    ),
):
    """Run the unified web UI (auditionr + beatbuildr on one server)."""
    run_webui(debug=debug, port=port, open_browser=open_browser)


@cli.command()
def reset(
    force: bool = typer.Option(
        False, "--force", "-f", help="Empty directories without confirmation"
    )
):
    """Delete all files within the exports, trash and midi directories"""
    if not force:
        confirmation = typer.confirm(
            "Are you sure you want to empty the exports, midi and trash directories?"
        )
        if not confirmation:
            return
    directories = [EXPORTS_DIR, MIDI_DIR, TRASH_DIR]
    for directory in directories:
        deleted_files_count = delete_all_files(directory)
        print(with_prompt(f"Deleted {deleted_files_count} files in {directory}"))


if __name__ == "__main__":
    cli()
