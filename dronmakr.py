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
from generate_transition import (
    generate_closh_sample,
    generate_sweep_sample,
    parse_closh_config,
    parse_sweep_config,
)
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
    tempo: int = typer.Option(120, "--tempo", "-t", help="Tempo in BPM (default: 120)."),
    bars: int = typer.Option(2, "--bars", "-b", help="Length in bars (default: 2)."),
    sound: str | None = typer.Option(
        None,
        "--sound",
        "-s",
        help="Sound: voice (noise|sine|saw|tri|square). Oscillator params: freq_low, freq_high, level, pulse_width, pwm_sweep, detune_cents, detune_mix, resonance.",
    ),
    curve: str | None = typer.Option(
        None,
        "--curve",
        "-c",
        help="Build curve: shape, decay_rate, peak_pos. E.g. shape:ease_in;decay_rate:4;peak_pos:0.5.",
    ),
    filter_str: str | None = typer.Option(
        None,
        "--filter",
        "-f",
        help="Filter: cutoff_low, cutoff_high. E.g. cutoff_low:200;cutoff_high:12000.",
    ),
    tremolo: str | None = typer.Option(
        None,
        "--tremolo",
        "-T",
        help="Tremolo: rate_min_hz, rate_max_hz, depth. E.g. depth:0.8;rate_min_hz:2;rate_max_hz:20.",
    ),
    phaser: str | None = typer.Option(
        None,
        "--phaser",
        help="Phaser: rate_hz, depth, centre_frequency_hz, feedback, mix. Use _ for random. Omit to randomize enable.",
    ),
    chorus: str | None = typer.Option(
        None,
        "--chorus",
        help="Chorus: rate_hz, depth, centre_delay_ms, feedback, mix.",
    ),
    flanger: str | None = typer.Option(
        None,
        "--flanger",
        help="Flanger: rate_hz, depth, centre_delay_ms, feedback, mix.",
    ),
    disable: str | None = typer.Option(
        None,
        "--disable",
        "-d",
        help="Disable effects: comma-separated list of tremolo,phaser,chorus,flanger or fx (disables all).",
    ),
    iterations: int = typer.Option(1, "--iterations", "-n", help="Number of samples to generate."),
    play: bool = typer.Option(False, "--play", "-p", help="Open output with default WAV player."),
):
    """Generate a noise riser (techno/trance/house).

    Params use key:value;key2:value2 syntax. Use _ or empty for random.
    Quote values: --sound "voice:noise;type:white" --sound "voice:square;freq_low:110"
    """
    start_time = time.time()
    iterations = max(1, iterations)

    print(get_version())
    print(generate_transition_header())
    print(with_prompt(f"sweep"))
    print(with_prompt(f"  tempo               {tempo}"))
    print(with_prompt(f"  bars                {bars}"))
    print(with_prompt(f"  iterations          {iterations}"))
    print(with_prompt(f"  play when done      {play}"))
    print(f"{RED}│{RESET}")

    results = []
    for i in range(iterations):
        config = parse_sweep_config(
            sound=sound,
            curve=curve,
            filter_str=filter_str,
            tremolo=tremolo,
            phaser=phaser,
            chorus=chorus,
            flanger=flanger,
            disable=disable,
        )
        beat_name = generate_beat_name()
        name_parts = ["transition_sweep", beat_name, f"{tempo}bpm", f"{bars}bars", generate_id()]
        sample_name = format_name("___".join(name_parts))
        output_path = f"{EXPORTS_DIR}/{sample_name}.wav"
        output_path, params_used = generate_sweep_sample(tempo=tempo, bars=bars, output=output_path, config=config)
        results.append(output_path)
        t = params_used
        voice = t.get("voice", "noise")
        if voice == "noise":
            voice_desc = f"{t.get('noise_type', 'white')} noise"
        else:
            voice_desc = f"{voice} {t.get('osc_freq_low', 0):.0f}–{t.get('osc_freq_high', 0):.0f}Hz"
        mod_str = ", ".join(m for m in ["phaser", "chorus", "flanger"] if t.get(m))
        fx_str = f", fx=[{mod_str}]" if mod_str else ""
        if iterations == 1:
            print(with_prompt(f"generated: {output_path}"))
            print(with_prompt(f"  used: {voice_desc}, cutoff {t['cutoff_low']}–{t['cutoff_high']}Hz, tremolo depth={t['tremolo_depth']:.2f} rate={t['tremolo_rate_min']:.1f}–{t['tremolo_rate_max']:.1f}Hz{fx_str}"))
        else:
            print(with_prompt(f"  [{i + 1}/{iterations}] {output_path}"))

    end_time = time.time()
    time_elapsed = round(end_time - start_time)
    print(f"{RED}■ completed in {time_elapsed}s{RESET}")
    if iterations > 1:
        for r in results:
            print(with_prompt(f"generated: {r}"))
    if play and results:
        open_files_with_default_player(results)
    return results


@transition_app.command("closh")
def transition_closh(
    tempo: int = typer.Option(120, "--tempo", "-t", help="Tempo in BPM (default: 120)."),
    bars: int = typer.Option(2, "--bars", "-b", help="Length in bars (default: 2)."),
    reverb: str | None = typer.Option(
        None,
        "--reverb",
        "-r",
        help="Reverb: wet_level, length_sec, decay_sec, early_reflections, highpass_hz, tail_diffusion (0.65-0.9). Use _ for random.",
    ),
    delay: str | None = typer.Option(
        None,
        "--delay",
        "-d",
        help="Tempo-synced delay: division (1/4|1/8|1/8d|1/16|1/16d|1/32), feedback, mix. Omit or 'off' to disable.",
    ),
    iterations: int = typer.Option(1, "--iterations", "-n", help="Number of samples to generate."),
    play: bool = typer.Option(False, "--play", "-p", help="Open output with default WAV player."),
):
    """Generate washed clap transition: random clap from DRUM_CLAP_PATHS with long reverb.

    Params use key:value;key2:value2. Use _ for random. E.g. --reverb "room_size:0.95;wet_level:0.9"
    """
    start_time = time.time()
    iterations = max(1, iterations)

    config = parse_closh_config(reverb=reverb, delay=delay)

    print(get_version())
    print(generate_transition_header())
    print(with_prompt(f"closh"))
    print(with_prompt(f"  tempo               {tempo}"))
    print(with_prompt(f"  bars                {bars}"))
    print(with_prompt(f"  iterations          {iterations}"))
    print(with_prompt(f"  delay               {'on' if config['delay_enabled'] else 'off'}"))
    print(f"{RED}│{RESET}")

    results = []
    for i in range(iterations):
        beat_name = generate_beat_name()
        name_parts = ["transition_closh", beat_name, f"{tempo}bpm", f"{bars}bars", generate_id()]
        sample_name = format_name("___".join(name_parts))
        output_path = f"{EXPORTS_DIR}/{sample_name}.wav"
        output_path, params_used = generate_closh_sample(
            tempo=tempo, bars=bars, output=output_path, config=config
        )
        results.append(output_path)
        p = params_used
        if iterations == 1:
            rev_str = f"wet={p['reverb_wet_level']:.2f} len={p['reverb_length_sec']:.1f}s decay={p['reverb_decay_sec']:.1f}s"
            dl_str = f" delay={p['delay_division']} fb={p['delay_feedback']:.2f} mix={p['delay_mix']:.2f}" if p["delay_enabled"] else ""
            print(with_prompt(f"generated: {output_path}"))
            print(with_prompt(f"  used: {rev_str}{dl_str}"))
        else:
            print(with_prompt(f"  [{i + 1}/{iterations}] {output_path}"))

    end_time = time.time()
    time_elapsed = round(end_time - start_time)
    print(f"{RED}■ completed in {time_elapsed}s{RESET}")
    if iterations > 1:
        for r in results:
            print(with_prompt(f"generated: {r}"))
    if play and results:
        open_files_with_default_player(results)
    return results


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
