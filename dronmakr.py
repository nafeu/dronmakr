import fnmatch
import time
import os
import sys
import webbrowser
import threading
import json
import random
import builtins
import subprocess

import typer
from build_preset import list_presets
from auditionr import main as run_server
from beatbuildr import main as run_beatbuildr_server
from generate_midi import generate_drone_midi
from generate_sample import generate_drone_sample, generate_beat_sample
from process_sample import process_drone_sample
from utils import (
    format_name,
    generate_beat_header,
    generate_beat_name,
    generate_drone_name,
    generate_id,
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
    """CLI entrypoint."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(generate_drone)


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


@cli.command(name="generate-beat")
def generate_beat(
    bpm: int = typer.Option(
        None, "--bpm", "-t", help="Beats per minute (random 80-180 if not specified)"
    ),
    loops: int = typer.Option(1, "--loops", "-l", help="Number of bars to generate"),
    pattern: str = typer.Option(
        None,
        "--pattern",
        "-p",
        help="Drum pattern style (random from config if not specified)",
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
            matching_patterns = [p for p in available_patterns if fnmatch.fnmatch(p, pattern)]
            if not matching_patterns:
                print(with_prompt(f"Error: No patterns match '{pattern}'"))
                sys.exit(1)
        else:
            if pattern not in available_patterns:
                print(with_prompt(f"Error: Pattern '{pattern}' not found in config/beat-patterns.json"))
                sys.exit(1)
            matching_patterns = [pattern]

    print(get_version())
    print(with_prompt(f"tempo"))
    print(with_prompt(f"  bpm                 {bpm if bpm else GENERATED_LABEL}"))
    print(with_prompt(f"  loops               {loops}"))
    print(
        with_prompt(f"pattern               {pattern if pattern else GENERATED_LABEL}")
    )
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

        # Determine BPM for this iteration
        current_bpm = bpm if bpm else random.randint(80, 180)

        # Determine pattern for this iteration
        if matching_patterns is not None:
            current_pattern = random.choice(matching_patterns)
        else:
            current_pattern = random.choice(available_patterns)

        # Generate beat name
        beat_name = generate_beat_name()

        # Generate output filename (metadata separated by ___ like generate-drone)
        sample_name = format_name(
            f"drumpattern___{beat_name}___{current_pattern}___{current_bpm}bpm___{generate_id()}"
        )
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
            swing=swing,
            play=False,  # Never play during generation
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


@cli.command()
def auditionr(
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Enable debug logs in auditionr"
    ),
    port: int = typer.Option(
        3766, "--port", "-p", help="The port for the webui auditionr to run on"
    ),
):
    """Run auditionr web server"""

    def _run_server():
        run_server(debug=debug, port=port)

    # Start the server in a background thread so we can wait briefly
    # and open the browser only if it appears to be running.
    server_thread = threading.Thread(target=_run_server, daemon=False)
    server_thread.start()

    # Give the server a moment to bind and start listening
    time.sleep(1)

    if server_thread.is_alive():
        try:
            webbrowser.open(f"http://localhost:{port}")
        except Exception:
            # Fail silently if the browser can't be opened.
            pass

    # Block until the server thread exits
    server_thread.join()


@cli.command()
def beatbuildr(
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Enable debug logs in beatbuildr"
    ),
    port: int = typer.Option(
        3767, "--port", "-p", help="The port for the beatbuildr webui to run on"
    ),
):
    """Run beatbuildr web server"""

    def _run_server():
        run_beatbuildr_server(debug=debug, port=port, open_browser=False)

    server_thread = threading.Thread(target=_run_server, daemon=False)
    server_thread.start()

    # Give the server a moment to bind and start listening
    time.sleep(1)

    if server_thread.is_alive():
        try:
            webbrowser.open(f"http://localhost:{port}")
        except Exception:
            # Fail silently if the browser can't be opened.
            pass

    server_thread.join()


if __name__ == "__main__":
    cli()
