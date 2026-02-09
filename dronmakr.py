import time
import os
import sys
import typer
from generate_midi import generate_drone_midi
from generate_sample import generate_drone_sample
from process_sample import process_drone_sample
from utils import (
    format_name,
    generate_id,
    generate_name,
    get_version,
    get_cli_version,
    RED,
    rename_samples,
    RESET,
    with_main_prompt as with_prompt,
    delete_all_files,
    EXPORTS_DIR,
    MIDI_DIR,
    TRASH_DIR,
)
from build_preset import list_presets
from server import main as run_server
from version import __version__

GENERATED_LABEL = f"{RED}...{RESET}"

cli = typer.Typer(invoke_without_command=True)


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
            f"{name or generate_name()}_-_{selected_chart}_-_{generate_id()}"
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
def server(
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Enable debug logs in server"
    ),
    port: int = typer.Option(
        3766, "--port", "-p", help="The port for the webui server on run on"
    ),
):
    """Run auditioner web server"""
    run_server(debug=debug, port=port)


if __name__ == "__main__":
    cli()
