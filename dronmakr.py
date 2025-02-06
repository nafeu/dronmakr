import time
import os
import sys
import typer
from generate_midi import generate_midi
from generate_sample import generate_sample
from stretch_sample import stretch_sample
from utils import with_main_prompt as with_prompt, get_version, generate_id, generate_name, format_name, RED, RESET

EXPORTS_FOLDER = "exports"

cli = typer.Typer()

@cli.command()
def generate(
    name: str = typer.Option(None, help="Name for the generated sample."),
    chart_name: str = typer.Option(None, help="Chart name to filter chords/scales."),
    tags: str = typer.Option(None, help="Comma delimited list of tags to filter chords/scales."),
    roots: str = typer.Option(None, help="Comma delimited list of roots to filter chords/scales."),
    chart_type: str = typer.Option(None, help="Type of chart used for midi, either 'chord' or 'scale'."),
    style: str = typer.Option(None, help='Style of sample. One of "chaotic_arpeggio", "chord", "split_chord", "quantized_arpeggio".'),
):
    start_time = time.time()
    print(get_version())

    if not os.path.exists('presets/presets.json'):
        print(with_prompt("'presets/presets.json' does not exist, please run 'build_preset.py'"))
        sys.exit(1)

    filters = {}

    if tags: filters["tags"] = tags.split(",")
    if roots: filters["roots"] = roots.split(",")
    if chart_type: filters["type"] = chart_type
    if chart_name: filters["name"] = chart_name

    midi_file, selected_chart = generate_midi(style=style, filters=filters)
    sample_name = format_name(f"{selected_chart}_{name or generate_name()}_{generate_id()}")
    output_path = f"{EXPORTS_FOLDER}/{sample_name}"
    generated_sample = generate_sample(input_path=midi_file, output_path=f"{output_path}.wav")
    generated_sample_stretched = stretch_sample(input_path=generated_sample, output_path=f"{output_path}_stretched.wav")

    end_time = time.time()
    time_elapsed = round(end_time - start_time)
    print(f"{RED}■ completed in {time_elapsed}s{RESET}")
    print(with_prompt(f"generated: {midi_file}"))
    print(with_prompt(f"           {generated_sample}"))
    print(with_prompt(f"           {generated_sample_stretched}"))

if __name__ == "__main__":
    cli()
