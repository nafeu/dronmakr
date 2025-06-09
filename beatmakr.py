import os
import random
import typer
import subprocess
import sys
from dotenv import load_dotenv
from pydub import AudioSegment
from pathlib import Path
from typing import List, Tuple

app = typer.Typer()


def get_random_sample(folder: Path) -> AudioSegment:
    files = [f for f in folder.iterdir() if f.suffix == ".wav"]
    if not files:
        raise typer.BadParameter(f"No WAV files found in {folder}")
    return AudioSegment.from_wav(random.choice(files))


def adjust_velocity(segment: AudioSegment, db_change: int) -> AudioSegment:
    return segment + db_change


def trim_decay(segment: AudioSegment, ms: int) -> AudioSegment:
    return segment[:ms].fade_out(ms)


@app.command()
def generate(
    bpm: int = typer.Option(120, help="Beats per minute"),
    bars: int = typer.Option(1, help="Number of bars to generate"),
    output: str = typer.Option("output.wav", help="Output filename"),
    humanize: bool = typer.Option(True, help="Apply humanization (velocity + timing)"),
    style: str = typer.Option(
        "breakbeat",
        help="Drum pattern style (breakbeat, dnb, trance, garage, halfstep)",
    ),
    play: bool = typer.Option(
        False, help="Open the exported file with the system's default WAV player"
    ),
):
    load_dotenv()

    beat_duration_ms = int((60 / bpm) * 1000 / 4)  # 16th note
    steps = bars * 16

    kicks = Path(random.choice(os.getenv("DRUM_KICK_PATHS", "").split(",")))
    hihats = Path(random.choice(os.getenv("DRUM_HIHAT_PATHS", "").split(",")))
    percs = Path(random.choice(os.getenv("DRUM_PERC_PATHS", "").split(",")))
    toms = Path(random.choice(os.getenv("DRUM_TOM_PATHS", "").split(",")))
    snares = Path(random.choice(os.getenv("DRUM_SNARE_PATHS", "").split(",")))
    shakers = Path(random.choice(os.getenv("DRUM_SHAKER_PATHS", "").split(",")))
    claps = Path(random.choice(os.getenv("DRUM_CLAP_PATHS", "").split(",")))
    cymbals = Path(random.choice(os.getenv("DRUM_CYMBAL_PATHS", "").split(",")))

    print(kicks)

    # Load samples
    kick = get_random_sample(kicks)
    snare = get_random_sample(snares)
    ghost_snare = get_random_sample(snares)
    hihat = get_random_sample(hihats)
    hihat_alt = get_random_sample(hihats)
    perc_a = get_random_sample(percs)
    perc_b = get_random_sample(percs)
    perc_c = get_random_sample(percs)
    clap = get_random_sample(claps)
    tom = get_random_sample(toms)
    shaker = get_random_sample(shakers)
    cymbal = get_random_sample(cymbals)

    # Patterns
    (
        kick_pattern,
        snare_pattern,
        ghost_snare_pattern,
        clap_pattern,
        hihat_pattern,
        hihat_alt_pattern,
        shaker_pattern,
        perc_a_pattern,
        perc_b_pattern,
        perc_c_pattern,
        tom_pattern,
        cymbal_pattern,
    ) = get_patterns(style, steps)

    track = AudioSegment.silent(duration=0)

    for i in range(steps):
        step = AudioSegment.silent(duration=beat_duration_ms)

        if kick_pattern[i]:
            step = step.overlay(kick[:beat_duration_ms])

        if snare_pattern[i]:
            snare_timing_offset = random.randint(-5, 5) if humanize else 0
            main_snare = snare[:beat_duration_ms]
            step = step.overlay(main_snare, position=max(0, snare_timing_offset))

        if ghost_snare_pattern[i]:
            ghost_snare_db = -6
            ghost_snare_timing_offset = random.randint(-5, 5) if humanize else 0
            main_ghost_snare = adjust_velocity(
                ghost_snare[:beat_duration_ms], ghost_snare_db
            )
            step = step.overlay(
                main_ghost_snare, position=max(0, ghost_snare_timing_offset)
            )

        if clap_pattern[i]:
            clap_timing_offset = random.randint(-5, 5) if humanize else 0
            main_clap = clap[:beat_duration_ms]
            step = step.overlay(main_clap, position=max(0, clap_timing_offset))

        if hihat_pattern[i]:
            hihat_db = random.randint(-6, 0) if humanize else 0
            hihat_sample = adjust_velocity(hihat[:beat_duration_ms], hihat_db)
            step = step.overlay(hihat_sample)

        if hihat_alt_pattern[i]:
            hihat_alt_db = random.randint(-6, 0) if humanize else 0
            hihat_alt_sample = adjust_velocity(
                hihat_alt[:beat_duration_ms], hihat_alt_db
            )
            step = step.overlay(hihat_alt_sample)

        if shaker_pattern[i]:
            shaker_db = random.randint(-6, 0) if humanize else 0
            shaker_sample = adjust_velocity(shaker[:beat_duration_ms], shaker_db)
            step = step.overlay(shaker_sample)

        if perc_a_pattern[i]:
            perc_a_timing_offset = random.randint(-5, 5) if humanize else 0
            perc_a_db = random.randint(-6, 0) if humanize else 0
            perc_a_sample = adjust_velocity(perc_a[:beat_duration_ms], perc_a_db)
            step = step.overlay(perc_a_sample, position=max(0, perc_a_timing_offset))

        if perc_b_pattern[i]:
            perc_b_timing_offset = random.randint(-5, 5) if humanize else 0
            perc_b_db = random.randint(-6, 0) if humanize else 0
            perc_b_sample = adjust_velocity(perc_b[:beat_duration_ms], perc_b_db)
            step = step.overlay(perc_b_sample, position=max(0, perc_b_timing_offset))

        if perc_c_pattern[i]:
            perc_c_timing_offset = random.randint(-5, 5) if humanize else 0
            perc_c_db = random.randint(-6, 0) if humanize else 0
            perc_c_sample = adjust_velocity(perc_c[:beat_duration_ms], perc_c_db)
            step = step.overlay(perc_c_sample, position=max(0, perc_c_timing_offset))

        if tom_pattern[i]:
            tom_timing_offset = random.randint(-5, 5) if humanize else 0
            tom_db = random.randint(-6, 0) if humanize else 0
            tom_sample = adjust_velocity(tom[:beat_duration_ms], tom_db)
            step = step.overlay(tom_sample, position=max(0, tom_timing_offset))

        if cymbal_pattern[i]:
            cymbal_timing_offset = random.randint(-5, 5) if humanize else 0
            cymbal_db = random.randint(-6, 0) if humanize else 0
            cymbal_sample = adjust_velocity(cymbal[:beat_duration_ms], cymbal_db)
            step = step.overlay(cymbal_sample, position=max(0, cymbal_timing_offset))

        track += step

    track.export(output, format="wav")
    typer.echo(f"✅ Drum loop exported to {output}")

    if play:
        typer.echo("▶️ Playing exported loop...")
        open_file_with_default_player(output)


def get_patterns(style: str, steps: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Returns (kick_pattern, snare_pattern, hihat_pattern) for a given style
    """
    # Default to 16 steps per bar
    base = 16
    mult = steps // base

    if style == "breakbeat":
        kick = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] * mult
        snar = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] * mult
        ghos = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0] * mult
        clap = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
        hhat = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * mult
        halt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
        shkr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
        prca = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
        prcb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
        prcc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
        tomm = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
        cymb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
    elif style == "trance":
        kick = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0] * mult
        snar = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
        ghos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
        clap = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] * mult
        hhat = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * mult
        halt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
        shkr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
        prca = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
        prcb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] * mult
        prcc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] * mult
        tomm = [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] * mult
        cymb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
    else:
        kick = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] * mult
        snar = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] * mult
        ghos = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0] * mult
        clap = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
        hhat = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * mult
        halt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
        shkr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
        prca = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
        prcb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
        prcc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
        tomm = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult
        cymb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * mult

    # Ensure all patterns are exactly 'steps' long
    def pad(pattern):
        return (pattern * ((steps // len(pattern)) + 1))[:steps]

    return (
        pad(kick),
        pad(snar),
        pad(ghos),
        pad(clap),
        pad(hhat),
        pad(halt),
        pad(shkr),
        pad(prca),
        pad(prcb),
        pad(prcc),
        pad(tomm),
        pad(cymb),
    )


def open_file_with_default_player(file_path: str):
    try:
        if sys.platform.startswith("darwin"):  # macOS
            subprocess.run(["open", file_path])
        elif sys.platform.startswith("win"):  # Windows
            os.startfile(file_path)
        elif sys.platform.startswith("linux"):  # Linux
            subprocess.run(["xdg-open", file_path])
        else:
            raise RuntimeError("Unsupported OS for auto-playing files.")
    except Exception as e:
        typer.echo(f"❌ Failed to open file: {e}")


if __name__ == "__main__":
    app()
