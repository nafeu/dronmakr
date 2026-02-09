import typer
import pretty_midi
import soundfile as sf
import numpy as np
import os
import tempfile
import subprocess
import sys


def play_file(file_path):
    """Plays an audio file using the default system player."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["afplay", file_path], check=True)
        elif sys.platform == "win32":
            os.startfile(file_path)
        else:
            subprocess.run(["xdg-open", file_path], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        typer.echo(
            f"Error playing file: {e}. Please ensure a default audio player is configured."
        )


def main(
    notes: str = typer.Argument(..., help="Comma-separated notes, e.g., C4,E4,G4"),
    pattern: str = typer.Option(
        "chord",
        "--pattern",
        help="Playback pattern: 'chord' or 'walk'.",
        case_sensitive=False,
    ),
):
    """Generates and plays a preview of the given MIDI notes as a WAV file."""
    if pattern not in ["chord", "walk"]:
        typer.echo("Error: Invalid pattern. Choose 'chord' or 'walk'.")
        raise typer.Exit(1)

    try:
        note_list = [note.strip() for note in notes.split(",")]
        midi_numbers = [pretty_midi.note_name_to_number(n) for n in note_list]
    except ValueError as e:
        typer.echo(f"Error: Invalid note format: {e}. Example format: C4,G#4,F5")
        raise typer.Exit(1)

    # Create a PrettyMIDI object
    midi_data = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    )

    # Add notes to the instrument based on the pattern
    if pattern == "chord":
        for note_number in midi_numbers:
            note = pretty_midi.Note(velocity=100, pitch=note_number, start=0, end=1.5)
            instrument.notes.append(note)
    elif pattern == "walk":
        # Eighth notes at 120 BPM
        note_duration = 0.25  # 60s / 120bpm / 2
        current_time = 0.0
        for note_number in midi_numbers:
            note = pretty_midi.Note(
                velocity=100,
                pitch=note_number,
                start=current_time,
                end=current_time + note_duration,
            )
            instrument.notes.append(note)
            current_time += note_duration

    midi_data.instruments.append(instrument)

    # Synthesize the MIDI data into audio
    typer.echo("Generating audio preview...")
    audio_data = midi_data.synthesize(fs=44100)

    # Write to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        sf.write(tmpfile.name, audio_data, 44100)
        temp_file_path = tmpfile.name

    typer.echo(f"Playing notes: {notes}")
    play_file(temp_file_path)

    # Clean up the temporary file
    os.unlink(temp_file_path)
    typer.echo("Preview finished.")


if __name__ == "__main__":
    typer.run(main)
