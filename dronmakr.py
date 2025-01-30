import random
import sys
import json
import pretty_midi

def generate_midi(
    note_density=2,  # How many notes per beat (higher = more active)
    duration_variance=0.5,  # How much note lengths vary (0 = fixed, 1 = max randomness)
    velocity_range=(80, 120),  # Randomized note velocity range (MIDI loudness)
    num_bars=16  # Default to 16 bars instead of 8
):
    # Load chords from JSON file
    with open("chords.json", "r") as f:
        chords = json.load(f)

    if not chords:
        print("Error: No chords found in chords.json")
        return

    # Randomly select a chord
    chord = random.choice(chords)["notes"]

    # Create a new PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    # Define tempo and timing
    bpm = 120
    beats_per_bar = 4
    seconds_per_beat = 60.0 / bpm
    bar_length = beats_per_bar * seconds_per_beat
    total_duration = num_bars * bar_length  # Now 16 bars instead of 8

    # Process chord notes
    midi_notes = []

    for note_str in chord:
        note_name, octave = note_str[:-1], int(note_str[-1])  # Extract note and octave
        midi_number = pretty_midi.note_name_to_number(note_name + str(octave))
        midi_notes.append(midi_number)

    # Drop the root note one octave down
    midi_notes[0] -= 12

    # Generate non-overlapping arpeggiated pattern
    time = 0.0
    while time < total_duration:
        # Pick a random note from the chord
        note = random.choice(midi_notes)

        # Randomize note duration (ensuring no overlap)
        min_duration = seconds_per_beat / note_density
        max_duration = min_duration * (1 + duration_variance)
        duration = random.uniform(min_duration, max_duration)

        # Ensure the note ends within the total duration
        end_time = min(time + duration, total_duration)

        # Random velocity (loudness)
        velocity = random.randint(*velocity_range)

        # Create the note
        midi_note = pretty_midi.Note(
            velocity=velocity,
            pitch=note,
            start=time,
            end=end_time
        )
        instrument.notes.append(midi_note)

        # Move time forward to prevent overlap
        time = end_time

    # Add instrument to MIDI object
    midi.instruments.append(instrument)

    # Write to a MIDI file
    output_path = "output.mid"
    midi.write(output_path)

    print(f"MIDI file generated: {output_path} (Chord: {chord})")

def main():
    args = sys.argv[1:]
    if not args:
        generate_midi()
        return

    command = args[0]

    if command == "-h" or command == "--help":
        print("Usage: python cli.py [options...]")
        print("\nOptions:")
        print("- -h, --help: Show this help message")
        print("- -v, --version: Display the program version")
    elif command in ("-v", "--version"):
        print("1.0.0")
    else:
        print(f"Unknown option or command: {command}. Use --help for available options.")

if __name__ == "__main__":
    main()
