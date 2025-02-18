import os
import random
import sys
import json
import pretty_midi
from utils import (
    with_generate_midi_prompt as with_prompt,
    generate_id,
    generate_midi_header,
    YELLOW,
    RESET,
    format_name,
)
from print_midi import print_midi

MIDI_FOLDER = "midi"
CHORD_SCALE_LIST = "resources/chord-scale-data.json"

SUPPORTED_STYLES = [
    "chaos_expand_up",
    "chaos",
    "chord",
    "lead_flat",
    "lead_straight_eighth",
    "lead_straight_sixteenth",
    "lead",
    "quantized_straight_eighth",
    "quantized_straight_quarter",
    "quantized_up_down_eighth",
    "quantized_up_down_quarter",
    "split_chord",
]


def get_styles():
    return {"styles": SUPPORTED_STYLES}


def filter_chords(chords, filters):
    """Filters a chord collection based on optional criteria.

    - `tags`: List of tags (OR logic, case-insensitive)
    - `name`: Partial case-insensitive match
    - `root`: Exact case-insensitive match
    - `type`: Exact case-insensitive match ("scale" or "chord")
    """

    def matches_criteria(chord):
        """Checks if a chord matches all provided filters in priority order."""

        chord_name = chord["name"].lower()
        chord_root = chord["root"].lower()
        chord_type = chord["type"].lower()
        chord_tags = [tag.lower() for tag in chord["tags"]]

        if "root" in filters and filters["root"]:
            filter_roots = [r.lower() for r in filters["root"]]
            if chord_root not in filter_roots:
                return False  # Must match at least one root

        if "name" in filters and filters["name"]:
            if filters["name"].lower() not in chord_name:
                return False  # Partial match required

        if "tags" in filters and filters["tags"]:
            filter_tags = [tag.lower() for tag in filters["tags"]]
            if not any(tag in chord_tags for tag in filter_tags):
                return False  # At least one tag must match

        if "type" in filters and filters["type"]:
            if filters["type"].lower() != chord_type:
                return False  # Must match "scale" or "chord"

        return True

    # Apply filtering (return all entries if filters are empty)
    return [chord for chord in chords if matches_criteria(chord)]


def generate_midi(
    style,  # Style of playback
    output_name="",
    note_density=2,  # Notes per beat (higher = more active)
    duration_variance=0.5,  # Variance in note lengths (0 = fixed, 1 = max randomness)
    velocity_range=(80, 120),  # MIDI note velocity range
    num_bars=4,  # Default to 4 bars
    humanization=0.02,  # Time shift variance in seconds (default: 20ms)
    shift_octave_down=None,
    shift_root_note=None,
    filters={},
    iteration=None,
    iterations=None,
):
    print(generate_midi_header())
    """Generates a MIDI file based on the selected style with exact num_bars length."""
    if not style:
        style = random.choice(SUPPORTED_STYLES)

    if shift_octave_down is None:
        shift_octave_down = random.choice([True, False])

    if shift_root_note is None:
        shift_root_note = random.choice([True, False])

    if output_name:
        output_name = "_" + output_name

    if style not in SUPPORTED_STYLES:
        print(with_prompt(f"error: unsupported style '{style}'"))
        sys.exit()

    # Load chords from JSON file
    with open(CHORD_SCALE_LIST, "r") as f:
        chords = json.load(f)

    if not chords:
        print(with_prompt(f"error: no chords or scales found in '{CHORD_SCALE_LIST}'"))
        return

    if filters:
        chords = filter_chords(chords, filters)

    # Randomly select a chord
    random_chord_choice = random.choice(chords)
    chord = random_chord_choice["notes"]
    root = random_chord_choice["root"]
    chord_name = random_chord_choice["name"]
    track_name = format_name(f"{root}_{chord_name}{output_name}_{style}")

    print(
        with_prompt(
            f"writing {YELLOW}{root} {chord_name}{RESET} as {YELLOW}{style}{RESET}"
        )
    )

    if shift_root_note:
        print(with_prompt("shifting root one octave down"))

    if shift_octave_down:
        print(with_prompt("shifting all notes one octave down"))

    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name=track_name)

    # Define tempo and timing
    bpm = 120
    beats_per_bar = 4
    seconds_per_beat = 60.0 / bpm
    bar_length = beats_per_bar * seconds_per_beat
    total_duration = num_bars * bar_length  # Ensure exact num_bars length

    # Process chord notes
    midi_notes = []
    for note_str in chord:
        note_name, octave = note_str[:-1], int(note_str[-1])  # Extract note and octave
        midi_number = pretty_midi.note_name_to_number(note_name + str(octave))
        midi_notes.append(midi_number)

    # Drop the root note one octave down
    if shift_root_note:
        midi_notes[0] -= 12

    if shift_octave_down:
        midi_notes = [note - 12 for note in midi_notes]

    # 🎵 **MIDI Note Generation Based on Style**
    time = 0.0

    if style == "chaos":
        # **chaos:** random notes and timings given scale
        while time < total_duration:
            note = random.choice(midi_notes)
            min_duration = seconds_per_beat / note_density
            max_duration = min_duration * (1 + duration_variance)
            duration = random.uniform(min_duration, max_duration)
            end_time = min(time + duration, total_duration)
            velocity = random.randint(*velocity_range)

            midi_note = pretty_midi.Note(
                velocity=velocity, pitch=note, start=time, end=end_time
            )
            instrument.notes.append(midi_note)
            time = end_time  # Move forward to prevent overlap

    elif style == "chaos_expand_up":
        # **Chaos Expand Up:** Doubles available notes, second set is transposed up 1 octave
        expanded_midi_notes = midi_notes + [
            note + 12 for note in midi_notes
        ]  # Transpose copy up 1 octave

        while time < total_duration:
            note = random.choice(expanded_midi_notes)  # Choose from expanded range
            min_duration = seconds_per_beat / note_density
            max_duration = min_duration * (1 + duration_variance)
            duration = random.uniform(min_duration, max_duration)
            end_time = min(time + duration, total_duration)
            velocity = random.randint(*velocity_range)

            midi_note = pretty_midi.Note(
                velocity=velocity, pitch=note, start=time, end=end_time
            )
            instrument.notes.append(midi_note)
            time = end_time  # Move forward to prevent overlap

    elif style == "split_chord":
        # **Split Chord:** Play full chord at start and again at the middle
        velocity = random.randint(*velocity_range)
        for start_time in [0.0, total_duration / 2]:
            start_time += random.uniform(-humanization, humanization)
            for note in midi_notes:
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=note,
                        start=max(0.0, start_time),
                        end=start_time + bar_length,
                    )
                )

    elif style == "quantized_straight_quarter":
        # **Quantized Arpeggio:** Play notes one at a time, quarter-notes, looping lowest to highest
        note_duration = seconds_per_beat * 1  # Quarter-note duration
        while time < total_duration:
            for note in midi_notes:
                if time >= total_duration:
                    break
                velocity = random.randint(*velocity_range)
                start_time = max(0.0, time)
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=note,
                        start=start_time,
                        end=min(start_time + note_duration, total_duration),
                    )
                )
                time += note_duration

    elif style == "quantized_straight_eighth":
        # **Quantized Arpeggio:** Play notes one at a time, eighth-notes, looping lowest to highest
        note_duration = seconds_per_beat * 0.5  # Eighth-note duration
        while time < total_duration:
            for note in midi_notes:
                if time >= total_duration:
                    break
                velocity = random.randint(*velocity_range)
                start_time = max(0.0, time)
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=note,
                        start=start_time,
                        end=min(start_time + note_duration, total_duration),
                    )
                )
                time += note_duration

    elif style == "quantized_up_down_quarter":
        # **Up-Down Arpeggio:** Ascends then descends, quarter-note timing
        note_duration = seconds_per_beat * 1  # Quarter-note duration
        up_down_pattern = (
            midi_notes + midi_notes[::-1][1:-1]
        )  # Ascend & descend, avoid repeat
        while time < total_duration:
            for note in up_down_pattern:
                if time >= total_duration:
                    break
                velocity = random.randint(*velocity_range)
                start_time = max(0.0, time)
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=note,
                        start=start_time,
                        end=min(start_time + note_duration, total_duration),
                    )
                )
                time += note_duration

    elif style == "quantized_up_down_eighth":
        # **Up-Down Arpeggio:** Ascends then descends, eigth-note timing
        note_duration = seconds_per_beat * 0.5  # Eigth-note duration
        up_down_pattern = (
            midi_notes + midi_notes[::-1][1:-1]
        )  # Ascend & descend, avoid repeat
        while time < total_duration:
            for note in up_down_pattern:
                if time >= total_duration:
                    break
                velocity = random.randint(*velocity_range)
                start_time = max(0.0, time)
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=note,
                        start=start_time,
                        end=min(start_time + note_duration, total_duration),
                    )
                )
                time += note_duration

    elif style == "lead":
        # **Lead Melody:** Semi-random movement within the scale
        note_durations = [
            seconds_per_beat * d for d in [0.25, 0.5, 1, 2]
        ]  # 16th, 8th, quarter, half
        total_notes = random.randint(
            len(midi_notes), 16
        )  # Random note count (scale size to 16)

        # 🎶 Start on a random note
        current_note = random.choice(midi_notes)

        for _ in range(total_notes):
            velocity = random.randint(*velocity_range)
            start_time = max(0.0, time)
            duration = random.choice(note_durations)  # Random note length
            end_time = min(start_time + duration, total_duration)

            instrument.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=current_note,
                    start=start_time,
                    end=end_time,
                )
            )

            time += duration  # Move forward

            # 🎼 Melody Movement: Sometimes move up/down stepwise before jumping randomly
            if random.random() < 0.6:  # 60% chance to move stepwise
                idx = midi_notes.index(current_note)
                if (
                    random.random() < 0.5 and idx < len(midi_notes) - 1
                ):  # Move up if possible
                    current_note = midi_notes[idx + 1]
                elif idx > 0:  # Move down if possible
                    current_note = midi_notes[idx - 1]
            else:
                current_note = random.choice(midi_notes)  # 40% chance for a random jump

    elif style == "lead_flat":
        # **Lead Melody (Flat):** Constant eighth notes, structured movement
        note_duration = seconds_per_beat * 0.5  # Fixed eighth-note duration
        total_notes = random.randint(
            len(midi_notes), 16
        )  # Random note count (scale size to 16)

        # 🎶 Start on a random note
        current_note = random.choice(midi_notes)

        for _ in range(total_notes):
            velocity = random.randint(*velocity_range)
            start_time = max(0.0, time)
            end_time = min(start_time + note_duration, total_duration)

            instrument.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=current_note,
                    start=start_time,
                    end=end_time,
                )
            )

            time += note_duration  # Move forward

            # 🎼 Melody Movement: Mostly stepwise, sometimes jumps
            if random.random() < 0.7:  # 70% chance to move stepwise
                idx = midi_notes.index(current_note)
                if (
                    random.random() < 0.5 and idx < len(midi_notes) - 1
                ):  # Move up if possible
                    current_note = midi_notes[idx + 1]
                elif idx > 0:  # Move down if possible
                    current_note = midi_notes[idx - 1]
            else:
                current_note = random.choice(midi_notes)  # 30% chance for a jump

    elif style == "lead_straight_sixteenth":
        # **Lead (Straight, Sixteenth Notes):** Plays notes from low to high once
        note_duration = seconds_per_beat * 0.25  # Sixteenth-note duration
        for note in midi_notes:
            if time >= total_duration:
                break
            velocity = random.randint(*velocity_range)
            start_time = max(0.0, time)
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=note,
                    start=start_time,
                    end=min(start_time + note_duration, total_duration),
                )
            )
            time += note_duration

    elif style == "lead_straight_eighth":
        # **Lead (Straight, Eighth Notes):** Plays notes from low to high once
        note_duration = seconds_per_beat * 0.5  # Eighth-note duration
        for note in midi_notes:
            if time >= total_duration:
                break
            velocity = random.randint(*velocity_range)
            start_time = max(0.0, time)
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=note,
                    start=start_time,
                    end=min(start_time + note_duration, total_duration),
                )
            )
            time += note_duration

    else:
        # **Straight Chord:** (default) Play all notes together from start to finish
        velocity = random.randint(*velocity_range)
        start_time = max(0.0, random.uniform(-humanization, humanization))
        for note in midi_notes:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=velocity, pitch=note, start=start_time, end=total_duration
                )
            )

    # Add instrument **without a name** to MIDI object
    midi.instruments.append(instrument)

    # Ensure the MIDI file is exactly `num_bars` long
    if midi.get_end_time() < total_duration:
        empty_note = pretty_midi.Note(
            velocity=0, pitch=60, start=midi.get_end_time(), end=total_duration
        )
        instrument.notes.append(empty_note)

    # Write to a MIDI file
    os.makedirs(MIDI_FOLDER, exist_ok=True)

    output_path = f"{MIDI_FOLDER}/{track_name}_{generate_id()}.mid"
    midi.write(output_path)

    print(f"{YELLOW}│{RESET}")
    print_midi(output_path, f"{YELLOW}│  {RESET}")

    print(f"{YELLOW}│{RESET}")

    return (output_path, f"{root} {chord_name}")


def main():
    args = sys.argv[1:]
    if not args:
        generate_midi()
        return

    generate_midi(args[0])


if __name__ == "__main__":
    main()
