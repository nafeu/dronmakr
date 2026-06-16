import mido
import sys
import shutil

# Define velocity display levels
VELOCITY_CHAR_MAP = {
    range(80, 128): "■",  # Loudest
    range(40, 80): "▦",  # Medium
    range(1, 40): "□",  # Soft
}

EMPTY_NOTE_CHAR = "╌"


def velocity_to_char(velocity):
    """Maps velocity to a Unicode block character."""
    for vel_range, char in VELOCITY_CHAR_MAP.items():
        if velocity in vel_range:
            return char
    return EMPTY_NOTE_CHAR  # Default empty character


def midi_note_to_label(note):
    """Converts MIDI note numbers to readable labels like 'C4, D#4'."""
    keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (note // 12) - 1
    return f"{keys[note % 12]}{octave}"


def find_first_track_with_notes(mid):
    """Finds the first MIDI track that contains note-on events with velocity > 0."""
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                return track  # Return the first valid track
    return None


def parse_midi(filename):
    """Parses a MIDI file and extracts note-on and note-off events from the first valid track."""
    mid = mido.MidiFile(filename)
    duration_ms = round(mid.length * 1000)

    track = find_first_track_with_notes(mid)
    if not track:
        print("No valid MIDI tracks with note data found.")
        return []

    notes = []
    active_notes = {}  # Track active notes (note number -> start time)
    current_time = 0

    for msg in track:
        current_time += msg.time  # Accumulate time

        if msg.type == "note_on":
            timestamp = int(
                current_time * 1000 / mid.ticks_per_beat
            )  # Convert ticks to ms

            if msg.velocity > 0:
                # Note-on: Store start time
                active_notes[msg.note] = (timestamp, msg.velocity)
            else:
                # Note-off (velocity=0): Determine duration and store it
                if msg.note in active_notes:
                    start_time, velocity = active_notes.pop(msg.note)
                    notes.append((start_time, timestamp, msg.note, velocity))

    # Handle any notes that never received a note-off event
    for note, (start_time, velocity) in active_notes.items():
        notes.append((start_time, duration_ms, note, velocity))

    return notes, duration_ms


def print_midi(filename, prompt_text=""):
    """Prints a static piano roll visualization for the first track of a MIDI file."""
    notes, duration_ms = parse_midi(filename)

    if not notes:
        print("No playable notes found in the file.")
        return

    # Determine note range (including all in-between notes)
    min_note = min(n[2] for n in notes)
    max_note = max(n[2] for n in notes)

    # Get terminal width and adjust number of columns
    term_width = shutil.get_terminal_size().columns - 40  # Adjust for labels
    num_columns = max(10, min(term_width, 64))  # Ensure at least 10 columns

    # Define time steps dynamically based on full MIDI length
    time_step_ms = max(1, duration_ms // num_columns)

    # Build empty grid covering all notes in range
    grid = {
        note: [EMPTY_NOTE_CHAR] * num_columns
        for note in range(max_note, min_note - 1, -1)
    }

    # Populate the grid with note data, ensuring notes are sustained
    for start_time, end_time, note, velocity in notes:
        start_col = min(start_time // time_step_ms, num_columns - 1)
        end_col = min(end_time // time_step_ms, num_columns - 1)

        for col in range(start_col, end_col + 1):
            grid[note][col] = velocity_to_char(velocity)

    # Print piano roll with note labels
    for note in range(max_note, min_note - 1, -1):
        note_label = f"{midi_note_to_label(note):>4} "  # Align note labels
        print(prompt_text + note_label + "".join(grid[note]))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python print_midi.py <midi_file>")
        sys.exit(1)

    midi_file = sys.argv[1]
    print_midi(midi_file)
