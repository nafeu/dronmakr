import os
import random
import sys
import json
import pretty_midi
from typing import List, Tuple

from utils import (
    with_generate_drone_midi_prompt as with_prompt,
    generate_id,
    generate_drone_midi_header,
    YELLOW,
    RESET,
    format_name,
    MIDI_DIR,
)
from bundle_paths import bundled_asset_path
from paths import get_managed_file
from print_midi import print_midi

CHORD_SCALE_LIST = str(bundled_asset_path("resources", "chord-scale-data.json"))
BEAT_PATTERNS_FILE = get_managed_file("config", "beat-patterns.json")
BEAT_PATTERNS_SAMPLE_FILE = str(bundled_asset_path("resources", "beat-patterns-sample.json"))
_BEAT_PATTERNS_CACHE = None


def ensure_beat_patterns_file():
    """
    Ensure config/beat-patterns.json exists.
    If not, copy from resources/beat-patterns-sample.json.
    """
    import os
    import shutil

    if not os.path.exists(BEAT_PATTERNS_FILE):
        os.makedirs(os.path.dirname(BEAT_PATTERNS_FILE), exist_ok=True)
        if os.path.exists(BEAT_PATTERNS_SAMPLE_FILE):
            shutil.copy2(BEAT_PATTERNS_SAMPLE_FILE, BEAT_PATTERNS_FILE)
        else:
            raise FileNotFoundError(
                f"Sample beat patterns file not found at {BEAT_PATTERNS_SAMPLE_FILE}"
            )


SUPPORTED_PATTERNS_INFO = [
    ("chaos", "random notes and timings given scale/chord"),
    (
        "chaos_expand_up",
        "random notes and timings with second set of notes transposed up 1 octave",
    ),
    ("lead", "semi-random movement within the scale/chord"),
    ("lead_flat", "constant eighth notes, structured movement"),
    ("lead_straight_eighth", "plays eighth notes from low to high, looping for the full phrase"),
    ("lead_straight_sixteenth", "plays sixteenth notes from low to high, looping for the full phrase"),
    (
        "quantized_straight_eighth",
        "play eighth-notes one at a time, looping lowest to highest",
    ),
    (
        "quantized_straight_quarter",
        "play quarter-notes one at a time, looping lowest to highest",
    ),
    ("quantized_up_down_eighth", "ascends then descends, eigth-note timing"),
    ("quantized_up_down_quarter", "ascends then descends, quarter-note timing"),
    ("split_chord", "play full chord at start and again at the middle"),
    (
        "reverse_arpeggio_quarter",
        "quarter-notes from highest to lowest, looping continuously",
    ),
    (
        "strum_burst_quarter",
        "each beat: chord notes staggered upward quickly inside the beat",
    ),
    (
        "pedal_high_arp_eighth",
        "sustain top note across the phrase while other tones arpeggiate in eighths",
    ),
    (
        "bounce_octaves_extremes",
        "alternating bass and highest scale tone, eighth-note staccato",
    ),
    (
        "triplet_straight",
        "eighth-note triplets cycling pitches low to high",
    ),
    (
        "syncopated_offbeat_eighth",
        "hits on the '&' subdivisions only, looping through chord tones",
    ),
    (
        "two_hand_alternate",
        "alternate lower-half and upper-half chord voices each eighth",
    ),
    (
        "rhythmic_long_short_short",
        "each beat subdivided long-short-short with stepwise-or-random melody motion",
    ),
    ("cluster_hit_bar", "short full-chord stab at each bar line, rests between"),
    (
        "double_layer_octave_ping",
        "each melody note doubled an octave higher, eighth-note pulses",
    ),
    (
        "pedal_low_arp_quarter",
        "sustain bottom pitch while remaining chord tones outline in quarter-notes",
    ),
    (
        "neighbor_pair_eighth",
        "adjacent scale steps as eighth-note pairs, wrapping around sorted chord pitch order",
    ),
    (
        "strum_burst_quarter_down",
        "each beat: chord pitches staggered quickly from high to low inside the beat",
    ),
    (
        "skip_step_quarter_arpeggio",
        "quarter-notes every other chord tone (skip-one index), looping the scale",
    ),
    (
        "random_dyad_stabs",
        "short bursts of random two-note pairs (single note when only one tone exists)",
    ),
    (
        "dotted_rhythm_triplet_cells",
        "per beat dotted-eighth-plus-sixteenth feel cycling through sorted pitches",
    ),
    (
        "accent_root_strong_weak_inner",
        "strong beats emphasize root pitch, weak beats use inner chord tones quarter-length",
    ),
    (
        "bar_rotating_arpeggio_quarter",
        "every bar the quarter arpeggio starts on the next rotated voice of the chord",
    ),
    (
        "stagger_voice_pad_hit",
        "each bar softly stagger chord entrances then hold slightly short of bar end",
    ),
    (
        "shuffled_chord_voice_quarters",
        "fresh random permutation of chord tones replayed each bar on quarter pulses",
    ),
]

SUPPORTED_PATTERNS = [item[0] for item in SUPPORTED_PATTERNS_INFO]

# Drone MIDI: fixed bar counts for CLI + Auditionr Generate Samples.
DRONE_MIDI_LENGTH_BARS_ALLOWED = frozenset({4, 8, 16, 32, 64})
DRONE_MIDI_PADDING_BARS_ALLOWED = frozenset({0, 4, 8, 16, 32, 64})
DEFAULT_DRONE_MIDI_LENGTH_BARS = 16
DEFAULT_DRONE_MIDI_PADDING_BARS = 0
DRONE_MIDI_TEMPO_BPM = 120
DRONE_MIDI_BEATS_PER_BAR = 4


def drone_midi_render_duration_sec(
    num_bars: int,
    padded_silence_bars: int = 0,
    *,
    tempo_bpm: float = DRONE_MIDI_TEMPO_BPM,
) -> float:
    """Wall-clock seconds for ``num_bars`` of musical content plus optional trailing silence."""
    bar_length = DRONE_MIDI_BEATS_PER_BAR * (60.0 / float(tempo_bpm))
    return (int(num_bars) + int(padded_silence_bars)) * bar_length


def midi_musical_end_seconds(midi_path: str) -> float:
    """Latest note-off/end time in a MIDI file (ignores trailing padding meta)."""
    import pretty_midi  # noqa: PLC0415

    return float(pretty_midi.PrettyMIDI(midi_path).get_end_time())


def coerce_drone_midi_length_bars(value, *, default: int = DEFAULT_DRONE_MIDI_LENGTH_BARS) -> int:
    """Parse UI/API ``lengthBars`` (must be 4, 8, 16, 32, or 64)."""
    if value is None or value == "":
        return default
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError("lengthBars must be an integer") from None
    if n not in DRONE_MIDI_LENGTH_BARS_ALLOWED:
        raise ValueError(
            f"lengthBars must be one of {sorted(DRONE_MIDI_LENGTH_BARS_ALLOWED)}"
        )
    return n


def coerce_drone_midi_padding_bars(
    value, *, default: int = DEFAULT_DRONE_MIDI_PADDING_BARS
) -> int:
    """Parse UI/API ``paddedSilenceBars`` (0 or 4/8/16/32/64)."""
    if value is None or value == "":
        return default
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError("paddedSilenceBars must be an integer") from None
    if n not in DRONE_MIDI_PADDING_BARS_ALLOWED:
        raise ValueError(
            f"paddedSilenceBars must be one of {sorted(DRONE_MIDI_PADDING_BARS_ALLOWED)}"
        )
    return n


def get_patterns():
    """Return list of MIDI pattern names for drone generation (used by webui/auditionr)."""
    return sorted(SUPPORTED_PATTERNS)


DRUM_ROW_ORDER = [
    "kick", "snar", "ghos", "clap", "hhat", "halt",
    "shkr", "prca", "prcb", "prcc", "tomm", "cymb",
]

# Beat pattern configuration
GRID_STEPS_PER_BEAT = {"1/16": 4, "1/16t": 6}
DEFAULT_GRID_SIZE = "1/16"
DEFAULT_TIME_SIGNATURE = [4, 4]
DEFAULT_PATTERN_LENGTH = 1


def compute_steps(grid_size: str, time_signature: list, length: int) -> int:
    """Compute total steps from pattern config."""
    steps_per_beat = GRID_STEPS_PER_BEAT.get(grid_size, 4)
    beats_per_bar = time_signature[0] if time_signature and len(time_signature) >= 1 else 4
    return steps_per_beat * beats_per_bar * length


def get_pattern_config(style_patterns: dict) -> tuple:
    """Extract config from pattern data. Returns (grid_size, time_signature, length, tempo, swing)."""
    meta = style_patterns.get("_meta") if isinstance(style_patterns.get("_meta"), dict) else {}
    grid_size = meta.get("gridSize") or meta.get("grid_size") or DEFAULT_GRID_SIZE
    if grid_size not in GRID_STEPS_PER_BEAT:
        grid_size = DEFAULT_GRID_SIZE
    ts = meta.get("timeSignature") or meta.get("time_signature") or DEFAULT_TIME_SIGNATURE
    if not isinstance(ts, list) or len(ts) < 2:
        ts = DEFAULT_TIME_SIGNATURE
    length = meta.get("length", DEFAULT_PATTERN_LENGTH)
    if not isinstance(length, (int, float)) or length < 1:
        length = DEFAULT_PATTERN_LENGTH
    tempo = meta.get("tempo")
    if tempo is not None and (not isinstance(tempo, (int, float)) or tempo < 1):
        tempo = None
    swing = meta.get("swing")
    if swing is not None and (not isinstance(swing, (int, float)) or swing < 0 or swing > 1):
        swing = None
    return (grid_size, ts, int(length), tempo, swing)


def get_beat_patterns(
    style: str,
    steps: int | None = None,
    grid_size: str | None = None,
    time_signature: list | None = None,
    length: int | None = None,
) -> Tuple[Tuple[List[int], ...], dict]:
    """
    Returns (drum_patterns_tuple, config_dict) for a given style.
    config_dict has gridSize, timeSignature, length, steps.
    Patterns are stored in JSON. Legacy patterns (no _meta) get defaults.
    """
    base = 16

    def load_beat_patterns():
        """Load and cache beat patterns from JSON on first use."""
        global _BEAT_PATTERNS_CACHE
        if _BEAT_PATTERNS_CACHE is None:
            ensure_beat_patterns_file()
            with open(BEAT_PATTERNS_FILE, "r") as f:
                _BEAT_PATTERNS_CACHE = json.load(f)
        return _BEAT_PATTERNS_CACHE

    patterns = load_beat_patterns()

    if style not in patterns:
        style = "default"

    raw = patterns[style]
    if not isinstance(raw, dict):
        raw = {}

    style_patterns = {k: v for k, v in raw.items() if k in DRUM_ROW_ORDER}
    if "_meta" in raw:
        gs, ts, ln, _, _ = get_pattern_config(raw)
    else:
        gs, ts, ln = DEFAULT_GRID_SIZE, DEFAULT_TIME_SIGNATURE, DEFAULT_PATTERN_LENGTH

    if grid_size is not None:
        gs = grid_size
    if time_signature is not None:
        ts = time_signature
    if length is not None:
        ln = length

    computed_steps = steps if steps is not None else compute_steps(gs, ts, ln)
    drum_sample_order = DRUM_ROW_ORDER

    def pad(pattern):
        return (list(pattern) * ((computed_steps // len(pattern)) + 1))[:computed_steps]

    def get_drum_sample_pattern(name):
        p = style_patterns.get(name)
        if p is None:
            return [0] * base
        return list(p) if hasattr(p, "__iter__") and not isinstance(p, str) else [0] * base

    config = {
        "gridSize": gs,
        "timeSignature": ts,
        "length": ln,
        "steps": computed_steps,
    }
    return tuple(pad(get_drum_sample_pattern(name)) for name in drum_sample_order), config


def filter_chords(chords, filters):
    """Filters a chord collection based on optional criteria.

    - `tags`: List of tags (OR logic, case-insensitive)
    - `name`: Partial case-insensitive match; comma-separated values use OR semantics
    - `root`: Exact case-insensitive match
    - `type`: Exact case-insensitive match ("scale" or "chord")
    """

    def matches_criteria(chord):
        """Checks if a chord matches all provided filters in priority order."""

        chord_name = chord["name"].lower()
        chord_root = chord["root"].lower()
        chord_type = chord["type"].lower()
        chord_tags = [tag.lower() for tag in chord["tags"]]

        if "roots" in filters and filters["roots"]:
            filter_roots = [r.lower() for r in filters["roots"]]
            if chord_root not in filter_roots:
                return False  # Must match at least one root

        if "name" in filters and filters["name"]:
            name_filter = filters["name"]
            if isinstance(name_filter, str):
                parts = [
                    p.strip().lower()
                    for p in name_filter.split(",")
                    if p.strip()
                ]
            elif isinstance(name_filter, (list, tuple)):
                parts = [
                    str(p).strip().lower()
                    for p in name_filter
                    if str(p).strip()
                ]
            else:
                parts = []

            if not parts:
                pass
            elif len(parts) == 1:
                if parts[0] not in chord_name:
                    return False  # substring match
            else:
                # Any substring may match chart name (OR), for multi-picker selections.
                if not any(part in chord_name for part in parts):
                    return False

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


def generate_drone_midi(
    pattern,  # Pattern of playback
    output_name="",
    note_density=2,  # Notes per beat (higher = more active)
    duration_variance=0.5,  # Variance in note lengths (0 = fixed, 1 = max randomness)
    velocity_range=(80, 120),  # MIDI note velocity range
    num_bars: int = DEFAULT_DRONE_MIDI_LENGTH_BARS,
    padded_silence_bars: int = DEFAULT_DRONE_MIDI_PADDING_BARS,
    humanization=0.02,  # Time shift variance in seconds (default: 20ms)
    swing: float = 0.0,  # Rhythmic swing amount (0 = straight, 1 = strong swing)
    shift_octave_down=None,
    shift_root_note=None,
    filters={},
    notes=None,  # Add notes parameter
    iteration=None,
    iterations=None,
):
    """Generate MIDI for ``num_bars`` of musical content plus optional trailing silence."""
    print(generate_drone_midi_header())

    try:
        num_bars = int(num_bars)
    except (TypeError, ValueError):
        raise ValueError("num_bars must be an integer") from None
    if num_bars not in DRONE_MIDI_LENGTH_BARS_ALLOWED:
        raise ValueError(
            f"num_bars must be one of {sorted(DRONE_MIDI_LENGTH_BARS_ALLOWED)}"
        )

    try:
        padded_silence_bars = int(padded_silence_bars)
    except (TypeError, ValueError):
        raise ValueError("padded_silence_bars must be an integer") from None
    if padded_silence_bars not in DRONE_MIDI_PADDING_BARS_ALLOWED:
        raise ValueError(
            "padded_silence_bars must be one of "
            f"{sorted(DRONE_MIDI_PADDING_BARS_ALLOWED)}"
        )

    if not pattern:
        pattern = random.choice(SUPPORTED_PATTERNS)

    if shift_octave_down is None:
        shift_octave_down = random.choice([True, False])

    if shift_root_note is None:
        shift_root_note = random.choice([True, False])

    if output_name:
        output_name = "_" + output_name

    if pattern not in SUPPORTED_PATTERNS:
        print(with_prompt(f"error: unsupported pattern '{pattern}'"))
        sys.exit()

    if notes:
        chord = notes
        root = notes[0][:-1]
        chord_name = "custom"
        track_name = format_name(f"custom_notes_{pattern}")
        print(with_prompt(f"writing custom notes as {YELLOW}{pattern}{RESET}"))
    else:
        # Load chords from JSON file
        with open(CHORD_SCALE_LIST, "r") as f:
            chords = json.load(f)

        if not chords:
            print(
                with_prompt(f"error: no chords or scales found in '{CHORD_SCALE_LIST}'")
            )
            return

        if filters:
            chords = filter_chords(chords, filters)

        # Randomly select a chord
        random_chord_choice = random.choice(chords)
        chord = random_chord_choice["notes"]
        root = random_chord_choice["root"]
        chord_name = random_chord_choice["name"]
        track_name = format_name(f"{root}_{chord_name}{output_name}_{pattern}")

        print(
            with_prompt(
                f"writing {YELLOW}{root} {chord_name}{RESET} as {YELLOW}{pattern}{RESET}"
            )
        )

    description = next(
        (info[1] for info in SUPPORTED_PATTERNS_INFO if info[0] == pattern), None
    )
    print(with_prompt(f"({description})"))

    if shift_root_note and not notes:
        print(with_prompt("shifting root one octave down"))

    if shift_octave_down and not notes:
        print(with_prompt("shifting all notes one octave down"))

    # Define tempo and timing
    bpm = 120
    beats_per_bar = 4
    seconds_per_beat = 60.0 / bpm
    bar_length = beats_per_bar * seconds_per_beat
    total_duration = num_bars * bar_length  # Musical content length only
    file_end_duration = drone_midi_render_duration_sec(num_bars, padded_silence_bars, tempo_bpm=bpm)
    if padded_silence_bars:
        padding_duration = padded_silence_bars * bar_length
        print(
            with_prompt(
                f"padded silence at end {YELLOW}{padded_silence_bars}{RESET} bars "
                f"({padding_duration:.2f}s)"
            )
        )

    # Create a PrettyMIDI object (explicit tempo keeps DawDreamer/file timing aligned).
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    instrument = pretty_midi.Instrument(program=0, name=track_name)

    # Process chord notes
    midi_notes = []
    for note_str in chord:
        note_name, octave = note_str[:-1], int(note_str[-1])  # Extract note and octave
        midi_number = pretty_midi.note_name_to_number(note_name + str(octave))
        midi_notes.append(midi_number)

    # Drop the root note one octave down
    if shift_root_note and not notes:
        midi_notes[0] -= 12

    if shift_octave_down and not notes:
        midi_notes = [note - 12 for note in midi_notes]

    # 🎵 **MIDI Note Generation Based on Pattern**
    time = 0.0

    if pattern == "chaos":
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

    elif pattern == "chaos_expand_up":
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

    elif pattern == "split_chord":
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

    elif pattern == "quantized_straight_quarter":
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

    elif pattern == "quantized_straight_eighth":
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

    elif pattern == "quantized_up_down_quarter":
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

    elif pattern == "quantized_up_down_eighth":
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

    elif pattern == "lead":
        # **Lead Melody:** Semi-random movement within the scale for the full phrase.
        note_durations = [
            seconds_per_beat * d for d in [0.25, 0.5, 1, 2]
        ]  # 16th, 8th, quarter, half

        current_note = random.choice(midi_notes)

        while time < total_duration:
            velocity = random.randint(*velocity_range)
            start_time = max(0.0, time)
            duration = random.choice(note_durations)
            end_time = min(start_time + duration, total_duration)

            instrument.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=current_note,
                    start=start_time,
                    end=end_time,
                )
            )

            time += duration

            if random.random() < 0.6:
                idx = midi_notes.index(current_note)
                if random.random() < 0.5 and idx < len(midi_notes) - 1:
                    current_note = midi_notes[idx + 1]
                elif idx > 0:
                    current_note = midi_notes[idx - 1]
            else:
                current_note = random.choice(midi_notes)

    elif pattern == "lead_flat":
        # **Lead Melody (Flat):** Constant eighth notes for the full phrase.
        note_duration = seconds_per_beat * 0.5
        current_note = random.choice(midi_notes)

        while time < total_duration:
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

            time += note_duration

            if random.random() < 0.7:
                idx = midi_notes.index(current_note)
                if random.random() < 0.5 and idx < len(midi_notes) - 1:
                    current_note = midi_notes[idx + 1]
                elif idx > 0:
                    current_note = midi_notes[idx - 1]
            else:
                current_note = random.choice(midi_notes)

    elif pattern == "lead_straight_sixteenth":
        # **Lead (Straight, Sixteenth Notes):** Loop low→high for the full phrase.
        note_duration = seconds_per_beat * 0.25
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

    elif pattern == "lead_straight_eighth":
        # **Lead (Straight, Eighth Notes):** Loop low→high for the full phrase.
        note_duration = seconds_per_beat * 0.5
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

    elif pattern == "reverse_arpeggio_quarter":
        notes_desc = list(reversed(sorted(midi_notes)))
        note_duration = seconds_per_beat
        while time < total_duration:
            for note in notes_desc:
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

    elif pattern == "strum_burst_quarter":
        ordered = sorted(midi_notes)
        while time < total_duration:
            if not ordered:
                break
            step = seconds_per_beat / len(ordered)
            for note in ordered:
                if time >= total_duration:
                    break
                velocity = random.randint(*velocity_range)
                start_time = max(0.0, time)
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=note,
                        start=start_time,
                        end=min(start_time + step * 0.92, total_duration),
                    )
                )
                time += step

    elif pattern == "pedal_high_arp_eighth":
        ordered = sorted(midi_notes)
        note_duration = seconds_per_beat * 0.5
        if len(ordered) == 1:
            pitch_note = ordered[0]
            while time < total_duration:
                velocity = random.randint(*velocity_range)
                start_time = max(0.0, time)
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch_note,
                        start=start_time,
                        end=min(start_time + note_duration, total_duration),
                    )
                )
                time += note_duration
        else:
            drone_velocity = velocity_range[0] + (
                velocity_range[1] - velocity_range[0]
            ) // 4
            high_pitch = ordered[-1]
            pool = ordered[:-1]
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=min(velocity_range[1], drone_velocity),
                    pitch=high_pitch,
                    start=0.0,
                    end=total_duration,
                )
            )
            ai = 0
            while time < total_duration:
                velocity = random.randint(*velocity_range)
                pitch_note = pool[ai % len(pool)]
                ai += 1
                start_time = max(0.0, time)
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch_note,
                        start=start_time,
                        end=min(start_time + note_duration, total_duration),
                    )
                )
                time += note_duration

    elif pattern == "bounce_octaves_extremes":
        low_pitch = min(midi_notes)
        high_pitch = max(midi_notes)
        note_duration = seconds_per_beat * 0.5
        use_low = True
        while time < total_duration:
            velocity = random.randint(*velocity_range)
            pitch = low_pitch if use_low else high_pitch
            use_low = not use_low
            start_time = max(0.0, time)
            short = min(note_duration * 0.55, seconds_per_beat * 0.4)
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start_time,
                    end=min(start_time + short, total_duration),
                )
            )
            time += note_duration

    elif pattern == "triplet_straight":
        ordered = sorted(midi_notes)
        slot = seconds_per_beat / 3.0  # eighth-note triplets inside one beat
        if not ordered:
            ordered = midi_notes[:]
        i = 0
        while time < total_duration:
            velocity = random.randint(*velocity_range)
            pitch = ordered[i % len(ordered)]
            i += 1
            start_time = max(0.0, time)
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start_time,
                    end=min(start_time + slot * 0.95, total_duration),
                )
            )
            time += slot

    elif pattern == "syncopated_offbeat_eighth":
        ordered = sorted(midi_notes)
        if not ordered:
            ordered = midi_notes[:]
        slot = seconds_per_beat * 0.5
        bar_t = 0.0
        ni = 0
        while bar_t < total_duration - 1e-9:
            for beat_ix in range(beats_per_bar):
                eighth_start = beat_ix * seconds_per_beat + slot  # '&' subdivision
                t = bar_t + eighth_start
                if t >= total_duration:
                    break
                velocity = random.randint(*velocity_range)
                pitch = ordered[ni % len(ordered)]
                ni += 1
                dur = slot * 0.85
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=t,
                        end=min(t + dur, total_duration),
                    )
                )
            bar_t += bar_length

    elif pattern == "two_hand_alternate":
        ordered = sorted(midi_notes)
        if not ordered:
            ordered = midi_notes[:]
        mid = max(1, len(ordered) // 2)
        lower_voice = ordered[:mid]
        upper_voice = ordered[mid:] or ordered[-1:]
        note_duration = seconds_per_beat * 0.5
        flip = False
        while time < total_duration:
            pool = upper_voice if flip else lower_voice
            flip = not flip
            velocity = random.randint(*velocity_range)
            pitch = random.choice(pool)
            start_time = max(0.0, time)
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start_time,
                    end=min(start_time + note_duration, total_duration),
                )
            )
            time += note_duration

    elif pattern == "rhythmic_long_short_short":
        # Within each beat: half-beat tone, then two quarter-beat tones
        motif = (
            seconds_per_beat * 0.5,
            seconds_per_beat * 0.25,
            seconds_per_beat * 0.25,
        )
        idx = random.randrange(len(midi_notes))
        ordered = sorted(midi_notes)

        while time < total_duration:
            for part_dur in motif:
                if time >= total_duration:
                    break
                velocity = random.randint(*velocity_range)
                if random.random() < 0.65:
                    step = random.choice([-1, 1])
                    idx = (idx + step) % len(ordered)
                    current = ordered[idx]
                else:
                    current = random.choice(ordered)
                start_time = max(0.0, time)
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=current,
                        start=start_time,
                        end=min(start_time + part_dur * 0.94, total_duration),
                    )
                )
                time += part_dur

    elif pattern == "cluster_hit_bar":
        stab = min(seconds_per_beat * 0.15, 0.08)
        for bar_index in range(num_bars):
            start_time = bar_index * bar_length
            velocity = random.randint(*velocity_range)
            for pitch in midi_notes:
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time,
                        end=min(start_time + stab, total_duration),
                    )
                )

    elif pattern == "double_layer_octave_ping":
        ordered = sorted(midi_notes)
        if not ordered:
            ordered = midi_notes[:]
        note_duration = seconds_per_beat * 0.5
        i = 0

        def _clamp_velocity(v):
            lo, hi = velocity_range
            return max(lo, min(hi, int(v)))

        while time < total_duration:
            v_lo = _clamp_velocity(random.randint(*velocity_range) + random.randint(-4, 4))
            v_hi = _clamp_velocity(random.randint(*velocity_range) - random.randint(4, 12))
            base_pitch = ordered[i % len(ordered)]
            i += 1
            ob_pitch = min(127, base_pitch + 12)
            start_time = max(0.0, time)
            end_t = min(start_time + note_duration, total_duration)
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=v_lo,
                    pitch=base_pitch,
                    start=start_time,
                    end=end_t,
                )
            )
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=v_hi,
                    pitch=ob_pitch,
                    start=start_time,
                    end=end_t,
                )
            )
            time += note_duration

    elif pattern == "pedal_low_arp_quarter":
        ordered = sorted(midi_notes)
        note_duration = seconds_per_beat
        if len(ordered) == 1:
            pn = ordered[0]
            while time < total_duration:
                velocity = random.randint(*velocity_range)
                st = max(0.0, time)
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=pn,
                        start=st,
                        end=min(st + note_duration, total_duration),
                    )
                )
                time += note_duration
        else:
            drone_vel = velocity_range[0] + (
                velocity_range[1] - velocity_range[0]
            ) // 4
            low_pitch = ordered[0]
            pool = ordered[1:]
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=min(velocity_range[1], drone_vel),
                    pitch=low_pitch,
                    start=0.0,
                    end=total_duration,
                )
            )
            ai = 0
            while time < total_duration:
                velocity = random.randint(*velocity_range)
                pitch_note = pool[ai % len(pool)]
                ai += 1
                start_time = max(0.0, time)
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch_note,
                        start=start_time,
                        end=min(start_time + note_duration, total_duration),
                    )
                )
                time += note_duration

    elif pattern == "neighbor_pair_eighth":
        ordered = sorted(midi_notes)
        if len(ordered) < 2:
            ordered = midi_notes[:]
        slot = seconds_per_beat * 0.5
        if len(ordered) < 2:
            pitch = ordered[0]
            while time < total_duration:
                velocity = random.randint(*velocity_range)
                start_time = max(0.0, time)
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time,
                        end=min(start_time + slot, total_duration),
                    )
                )
                time += slot
        else:
            pairs = [(ordered[i], ordered[(i + 1) % len(ordered)]) for i in range(len(ordered))]
            pi = 0
            while time < total_duration:
                a_pitch, b_pitch = pairs[pi % len(pairs)]
                pi += 1
                for pitch in (a_pitch, b_pitch):
                    if time >= total_duration:
                        break
                    velocity = random.randint(*velocity_range)
                    start_time = max(0.0, time)
                    gate = slot * 0.88
                    instrument.notes.append(
                        pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=start_time,
                            end=min(start_time + gate, total_duration),
                        )
                    )
                    time += slot

    elif pattern == "strum_burst_quarter_down":
        ordered = list(reversed(sorted(midi_notes)))
        while time < total_duration:
            if not ordered:
                break
            step = seconds_per_beat / len(ordered)
            for note in ordered:
                if time >= total_duration:
                    break
                velocity = random.randint(*velocity_range)
                start_time = max(0.0, time)
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=note,
                        start=start_time,
                        end=min(start_time + step * 0.92, total_duration),
                    )
                )
                time += step

    elif pattern == "skip_step_quarter_arpeggio":
        ordered = sorted(midi_notes)
        if not ordered:
            ordered = midi_notes[:]
        note_duration = seconds_per_beat
        idx = 0
        n = len(ordered)
        step = 2 if n > 2 else 1
        while time < total_duration:
            velocity = random.randint(*velocity_range)
            pitch = ordered[idx % n]
            idx = (idx + step) % n
            start_time = max(0.0, time)
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start_time,
                    end=min(start_time + note_duration, total_duration),
                )
            )
            time += note_duration

    elif pattern == "random_dyad_stabs":
        notes_list = midi_notes[:] if midi_notes else [60]
        stab = max(0.04, seconds_per_beat * 0.08)
        step = seconds_per_beat * 0.65
        while time < total_duration:
            if len(notes_list) >= 2:
                a, b = random.sample(sorted(set(notes_list)), 2)
                pair = [a, b]
            else:
                pair = [notes_list[0]]
            lo, hi = velocity_range
            base_vel = random.randint(lo, hi)
            start_time = max(0.0, time)
            for ix, pitch in enumerate(pair):
                v = base_vel - (ix * 8)
                if v < lo:
                    v = lo
                if v > hi:
                    v = hi
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=v,
                        pitch=pitch,
                        start=start_time,
                        end=min(start_time + stab, total_duration),
                    )
                )
            time += step

    elif pattern == "dotted_rhythm_triplet_cells":
        ordered = sorted(midi_notes)
        if not ordered:
            ordered = midi_notes[:]
        long_n = seconds_per_beat * 0.5625  # dotted-eighth-ish (9/16 of beat)
        short_n = seconds_per_beat * 0.1875  # remaining sixteenth-ish
        i = 0
        while time < total_duration:
            for dur in (long_n, short_n):
                if time >= total_duration:
                    break
                pitch = ordered[i % len(ordered)]
                i += 1
                velocity = random.randint(*velocity_range)
                start_time = max(0.0, time)
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time,
                        end=min(start_time + dur * 0.92, total_duration),
                    )
                )
                time += dur

    elif pattern == "accent_root_strong_weak_inner":
        ordered = sorted(midi_notes)
        if not ordered:
            ordered = midi_notes[:]
        root_pitch = ordered[0]
        inner = ordered[1:] or ordered[:]
        bar_t = 0.0
        ni = 0
        while bar_t < total_duration - 1e-9:
            for beat_ix in range(beats_per_bar):
                t_abs = bar_t + beat_ix * seconds_per_beat
                if t_abs >= total_duration:
                    break
                strong = beat_ix in (0, 2)
                if strong:
                    vel = random.randint(max(velocity_range[0], velocity_range[1] - 20), velocity_range[1])
                    pitch = root_pitch
                else:
                    vel = random.randint(velocity_range[0], max(velocity_range[0], velocity_range[1] - 12))
                    pitch = inner[ni % len(inner)]
                    ni += 1
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=vel,
                        pitch=pitch,
                        start=t_abs,
                        end=min(t_abs + seconds_per_beat * 0.88, total_duration),
                    )
                )
            bar_t += bar_length

    elif pattern == "bar_rotating_arpeggio_quarter":
        ordered = sorted(midi_notes)
        if not ordered:
            ordered = midi_notes[:]
        n = len(ordered)
        note_duration = seconds_per_beat
        for bar_index in range(num_bars):
            start_bar = bar_index * bar_length
            k = bar_index % n
            rotated = ordered[k:] + ordered[:k]
            local_t = 0.0
            rp = 0
            while local_t + start_bar < total_duration:
                velocity = random.randint(*velocity_range)
                pitch = rotated[rp % n]
                rp += 1
                abs_t = max(0.0, start_bar + local_t)
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=abs_t,
                        end=min(abs_t + note_duration, total_duration),
                    )
                )
                local_t += note_duration

    elif pattern == "stagger_voice_pad_hit":
        voices = sorted(midi_notes)
        if not voices:
            voices = midi_notes[:]
        tail_trim = seconds_per_beat * 0.06
        for bar_index in range(num_bars):
            bar_start = bar_index * bar_length
            vn = len(voices)
            spread = seconds_per_beat / max(12, vn * 3 + 4)
            for j, pitch in enumerate(voices):
                start_time = bar_start + min(j * spread, bar_length - spread - 1e-6)
                if start_time >= total_duration:
                    break
                velocity = random.randint(max(velocity_range[0], velocity_range[1] - 38), velocity_range[1])
                dur = bar_length - (start_time - bar_start) - tail_trim
                if dur <= spread:
                    dur = spread * 1.25
                end_t = min(start_time + max(dur, seconds_per_beat * 0.15), bar_start + bar_length - 1e-4)
                end_t = min(end_t, total_duration)
                if end_t > start_time:
                    instrument.notes.append(
                        pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=max(0.0, start_time),
                            end=min(end_t, total_duration),
                        )
                    )

    elif pattern == "shuffled_chord_voice_quarters":
        ordered = sorted(midi_notes)
        if not ordered:
            ordered = midi_notes[:]
        note_duration = seconds_per_beat
        for bar_index in range(num_bars):
            start_bar = bar_index * bar_length
            seq = ordered[:]
            random.shuffle(seq)
            local_t = 0.0
            si = 0
            while local_t + start_bar < total_duration:
                velocity = random.randint(*velocity_range)
                pitch = seq[si % len(seq)]
                si += 1
                abs_t = max(0.0, start_bar + local_t)
                instrument.notes.append(
                    pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=abs_t,
                        end=min(abs_t + note_duration, total_duration),
                    )
                )
                local_t += note_duration

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

    # Apply rhythmic swing after note generation (if requested)
    def apply_swing_to_instrument(
        inst: pretty_midi.Instrument, bpm_val: int, swing_amt: float
    ):
        """
        Apply simple 8th-note swing: delay off-beat 8ths towards a triplet feel.
        swing_amt in [0, 1], where 0 = straight, 1 ≈ triplet swing.
        """
        swing_clamped = max(0.0, min(1.0, float(swing_amt)))
        if swing_clamped <= 0.0:
            return

        seconds_per_beat_local = 60.0 / bpm_val
        subdivision_beats = 0.5  # eighth-note grid

        straight_off_beat = subdivision_beats  # 0.5 beat
        swung_off_beat = 2.0 / 3.0  # ≈0.666 beat (triplet feel)
        shift_beats = (swung_off_beat - straight_off_beat) * swing_clamped
        shift_seconds = shift_beats * seconds_per_beat_local

        for note in inst.notes:
            beat_pos = note.start / seconds_per_beat_local
            idx = int(beat_pos / subdivision_beats)
            # Odd indices correspond to off-beat 8ths (positions between quarter notes)
            if idx % 2 == 1:
                note.start += shift_seconds
                note.end += shift_seconds

    if swing:
        apply_swing_to_instrument(instrument, bpm, swing)

    # Add instrument **without a name** to MIDI object
    midi.instruments.append(instrument)

    # Write to a MIDI file
    os.makedirs(MIDI_DIR, exist_ok=True)

    output_path = f"{MIDI_DIR}/{track_name}_{generate_id()}.mid"
    midi.write(output_path)

    print(f"{YELLOW}│{RESET}")
    print_midi(output_path, f"{YELLOW}│  {RESET}")

    print(f"{YELLOW}│{RESET}")

    return (output_path, f"{root} {chord_name}", file_end_duration)


def main():
    # Preserve simple legacy behavior if this module is run directly.
    args = sys.argv[1:]
    if not args:
        generate_drone_midi(pattern=None)
        return

    generate_drone_midi(pattern=args[0])


if __name__ == "__main__":
    main()
