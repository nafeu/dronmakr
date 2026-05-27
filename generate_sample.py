import sys
import os
import json
import random
import subprocess
import math
import pedalboard
from pathlib import Path
from settings import get_setting, parse_escaped_csv
from pydub import AudioSegment
from pedalboard import (
    Compressor,
    Gain,
    HighpassFilter,
    LowpassFilter,
    Limiter,
    Pedalboard,
)
from pedalboard.io import AudioFile
from mido import MidiFile, Message

from generate_midi import get_beat_patterns
from processing_actions import apply_post_processing_actions
from utils import (
    CYAN,
    RESET,
    extract_plugin,
    generate_drone_sample_header,
    GREEN,
    with_generate_beat_prompt,
    with_generate_drone_sample_prompt as with_prompt,
    PRESETS_PATH,
    resolve_presets_index_path,
)

# Convert MIDI file to MIDI messages
def midi_to_messages(midi_file_path):
    """Reads a MIDI file and converts it to MIDI messages for processing."""
    if not midi_file_path.lower().endswith(".mid"):
        raise ValueError(
            f"Invalid file type: {midi_file_path}. The script requires a MIDI (.mid) file."
        )

    mid = MidiFile(midi_file_path)
    midi_messages = []
    current_time = 0

    for track in mid.tracks:
        for msg in track:
            if msg.type in ["note_on", "note_off"]:  # Only process note messages
                current_time += (
                    msg.time / mid.ticks_per_beat
                )  # Convert ticks to seconds
                midi_messages.append(
                    Message(
                        msg.type,
                        note=msg.note,
                        velocity=msg.velocity,
                        time=current_time,
                    )
                )

    return midi_messages, mid.length  # Return the list of messages and the duration


SAMPLE_RATE = 44100


def effect_slot_entries(effect_preset: dict) -> list[dict]:
    """Expand a preset row into ordered effect steps for loading plug-ins."""
    if effect_preset.get("type") == "effect":
        return [
            {
                "plugin_path": effect_preset["plugin_path"],
                "plugin_name": effect_preset.get("plugin_name") or "",
                "preset_path": effect_preset["preset_path"],
                "name": effect_preset.get("name", "effect"),
            }
        ]
    return list(effect_preset.get("effects") or [])


def generate_drone_sample(
    input_path="input.mid",
    output_path="generated_sample.wav",
    presets_path=PRESETS_PATH,
    instrument=None,
    effect=None,
):
    presets_path = presets_path or resolve_presets_index_path() or PRESETS_PATH

    # Desktop tray mode: Flask runs on a worker thread; many plug-ins require the process main thread.
    from pedalboard_isolated_runner import delegate_generate_drone_sample_if_needed

    delegated = delegate_generate_drone_sample_if_needed(
        os.path.abspath(input_path),
        os.path.abspath(output_path),
        os.path.abspath(presets_path),
        instrument,
        effect,
    )
    if delegated is not None:
        return delegated

    print(generate_drone_sample_header())

    loaded_effects = []

    from pedalboard_isolated_runner import ensure_pedalboard_midi_utils

    ensure_pedalboard_midi_utils()

    # Load presets from JSON
    with open(presets_path, "r") as f:
        presets = json.load(f)

    if not isinstance(presets, list):
        raise ValueError(
            f"{presets_path} must contain a JSON array of preset objects, "
            f"not {type(presets).__name__}"
        )

    instruments = [p for p in presets if isinstance(p, dict) and p.get("type") == "instrument"]
    fx_presets = [
        p for p in presets if isinstance(p, dict) and p.get("type") in ("effect", "effect_chain")
    ]

    if not instruments:
        raise ValueError(
            f"No instrument presets in {presets_path}. Save at least one synth/instrument preset "
            "from Patchcraftr (saved as type “instrument”)."
        )

    if not instrument:
        instrument_preset = random.choice(instruments)
    else:
        for preset in instruments:
            if preset.get("name") == instrument:
                instrument_preset = preset
                break
        else:
            raise ValueError(f"No instrument found with the name '{instrument}'")

    if effect:
        effect_preset = next((p for p in fx_presets if p.get("name") == effect), None)
        if effect_preset is None:
            avail = sorted({p.get("name", "") for p in fx_presets if isinstance(p.get("name"), str)})
            hint = ""
            if not fx_presets:
                hint = " There are no saved effect or effect_chain presets yet — save FX from Patchcraftr."
            elif avail:
                hint = f" Available effect names (sample): {', '.join(avail[:12])}"
                if len(avail) > 12:
                    hint += ", …"
            raise ValueError(
                f"No saved effect or effect_chain preset named '{effect}' in {presets_path}.{hint}"
            )
    elif fx_presets:
        effect_preset = random.choice(fx_presets)
    else:
        # Instrument-only rigs (common right after authoring a first synth preset).
        effect_preset = None

    slots: list[dict] = []
    if effect_preset is not None:
        slots = effect_slot_entries(effect_preset)
        if not slots:
            raise ValueError(f"Effect preset '{effect_preset.get('name', '')}' has no processors to load.")

    if effect_preset is not None:
        print(
            with_prompt(
                f"selected {GREEN}{instrument_preset['name']}{RESET} sound processed with "
                f"{GREEN}{effect_preset['name']}{RESET}"
            )
        )
    else:
        print(
            with_prompt(
                f"selected {GREEN}{instrument_preset['name']}{RESET} only "
                f"(no FX presets saved — skipping effect chain)"
            )
        )

    # Load the instrument plugin with `plugin_name` if available
    if instrument_preset.get("plugin_name"):
        print(
            with_prompt(
                f"loading instrument {GREEN}{extract_plugin(instrument_preset['plugin_path'])}{RESET} as {GREEN}{instrument_preset['plugin_name']}{RESET}"
            )
        )
        instrument_plugin = pedalboard.load_plugin(
            instrument_preset["plugin_path"],
            plugin_name=instrument_preset["plugin_name"],
        )
    else:
        print(
            with_prompt(
                f"loading instrument {GREEN}{extract_plugin(instrument_preset['plugin_path'])}{RESET}"
            )
        )
        instrument_plugin = pedalboard.load_plugin(instrument_preset["plugin_path"])

    with open(instrument_preset["preset_path"], "rb") as f:
        ins_blob = f.read()
        print(with_prompt(f"using instrument preset {GREEN}{instrument_preset['name']}{RESET}"))
        if hasattr(instrument_plugin, "preset_data"):
            instrument_plugin.preset_data = ins_blob
        else:
            instrument_plugin.raw_state = ins_blob

    if slots:
        print(with_prompt(f"loading effect {GREEN}{effect_preset['name']}{RESET}"))

        for eff in slots:
            if eff.get("plugin_name"):
                print(
                    with_prompt(
                        f"inserting {GREEN}{extract_plugin(eff['plugin_path'])}{RESET} as {GREEN}{eff['plugin_name']}{RESET}"
                    )
                )
                effect_plugin = pedalboard.load_plugin(
                    eff["plugin_path"], plugin_name=eff["plugin_name"]
                )
            else:
                print(
                    with_prompt(
                        f"inserting {GREEN}{extract_plugin(eff['plugin_path'])}{RESET}"
                    )
                )
                effect_plugin = pedalboard.load_plugin(eff["plugin_path"])

            with open(eff["preset_path"], "rb") as f:
                eblob = f.read()
                print(with_prompt(f"using effect step {GREEN}{eff['name']}{RESET}"))
                if hasattr(effect_plugin, "preset_data"):
                    effect_plugin.preset_data = eblob
                else:
                    effect_plugin.raw_state = eblob

            loaded_effects.append(effect_plugin)
    else:
        print(with_prompt("Skipping effect plug-ins — none listed in presets.json."))

    # Load MIDI file and get messages
    midi_messages, audio_length_s = midi_to_messages(input_path)

    # Process MIDI through the instrument plugin
    # reset=False: required when this runs on Flask worker threads (desktop tray mode);
    # many VSTs error with "must be reloaded on the main thread" if reset defaults to True.
    pre_fx_signal = instrument_plugin(
        midi_messages,
        duration=audio_length_s,
        sample_rate=SAMPLE_RATE,
        num_channels=2,  # Stereo
        buffer_size=8192,
        reset=False,
    )

    # Create a gain reduction effect (-6dB = 0.5 multiplier)
    gain_reduction = Pedalboard([Gain(gain_db=-6)])  # -6dB reduction

    pre_fx_signal = gain_reduction(pre_fx_signal, SAMPLE_RATE, reset=False)

    print(with_prompt(f"sending midi and rendering audio ({audio_length_s:.2f}s)"))

    # Apply the selected effect plugin
    fx_chain = Pedalboard(loaded_effects)
    post_fx_signal = fx_chain(pre_fx_signal, SAMPLE_RATE, reset=False)

    output_path = output_path.replace("#", "sharp")

    # Export processed audio
    with AudioFile(output_path, "w", SAMPLE_RATE, 2) as f:
        f.write(post_fx_signal)

    print(f"{GREEN}│{RESET}")

    return output_path


def apply_effect(input_path, effect_chain, presets_path=PRESETS_PATH):
    """Applies an effect chain from presets to a WAV file and overwrites it after backing up."""

    presets_path = presets_path or PRESETS_PATH
    from pedalboard_isolated_runner import delegate_apply_effect_if_needed

    if delegate_apply_effect_if_needed(
        os.path.abspath(input_path),
        effect_chain,
        os.path.abspath(presets_path),
    ):
        return

    # Load presets
    with open(presets_path, "r") as f:
        presets = json.load(f)

    fx_pool = [p for p in presets if p["type"] in ("effect", "effect_chain")]

    if effect_chain == "":
        effect_preset = random.choice(fx_pool)
    else:
        effect_preset = next(
            (
                p
                for p in presets
                if p["type"] in ("effect", "effect_chain") and p["name"] == effect_chain
            ),
            None,
        )

    if not effect_preset:
        raise ValueError(f"Effect preset '{effect_chain}' not found in presets.")

    slot_list = effect_slot_entries(effect_preset)
    if not slot_list:
        raise ValueError(f"Effect preset '{effect_preset['name']}' has no processors to load.")

    output_path = input_path.replace(".wav", f"_{effect_preset['name']}.wav")

    print(f"Applying effect: {effect_preset['name']}")

    loaded_effects = []

    for eff in slot_list:
        if eff.get("plugin_name"):
            print(
                f"inserting {extract_plugin(eff['plugin_path'])} as {eff['plugin_name']}"
            )
            effect_plugin = pedalboard.load_plugin(eff["plugin_path"], plugin_name=eff["plugin_name"])
        else:
            print(f"inserting {extract_plugin(eff['plugin_path'])}")
            effect_plugin = pedalboard.load_plugin(eff["plugin_path"])

        with open(eff["preset_path"], "rb") as f:
            eblob = f.read()
            print(f"using preset {eff['name']}")
            if hasattr(effect_plugin, "preset_data"):
                effect_plugin.preset_data = eblob
            else:
                effect_plugin.raw_state = eblob

        loaded_effects.append(effect_plugin)

    # Load audio file
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate

    # Apply effects
    normalization_chain = [
        HighpassFilter(cutoff_frequency_hz=40),  # Remove sub-bass rumble
        LowpassFilter(cutoff_frequency_hz=18000),  # Remove harsh highs if needed
        Compressor(
            threshold_db=-24, ratio=1.5, attack_ms=30, release_ms=200
        ),  # Gentle compression for control
        Limiter(
            threshold_db=-4, release_ms=250
        ),  # Ensure headroom without crushing dynamics
    ]
    fx_chain = Pedalboard(loaded_effects + normalization_chain)
    processed_audio = fx_chain(audio, sample_rate, reset=False)

    with AudioFile(
        output_path, "w", samplerate=sample_rate, num_channels=processed_audio.shape[0]
    ) as f:
        f.write(processed_audio)

    print(f"Effect '{effect_chain}' applied and saved: {input_path}")


# --- Beat sample helpers (drum loops) ---


def get_random_sample(folder: Path) -> AudioSegment:
    """Load a random WAV from the given folder."""
    files = [f for f in folder.iterdir() if f.suffix == ".wav"]
    if not files:
        raise ValueError(f"No WAV files found in {folder}")
    return AudioSegment.from_wav(str(random.choice(files)))


def load_sample_from_path(path: str | Path) -> AudioSegment:
    """Load a single WAV file from the given path."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise ValueError(f"Sample file not found: {path}")
    return AudioSegment.from_wav(str(p))


def adjust_velocity(segment: AudioSegment, db_change: int) -> AudioSegment:
    """Adjust segment level by db_change decibels."""
    return segment + db_change


def trim_decay(segment: AudioSegment, ms: int) -> AudioSegment:
    """Trim segment to ms and apply fade-out."""
    return segment[:ms].fade_out(ms)


def open_file_with_default_player(file_path: str) -> None:
    """Open a file with the system default application."""
    try:
        if sys.platform.startswith("darwin"):
            subprocess.run(["open", file_path])
        elif sys.platform.startswith("win"):
            os.startfile(file_path)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", file_path])
        else:
            raise RuntimeError("Unsupported OS for auto-playing files.")
    except Exception as e:
        print(f"Failed to open file: {e}")


# Export sample rate for drum loops: 44.1 kHz is standard for DAWs (Ableton, etc.).
BEAT_EXPORT_SAMPLE_RATE = 44100


def generate_beat_sample(
    bpm: int = 120,
    bars: int | None = None,
    output: str = "output.wav",
    humanize: bool = True,
    style: str = "breakbeat",
    swing: float = 0.0,
    play: bool = False,
    pattern_config: dict | None = None,
    kit_paths: dict[str, str] | None = None,
    pattern_data: dict | None = None,
    loops: int = 1,
) -> str:
    """
    Generate a drum loop and export to WAV.
    - When kit_paths is provided: load samples from those paths (row -> path).
    - When pattern_data is provided: use pattern arrays + _meta instead of beat-patterns.json.
    - Otherwise: use settings-configured sample folders and get_beat_patterns.
    """
    if kit_paths:
        print(with_generate_beat_prompt("loading samples from kit"))
    else:
        print(with_generate_beat_prompt("loading samples from settings"))

    cfg = pattern_config or {}
    if pattern_data and isinstance(pattern_data.get("_meta"), dict):
        meta = pattern_data["_meta"]
        cfg = {**cfg, "gridSize": meta.get("gridSize"), "timeSignature": meta.get("timeSignature"), "length": meta.get("length")}
    grid_size = cfg.get("gridSize") or "1/16"
    time_sig = cfg.get("timeSignature") or [4, 4]
    length = int(cfg.get("length") or 1)
    if bars is not None:
        length = bars

    steps_per_beat = 4 if grid_size == "1/16" else 6
    beats_per_bar = time_sig[0] if time_sig else 4
    steps = steps_per_beat * beats_per_bar * length

    step_duration_ms = (60.0 / bpm) * 1000.0 / steps_per_beat
    swing_clamped = max(0.0, min(1.0, float(swing)))
    beat_ms = (60.0 / bpm) * 1000.0
    swing_triplet_offset_ms = beat_ms / 6.0

    def swing_offset_for_step(step_index: int) -> float:
        if swing_clamped <= 0.0:
            return 0.0
        pos_in_beat = step_index % steps_per_beat
        off_beat_step = steps_per_beat // 2
        if pos_in_beat == off_beat_step:
            return swing_clamped * swing_triplet_offset_ms
        return 0.0

    # Load samples: from kit_paths or from settings
    def _pad(arr, n):
        return (list(arr) * ((n // len(arr)) + 1))[:n] if arr else [0] * n

    if kit_paths:
        def _load(row: str, fallback_row: str | None = None) -> AudioSegment:
            path = kit_paths.get(row) or (kit_paths.get(fallback_row) if fallback_row else None)
            if not path:
                raise ValueError(f"No sample path for row {row}")
            return load_sample_from_path(path)
        kick = _load("kick")
        snare = _load("snar")
        ghost_snare = _load("ghos", "snar")
        hihat = _load("hhat")
        hihat_alt = _load("halt", "hhat")
        perc_a = _load("prca")
        perc_b = _load("prcb")
        perc_c = _load("prcc")
        clap = _load("clap")
        tom = _load("tomm")
        shaker = _load("shkr")
        cymbal = _load("cymb")
    else:
        drum_path = lambda key: random.choice(parse_escaped_csv(get_setting(key, "")) or [""])
        kicks = Path(drum_path("DRUM_KICK_PATHS") or ".")
        hihats = Path(drum_path("DRUM_HIHAT_PATHS") or ".")
        percs = Path(drum_path("DRUM_PERC_PATHS") or ".")
        toms = Path(drum_path("DRUM_TOM_PATHS") or ".")
        snares = Path(drum_path("DRUM_SNARE_PATHS") or ".")
        shakers = Path(drum_path("DRUM_SHAKER_PATHS") or ".")
        claps = Path(drum_path("DRUM_CLAP_PATHS") or ".")
        cymbals = Path(drum_path("DRUM_CYMBAL_PATHS") or ".")
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

    # Pattern arrays: from pattern_data or get_beat_patterns
    if pattern_data and isinstance(pattern_data, dict):
        def get_row_pattern(name: str):
            p = pattern_data.get(name)
            return _pad(p, steps) if p is not None and hasattr(p, "__iter__") and not isinstance(p, str) else [0] * steps
        kick_pattern = get_row_pattern("kick")
        snare_pattern = get_row_pattern("snar")
        ghost_snare_pattern = get_row_pattern("ghos")
        clap_pattern = get_row_pattern("clap")
        hihat_pattern = get_row_pattern("hhat")
        hihat_alt_pattern = get_row_pattern("halt")
        shaker_pattern = get_row_pattern("shkr")
        perc_a_pattern = get_row_pattern("prca")
        perc_b_pattern = get_row_pattern("prcb")
        perc_c_pattern = get_row_pattern("prcc")
        tom_pattern = get_row_pattern("tomm")
        cymbal_pattern = get_row_pattern("cymb")
    else:
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
        ) = get_beat_patterns(style, steps=steps, grid_size=grid_size, time_signature=time_sig, length=length)[0]

    meta = pattern_data.get("_meta") if pattern_data and isinstance(pattern_data.get("_meta"), dict) else {}

    def _sanitize_velocity_randomization_map(raw: dict | None) -> dict[str, tuple[float, float]]:
        if not isinstance(raw, dict):
            return {}
        sanitized: dict[str, tuple[float, float]] = {}
        for row_key, value in raw.items():
            if row_key not in ROW_KEYS:
                continue
            if not isinstance(value, list) or len(value) != 2:
                continue
            try:
                min_v = float(value[0])
                max_v = float(value[1])
            except (TypeError, ValueError):
                continue
            if not (0.0 <= min_v <= 1.0 and 0.0 <= max_v <= 1.0):
                continue
            if min_v >= max_v:
                continue
            sanitized[row_key] = (min_v, max_v)
        return sanitized

    def _sanitize_timing_randomization_map(raw: dict | None) -> dict[str, float]:
        if not isinstance(raw, dict):
            return {}
        sanitized: dict[str, float] = {}
        for row_key, value in raw.items():
            if row_key not in ROW_KEYS:
                continue
            try:
                amount = float(value)
            except (TypeError, ValueError):
                continue
            if 0.0 <= amount <= 1.0:
                sanitized[row_key] = amount
        return sanitized

    def _random_velocity_for_row(row_key: str) -> float:
        vr = velocity_randomization.get(row_key)
        if not vr:
            return 1.0
        min_v, max_v = vr
        return random.uniform(min_v, max_v)

    def _timing_randomization_offset_ms_for_row(row_key: str) -> float:
        amount = timing_randomization_by_row.get(row_key, 0.0)
        if not humanize or amount <= 0.0:
            return 0.0
        max_offset_ms = step_duration_ms * 0.1 * amount
        return random.uniform(0.0, max_offset_ms)

    def _apply_linear_velocity(segment: AudioSegment, velocity: float) -> AudioSegment:
        v = max(0.0, min(1.0, velocity))
        if v >= 1.0:
            return segment
        if v <= 0.0:
            return segment - 120
        return segment.apply_gain(20.0 * math.log10(v))

    ROW_KEYS = {
        "kick",
        "snar",
        "ghos",
        "clap",
        "hhat",
        "halt",
        "shkr",
        "prca",
        "prcb",
        "prcc",
        "tomm",
        "cymb",
    }
    velocity_randomization = _sanitize_velocity_randomization_map(meta.get("velocityRandomization"))
    timing_randomization_by_row = _sanitize_timing_randomization_map(meta.get("timingRandomization"))

    approx_loop_ms = steps * step_duration_ms + swing_triplet_offset_ms * swing_clamped + 100
    track = AudioSegment.silent(duration=int(round(approx_loop_ms))).set_frame_rate(BEAT_EXPORT_SAMPLE_RATE)

    for i in range(steps):
        base_start_ms = i * step_duration_ms
        swing_offset_ms = swing_offset_for_step(i)
        step_start_ms = base_start_ms + swing_offset_ms
        position_ms = max(0, int(round(step_start_ms)))

        def overlay_row(row_key: str, pattern_value: int, sample: AudioSegment, base_ms: float):
            nonlocal track
            if not pattern_value:
                return
            velocity = _random_velocity_for_row(row_key)
            human_offset = _timing_randomization_offset_ms_for_row(row_key)
            position = max(0, int(round(base_ms + human_offset)))
            # Match browser preview behavior: play full one-shot sample per trigger.
            # Truncating to a single step makes rendered exports sound unnaturally choked.
            segment = _apply_linear_velocity(sample, velocity)
            track = track.overlay(segment, position=position)

        overlay_row("kick", kick_pattern[i], kick, step_start_ms)
        overlay_row("snar", snare_pattern[i], snare, step_start_ms)
        overlay_row("ghos", ghost_snare_pattern[i], ghost_snare, step_start_ms)
        overlay_row("clap", clap_pattern[i], clap, step_start_ms)
        overlay_row("hhat", hihat_pattern[i], hihat, step_start_ms)
        overlay_row("halt", hihat_alt_pattern[i], hihat_alt, step_start_ms)
        overlay_row("shkr", shaker_pattern[i], shaker, step_start_ms)
        overlay_row("prca", perc_a_pattern[i], perc_a, step_start_ms)
        overlay_row("prcb", perc_b_pattern[i], perc_b, step_start_ms)
        overlay_row("prcc", perc_c_pattern[i], perc_c, step_start_ms)
        overlay_row("tomm", tom_pattern[i], tom, step_start_ms)
        overlay_row("cymb", cymbal_pattern[i], cymbal, step_start_ms)

    # Trim to exact loop length so the exported WAV loops seamlessly and lines up in DAWs.
    exact_loop_sec = length * beats_per_bar * 60.0 / bpm
    exact_sample_count = round(exact_loop_sec * BEAT_EXPORT_SAMPLE_RATE)
    exact_loop_ms = exact_sample_count * 1000.0 / BEAT_EXPORT_SAMPLE_RATE
    one_loop = track[: int(round(exact_loop_ms))]

    # Repeat loop N times when loops > 1
    loops_clamped = max(1, int(loops))
    if loops_clamped > 1:
        track = one_loop * loops_clamped
    else:
        track = one_loop

    # Export as 16-bit, 44.1 kHz for DAW compatibility (Ableton, etc.). Ableton rejects
    # 32-bit and non-standard WAV formats. Normalize to this format so tempo analysis works.
    track = track.set_frame_rate(BEAT_EXPORT_SAMPLE_RATE).set_sample_width(2)
    track.export(output, format="wav")
    print(with_generate_beat_prompt(f"exported {output}"))

    if play:
        print(with_generate_beat_prompt("playing..."))
        open_file_with_default_player(output)

    print(f"{CYAN}│{RESET}")
    return output


def main():
    args = sys.argv[1:]

    if not args:
        print("Error: input_path is required (.mid)")
        return

    input_path = args[0]
    output_path = args[1] if len(args) > 1 else "generated_sample.wav"

    generate_drone_sample(input_path, output_path)
    apply_post_processing_actions(output_path, [])


if __name__ == "__main__":
    main()
