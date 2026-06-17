import dronmakr.audio.audio_host as audio_host  # noqa: F401 — DawDreamer before numba-backed dsp

import sys
import os
import json
import random
import subprocess
from pathlib import Path
from dronmakr.core.settings import get_setting, parse_escaped_csv
from pydub import AudioSegment
import soundfile as sf
import numpy as np
from mido import MidiFile, Message

import dronmakr.audio.dsp as dsp
from dronmakr.audio.audio_host import (
    HEADROOM_GAIN,
    SAMPLE_RATE,
    daw_audio_to_samples_channels,
    render_midi_chain_from_paths,
    render_wav_through_fx_paths,
    samples_channels_to_daw_audio,
)

from dronmakr.generate.generate_midi import get_beat_patterns, midi_musical_end_seconds
from dronmakr.processing.processing_actions import apply_post_processing_actions
from dronmakr.core.utils import (
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

    return midi_messages, mid.length


SAMPLE_RATE = audio_host.SAMPLE_RATE


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
    presets_path=None,
    instrument=None,
    effect=None,
    render_duration_sec: float | None = None,
):
    presets_path = presets_path or resolve_presets_index_path()
    if not presets_path:
        raise FileNotFoundError(
            "config/presets.json does not exist — open Patchcraftr from the desktop tray (Launch patchcraftr)."
        )

    # Desktop tray mode: Flask runs on a worker thread; many plug-ins require the process main thread.
    from dronmakr.audio.audio_worker import delegate_generate_drone_sample_if_needed

    delegated = delegate_generate_drone_sample_if_needed(
        os.path.abspath(input_path),
        os.path.abspath(output_path),
        os.path.abspath(presets_path),
        instrument,
        effect,
        render_duration_sec,
    )
    if delegated is not None:
        return delegated

    print(generate_drone_sample_header())

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

    if instrument_preset.get("plugin_name"):
        print(
            with_prompt(
                f"loading instrument {GREEN}{extract_plugin(instrument_preset['plugin_path'])}{RESET} as {GREEN}{instrument_preset['plugin_name']}{RESET}"
            )
        )
    else:
        print(
            with_prompt(
                f"loading instrument {GREEN}{extract_plugin(instrument_preset['plugin_path'])}{RESET}"
            )
        )

    fx_specs = []
    if slots:
        print(with_prompt(f"loading effect {GREEN}{effect_preset['name']}{RESET}"))
        for eff in slots:
            if eff.get("plugin_name"):
                print(
                    with_prompt(
                        f"inserting {GREEN}{extract_plugin(eff['plugin_path'])}{RESET} as {GREEN}{eff['plugin_name']}{RESET}"
                    )
                )
            else:
                print(
                    with_prompt(
                        f"inserting {GREEN}{extract_plugin(eff['plugin_path'])}{RESET}"
                    )
                )
            print(with_prompt(f"using effect step {GREEN}{eff['name']}{RESET}"))
            fx_specs.append((eff["plugin_path"], eff["preset_path"]))
    else:
        print(with_prompt("Skipping effect plug-ins — none listed in presets.json."))

    audio_length_s = (
        float(render_duration_sec)
        if render_duration_sec is not None
        else midi_musical_end_seconds(input_path)
    )
    print(with_prompt(f"sending midi and rendering audio ({audio_length_s:.2f}s)"))

    post_fx_signal = render_midi_chain_from_paths(
        instrument_preset["plugin_path"],
        instrument_preset["preset_path"],
        fx_specs,
        input_path,
        duration_sec=audio_length_s,
        headroom_gain=HEADROOM_GAIN,
    )

    output_path = output_path.replace("#", "sharp")

    sf.write(output_path, post_fx_signal, SAMPLE_RATE, subtype="PCM_16")

    print(f"{GREEN}│{RESET}")

    return output_path


def apply_effect(input_path, effect_chain, presets_path=None):
    """Applies an effect chain from presets to a WAV file and overwrites it after backing up."""

    presets_path = presets_path or resolve_presets_index_path()
    if not presets_path:
        raise FileNotFoundError(
            "config/presets.json does not exist — open Patchcraftr from the desktop tray (Launch patchcraftr)."
        )
    from dronmakr.audio.audio_worker import delegate_apply_effect_if_needed

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

    fx_specs = []
    for eff in slot_list:
        if eff.get("plugin_name"):
            print(
                f"inserting {extract_plugin(eff['plugin_path'])} as {eff['plugin_name']}"
            )
        else:
            print(f"inserting {extract_plugin(eff['plugin_path'])}")
        print(f"using preset {eff['name']}")
        fx_specs.append((eff["plugin_path"], eff["preset_path"]))

    processed, sample_rate = render_wav_through_fx_paths(input_path, fx_specs)
    normalized = dsp.apply_master_normalization_chain(
        samples_channels_to_daw_audio(processed), sample_rate
    )
    out = daw_audio_to_samples_channels(normalized)

    sf.write(output_path, out, sample_rate, subtype="PCM_16")

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
# Drum loops: peak at 0 dBFS with makeup gain so level matches beatbuildr preview / legacy exports.
BEAT_EXPORT_PEAK_DB = 0.0
BEAT_EXPORT_MAKEUP_DB = 10.0

BEAT_ROW_KEYS = frozenset(
    {
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
)


def _resample_beat_hit(
    audio: np.ndarray, orig_sr: int, target_sr: int = BEAT_EXPORT_SAMPLE_RATE
) -> np.ndarray:
    if orig_sr == target_sr:
        return np.ascontiguousarray(audio, dtype=np.float32)
    import librosa  # noqa: PLC0415

    if audio.ndim == 1:
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(
            np.float32
        )
    resampled = [
        librosa.resample(audio[:, ch], orig_sr=orig_sr, target_sr=target_sr)
        for ch in range(audio.shape[1])
    ]
    return np.stack(resampled, axis=1).astype(np.float32)


def _load_beat_hit(path: str | Path) -> np.ndarray:
    """Load a one-shot as float32 (samples, channels) at BEAT_EXPORT_SAMPLE_RATE."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise ValueError(f"Sample file not found: {path}")
    audio, sr = sf.read(str(p), dtype="float32", always_2d=True)
    return _resample_beat_hit(audio, int(sr))


def _get_random_beat_hit(folder: Path) -> np.ndarray | None:
    files = [f for f in folder.iterdir() if f.suffix.lower() == ".wav"]
    if not files:
        return None
    return _load_beat_hit(random.choice(files))


def _beat_hit_channel_count(hit: np.ndarray) -> int:
    return 1 if hit.ndim == 1 else int(hit.shape[1])


def _align_hit_channels(hit: np.ndarray, channels: int) -> np.ndarray:
    if hit.ndim == 1:
        hit = hit[:, np.newaxis]
    hit_ch = hit.shape[1]
    if hit_ch == channels:
        return hit
    if hit_ch == 1 and channels == 2:
        return np.repeat(hit, 2, axis=1)
    if hit_ch == 2 and channels == 1:
        return hit.mean(axis=1, keepdims=True)
    return hit[:, :channels]


def _mix_hit_into_buffer(
    mix: np.ndarray,
    hit: np.ndarray,
    start_sample: int,
    velocity: float,
) -> None:
    """Sum a velocity-scaled one-shot into a float mix buffer (matches Web Audio gain)."""
    gain = max(0.0, min(1.0, float(velocity)))
    if gain <= 0.0 or start_sample >= mix.shape[0]:
        return
    hit = _align_hit_channels(hit, mix.shape[1])
    scaled = hit * gain
    end = min(mix.shape[0], start_sample + scaled.shape[0])
    count = end - start_sample
    if count <= 0:
        return
    mix[start_sample:end, :] += scaled[:count, :]


def _finalize_beat_export_mix(mix: np.ndarray) -> np.ndarray:
    """
    Apply makeup gain then peak-normalize for beat exports.

    Float32 summing avoids the old pydub 16-bit clip "fattening", so we add makeup
    gain before peaking at BEAT_EXPORT_PEAK_DB to match beatbuildr preview loudness.
    """
    out = np.asarray(mix, dtype=np.float64)
    if out.size == 0:
        return out.astype(np.float32)
    out *= 10 ** (BEAT_EXPORT_MAKEUP_DB / 20.0)
    peak = float(np.max(np.abs(out)))
    if peak <= 1e-12:
        return np.zeros_like(out, dtype=np.float32)
    target = 10 ** (BEAT_EXPORT_PEAK_DB / 20.0)
    out *= target / peak
    return np.clip(out, -1.0, 1.0).astype(np.float32)


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
        def _load(row: str, fallback_row: str | None = None) -> np.ndarray | None:
            path = kit_paths.get(row) or (kit_paths.get(fallback_row) if fallback_row else None)
            if not path:
                return None
            return _load_beat_hit(path)

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
        kick = _get_random_beat_hit(kicks)
        snare = _get_random_beat_hit(snares)
        ghost_snare = _get_random_beat_hit(snares)
        hihat = _get_random_beat_hit(hihats)
        hihat_alt = _get_random_beat_hit(hihats)
        perc_a = _get_random_beat_hit(percs)
        perc_b = _get_random_beat_hit(percs)
        perc_c = _get_random_beat_hit(percs)
        clap = _get_random_beat_hit(claps)
        tom = _get_random_beat_hit(toms)
        shaker = _get_random_beat_hit(shakers)
        cymbal = _get_random_beat_hit(cymbals)

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
            if row_key not in BEAT_ROW_KEYS:
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
            if row_key not in BEAT_ROW_KEYS:
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

    velocity_randomization = _sanitize_velocity_randomization_map(meta.get("velocityRandomization"))
    timing_randomization_by_row = _sanitize_timing_randomization_map(meta.get("timingRandomization"))

    row_hits: dict[str, np.ndarray | None] = {
        "kick": kick,
        "snar": snare,
        "ghos": ghost_snare,
        "clap": clap,
        "hhat": hihat,
        "halt": hihat_alt,
        "shkr": shaker,
        "prca": perc_a,
        "prcb": perc_b,
        "prcc": perc_c,
        "tomm": tom,
        "cymb": cymbal,
    }
    channel_count = 1
    for hit in row_hits.values():
        if hit is not None:
            channel_count = max(channel_count, _beat_hit_channel_count(hit))

    approx_loop_ms = steps * step_duration_ms + swing_triplet_offset_ms * swing_clamped + 100
    exact_loop_sec = length * beats_per_bar * 60.0 / bpm
    exact_sample_count = round(exact_loop_sec * BEAT_EXPORT_SAMPLE_RATE)
    approx_sample_count = int(round(approx_loop_ms * BEAT_EXPORT_SAMPLE_RATE / 1000.0))
    buffer_samples = max(exact_sample_count, approx_sample_count)
    mix = np.zeros((buffer_samples, channel_count), dtype=np.float32)

    for i in range(steps):
        base_start_ms = i * step_duration_ms
        swing_offset_ms = swing_offset_for_step(i)
        step_start_ms = base_start_ms + swing_offset_ms

        def overlay_row(row_key: str, pattern_value: int, hit: np.ndarray | None, base_ms: float):
            if not pattern_value or hit is None:
                return
            velocity = _random_velocity_for_row(row_key)
            human_offset = _timing_randomization_offset_ms_for_row(row_key)
            position_ms = max(0.0, base_ms + human_offset)
            start_sample = int(round(position_ms * BEAT_EXPORT_SAMPLE_RATE / 1000.0))
            # Match browser preview: full one-shot per trigger, float32 linear gain.
            _mix_hit_into_buffer(mix, hit, start_sample, velocity)

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
    one_loop = mix[:exact_sample_count]

    # Repeat loop N times when loops > 1
    loops_clamped = max(1, int(loops))
    if loops_clamped > 1:
        mix = np.tile(one_loop, (loops_clamped, 1))
    else:
        mix = one_loop

    # Export as 16-bit, 44.1 kHz for DAW compatibility (Ableton, etc.).
    mix = _finalize_beat_export_mix(mix)
    sf.write(output, mix, BEAT_EXPORT_SAMPLE_RATE, subtype="PCM_16")
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
