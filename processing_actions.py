import copy

from process_sample import (
    apply_bandpass_to_sample,
    apply_chorus_heavy_to_sample,
    apply_chorus_medium_to_sample,
    apply_chorus_mild_to_sample,
    apply_compress_heavy_to_sample,
    apply_compress_medium_to_sample,
    apply_compress_mild_to_sample,
    apply_distortion_heavy_to_sample,
    apply_distortion_medium_to_sample,
    apply_distortion_mild_to_sample,
    apply_eq_highs_to_sample,
    apply_eq_lows_to_sample,
    apply_eq_mids_to_sample,
    apply_flanger_heavy_to_sample,
    apply_flanger_medium_to_sample,
    apply_flanger_mild_to_sample,
    apply_highpass_to_sample,
    apply_lowpass_to_sample,
    apply_overdrive_heavy_to_sample,
    apply_overdrive_medium_to_sample,
    apply_overdrive_mild_to_sample,
    apply_phaser_heavy_to_sample,
    apply_phaser_medium_to_sample,
    apply_phaser_mild_to_sample,
    apply_pitch_shift_preserve_length,
    apply_reverb_amphitheatre_to_sample,
    apply_reverb_bedroom_to_sample,
    apply_reverb_hall_to_sample,
    apply_reverb_large_to_sample,
    apply_reverb_room_to_sample,
    apply_reverb_space_to_sample,
    apply_time_stretch_simple,
    decrease_sample_gain,
    fade_sample_end,
    fade_sample_start,
    increase_sample_gain,
)

PROCESSING_TYPES = [
    {"key": "fade", "label": "Fade"},
    {"key": "eq", "label": "EQ"},
    {"key": "gain", "label": "Gain"},
    {"key": "filter", "label": "Filter"},
    {"key": "stretch", "label": "Stretch"},
    {"key": "pitch", "label": "Pitch"},
    {"key": "reverb", "label": "Reverb"},
    {"key": "compress", "label": "Compress"},
    {"key": "overdrive", "label": "Overdrive"},
    {"key": "distort", "label": "Distort"},
    {"key": "chorus", "label": "Chorus"},
    {"key": "flanger", "label": "Flanger"},
    {"key": "phaser", "label": "Phaser"},
]

PROCESSING_ACTIONS = [
    {"token": "fade:in 2s", "type": "fade", "label": "In 2s", "command": "fade_sample_start", "params": {"seconds": 2}},
    {"token": "fade:in 5s", "type": "fade", "label": "In 5s", "command": "fade_sample_start", "params": {"seconds": 5}},
    {"token": "fade:out 2s", "type": "fade", "label": "Out 2s", "command": "fade_sample_end", "params": {"seconds": 2}},
    {"token": "fade:out 5s", "type": "fade", "label": "Out 5s", "command": "fade_sample_end", "params": {"seconds": 5}},
    {"token": "eq:lows +5db", "type": "eq", "label": "Lows +5dB", "command": "eq_lows_sample", "params": {"db": 5}},
    {"token": "eq:lows -5db", "type": "eq", "label": "Lows -5dB", "command": "eq_lows_sample", "params": {"db": -5}},
    {"token": "eq:mids +5db", "type": "eq", "label": "Mids +5dB", "command": "eq_mids_sample", "params": {"db": 5}},
    {"token": "eq:mids -5db", "type": "eq", "label": "Mids -5dB", "command": "eq_mids_sample", "params": {"db": -5}},
    {"token": "eq:highs +5db", "type": "eq", "label": "Highs +5dB", "command": "eq_highs_sample", "params": {"db": 5}},
    {"token": "eq:highs -5db", "type": "eq", "label": "Highs -5dB", "command": "eq_highs_sample", "params": {"db": -5}},
    {"token": "gain:+2db", "type": "gain", "label": "+2db", "command": "increase_sample_gain", "params": {"db": 2}},
    {"token": "gain:-2db", "type": "gain", "label": "-2db", "command": "decrease_sample_gain", "params": {"db": 2}},
    {"token": "filter:lpf", "type": "filter", "label": "LPF", "command": "lpf_sample", "params": {}},
    {"token": "filter:lpf-", "type": "filter", "label": "LPF-", "command": "lpf_sample", "params": {"cutoff_hz": 2500}},
    {"token": "filter:lpf--", "type": "filter", "label": "LPF--", "command": "lpf_sample", "params": {"cutoff_hz": 800}},
    {"token": "filter:hpf", "type": "filter", "label": "HPF", "command": "hpf_sample", "params": {}},
    {"token": "filter:hpf+", "type": "filter", "label": "HPF+", "command": "hpf_sample", "params": {"cutoff_hz": 350}},
    {"token": "filter:hpf++", "type": "filter", "label": "HPF++", "command": "hpf_sample", "params": {"cutoff_hz": 800}},
    {"token": "filter:bpf", "type": "filter", "label": "BPF", "command": "bpf_sample", "params": {}},
    {"token": "filter:bpf-", "type": "filter", "label": "BPF-", "command": "bpf_sample", "params": {"low_hz": 180, "high_hz": 3200}},
    {"token": "filter:bpf--", "type": "filter", "label": "BPF--", "command": "bpf_sample", "params": {"low_hz": 100, "high_hz": 1000}},
    {"token": "filter:bpf+", "type": "filter", "label": "BPF+", "command": "bpf_sample", "params": {"low_hz": 500, "high_hz": 10000}},
    {"token": "filter:bpf++", "type": "filter", "label": "BPF++", "command": "bpf_sample", "params": {"low_hz": 1200, "high_hz": 16000}},
    {"token": "stretch:50% speed", "type": "stretch", "label": "50% Speed", "command": "stretch_sample", "params": {"stretch_factor": 2.0}},
    {"token": "stretch:125% speed", "type": "stretch", "label": "125% Speed", "command": "stretch_sample", "params": {"stretch_factor": 0.8}},
    {"token": "stretch:175% speed", "type": "stretch", "label": "175% Speed", "command": "stretch_sample", "params": {"stretch_factor": 0.5714286}},
    {"token": "stretch:200% speed", "type": "stretch", "label": "200% Speed", "command": "stretch_sample", "params": {"stretch_factor": 0.5}},
    {"token": "pitch:+1", "type": "pitch", "label": "+1", "command": "pitch_shift_sample", "params": {"semitones": 1}},
    {"token": "pitch:+2", "type": "pitch", "label": "+2", "command": "pitch_shift_sample", "params": {"semitones": 2}},
    {"token": "pitch:+12", "type": "pitch", "label": "+12", "command": "pitch_shift_sample", "params": {"semitones": 12}},
    {"token": "pitch:-1", "type": "pitch", "label": "-1", "command": "pitch_shift_sample", "params": {"semitones": -1}},
    {"token": "pitch:-2", "type": "pitch", "label": "-2", "command": "pitch_shift_sample", "params": {"semitones": -2}},
    {"token": "pitch:-12", "type": "pitch", "label": "-12", "command": "pitch_shift_sample", "params": {"semitones": -12}},
    {"token": "reverb:bedroom", "type": "reverb", "label": "Bedroom", "command": "reverb_bedroom_sample", "params": {}},
    {"token": "reverb:classroom", "type": "reverb", "label": "Classroom", "command": "reverb_room_sample", "params": {}},
    {"token": "reverb:warehouse", "type": "reverb", "label": "Warehouse", "command": "reverb_hall_sample", "params": {}},
    {"token": "reverb:large", "type": "reverb", "label": "Large", "command": "reverb_large_sample", "params": {}},
    {"token": "reverb:ampitheatre", "type": "reverb", "label": "Ampitheatre", "command": "reverb_amphitheatre_sample", "params": {}},
    {"token": "reverb:space", "type": "reverb", "label": "Space", "command": "reverb_space_sample", "params": {}},
    {"token": "compress:mild", "type": "compress", "label": "Mild", "command": "compress_mild_sample", "params": {}},
    {"token": "compress:medium", "type": "compress", "label": "Medium", "command": "compress_medium_sample", "params": {}},
    {"token": "compress:heavy", "type": "compress", "label": "Heavy", "command": "compress_heavy_sample", "params": {}},
    {"token": "overdrive:mild", "type": "overdrive", "label": "Mild", "command": "overdrive_mild_sample", "params": {}},
    {"token": "overdrive:medium", "type": "overdrive", "label": "Medium", "command": "overdrive_medium_sample", "params": {}},
    {"token": "overdrive:heavy", "type": "overdrive", "label": "Heavy", "command": "overdrive_heavy_sample", "params": {}},
    {"token": "distort:mild", "type": "distort", "label": "Mild", "command": "distort_mild_sample", "params": {}},
    {"token": "distort:medium", "type": "distort", "label": "Medium", "command": "distort_medium_sample", "params": {}},
    {"token": "distort:heavy", "type": "distort", "label": "Heavy", "command": "distort_heavy_sample", "params": {}},
    {"token": "chorus:mild", "type": "chorus", "label": "Mild", "command": "chorus_mild_sample", "params": {}},
    {"token": "chorus:medium", "type": "chorus", "label": "Medium", "command": "chorus_medium_sample", "params": {}},
    {"token": "chorus:heavy", "type": "chorus", "label": "Heavy", "command": "chorus_heavy_sample", "params": {}},
    {"token": "flanger:mild", "type": "flanger", "label": "Mild", "command": "flanger_mild_sample", "params": {}},
    {"token": "flanger:medium", "type": "flanger", "label": "Medium", "command": "flanger_medium_sample", "params": {}},
    {"token": "flanger:heavy", "type": "flanger", "label": "Heavy", "command": "flanger_heavy_sample", "params": {}},
    {"token": "phaser:mild", "type": "phaser", "label": "Mild", "command": "phaser_mild_sample", "params": {}},
    {"token": "phaser:medium", "type": "phaser", "label": "Medium", "command": "phaser_medium_sample", "params": {}},
    {"token": "phaser:heavy", "type": "phaser", "label": "Heavy", "command": "phaser_heavy_sample", "params": {}},
]

PROCESSING_ACTIONS_BY_TOKEN = {a["token"]: a for a in PROCESSING_ACTIONS}


def get_processing_actions_payload() -> dict:
    return {
        "types": copy.deepcopy(PROCESSING_TYPES),
        "actions": copy.deepcopy(PROCESSING_ACTIONS),
    }


def parse_post_processing_spec(spec: str | None) -> list[dict]:
    if not spec:
        return []
    tokens = [part.strip().lower() for part in spec.split(",") if part.strip()]
    actions: list[dict] = []
    unknown: list[str] = []
    for token in tokens:
        action = PROCESSING_ACTIONS_BY_TOKEN.get(token)
        if not action:
            unknown.append(token)
            continue
        actions.append(action)
    if unknown:
        raise ValueError(
            "Unknown post-processing action(s): "
            + ", ".join(unknown)
            + ". Use the action names from auditionr processing controls."
        )
    return actions


def apply_processing_command(file_path: str, command: str, params: dict | None = None) -> None:
    p = params or {}
    match command:
        case "fade_sample_start":
            fade_sample_start(file_path, p.get("seconds", 2))
        case "fade_sample_end":
            fade_sample_end(file_path, p.get("seconds", 2))
        case "increase_sample_gain":
            increase_sample_gain(file_path, p.get("db", 2))
        case "decrease_sample_gain":
            decrease_sample_gain(file_path, p.get("db", 2))
        case "stretch_sample":
            apply_time_stretch_simple(file_path, p.get("stretch_factor", 1.0))
        case "pitch_shift_sample":
            apply_pitch_shift_preserve_length(file_path, p.get("semitones", 0))
        case "reverb_bedroom_sample":
            apply_reverb_bedroom_to_sample(file_path)
        case "reverb_room_sample":
            apply_reverb_room_to_sample(file_path)
        case "reverb_hall_sample":
            apply_reverb_hall_to_sample(file_path)
        case "reverb_large_sample":
            apply_reverb_large_to_sample(file_path)
        case "reverb_amphitheatre_sample":
            apply_reverb_amphitheatre_to_sample(file_path)
        case "reverb_space_sample":
            apply_reverb_space_to_sample(file_path)
        case "compress_mild_sample":
            apply_compress_mild_to_sample(file_path)
        case "compress_medium_sample":
            apply_compress_medium_to_sample(file_path)
        case "compress_heavy_sample":
            apply_compress_heavy_to_sample(file_path)
        case "overdrive_mild_sample":
            apply_overdrive_mild_to_sample(file_path)
        case "overdrive_medium_sample":
            apply_overdrive_medium_to_sample(file_path)
        case "overdrive_heavy_sample":
            apply_overdrive_heavy_to_sample(file_path)
        case "distort_mild_sample":
            apply_distortion_mild_to_sample(file_path)
        case "distort_medium_sample":
            apply_distortion_medium_to_sample(file_path)
        case "distort_heavy_sample":
            apply_distortion_heavy_to_sample(file_path)
        case "chorus_mild_sample":
            apply_chorus_mild_to_sample(file_path)
        case "chorus_medium_sample":
            apply_chorus_medium_to_sample(file_path)
        case "chorus_heavy_sample":
            apply_chorus_heavy_to_sample(file_path)
        case "flanger_mild_sample":
            apply_flanger_mild_to_sample(file_path)
        case "flanger_medium_sample":
            apply_flanger_medium_to_sample(file_path)
        case "flanger_heavy_sample":
            apply_flanger_heavy_to_sample(file_path)
        case "phaser_mild_sample":
            apply_phaser_mild_to_sample(file_path)
        case "phaser_medium_sample":
            apply_phaser_medium_to_sample(file_path)
        case "phaser_heavy_sample":
            apply_phaser_heavy_to_sample(file_path)
        case "lpf_sample":
            apply_lowpass_to_sample(file_path, cutoff_hz=p.get("cutoff_hz", 6000))
        case "hpf_sample":
            apply_highpass_to_sample(file_path, cutoff_hz=p.get("cutoff_hz", 100))
        case "bpf_sample":
            apply_bandpass_to_sample(
                file_path,
                low_hz=p.get("low_hz", 300),
                high_hz=p.get("high_hz", 6000),
            )
        case "eq_lows_sample":
            apply_eq_lows_to_sample(file_path, p.get("db", 0))
        case "eq_mids_sample":
            apply_eq_mids_to_sample(file_path, p.get("db", 0))
        case "eq_highs_sample":
            apply_eq_highs_to_sample(file_path, p.get("db", 0))
        case _:
            raise ValueError(f"Unsupported processing command: {command}")


def apply_post_processing_actions(file_path: str, actions: list[dict]) -> None:
    for action in actions:
        apply_processing_command(
            file_path,
            action.get("command", ""),
            action.get("params", {}),
        )
