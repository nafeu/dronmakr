import copy
import re

from process_sample import (
    apply_bandpass_to_sample,
    apply_chorus_heavy_to_sample,
    apply_chorus_medium_to_sample,
    apply_chorus_mild_to_sample,
    apply_chorus_to_sample,
    apply_compress_heavy_to_sample,
    apply_compress_medium_to_sample,
    apply_compress_mild_to_sample,
    apply_compress_to_sample,
    apply_distortion_heavy_to_sample,
    apply_distortion_medium_to_sample,
    apply_distortion_mild_to_sample,
    apply_distortion_to_sample,
    apply_eq_highs_to_sample,
    apply_eq_lows_to_sample,
    apply_eq_mids_to_sample,
    apply_flanger_heavy_to_sample,
    apply_flanger_medium_to_sample,
    apply_flanger_mild_to_sample,
    apply_flanger_to_sample,
    apply_highpass_to_sample,
    apply_lowpass_to_sample,
    apply_noise_gate_to_sample,
    apply_overdrive_heavy_to_sample,
    apply_overdrive_medium_to_sample,
    apply_overdrive_mild_to_sample,
    apply_overdrive_mids_to_sample,
    apply_phaser_heavy_to_sample,
    apply_phaser_medium_to_sample,
    apply_phaser_mild_to_sample,
    apply_phaser_to_sample,
    apply_pitch_shift_preserve_length,
    apply_paulstretch_to_sample,
    apply_reverb_amphitheatre_to_sample,
    apply_reverb_bedroom_to_sample,
    apply_reverb_hall_to_sample,
    apply_reverb_large_to_sample,
    apply_reverb_room_to_sample,
    apply_reverb_space_to_sample,
    apply_reverb_void_to_sample,
    apply_reverb_distant_to_sample,
    apply_reverb_to_sample,
    apply_time_stretch_simple,
    apply_transpose_pitch_by_resampling_inplace,
    decrease_sample_gain,
    fade_sample_end,
    fade_sample_start,
    increase_sample_gain,
    normalize_sample,
    trim_sample_end,
    trim_sample_start,
)

PROCESSING_TYPES = [
    {"key": "cut", "label": "Cut"},
    {"key": "fade", "label": "Fade"},
    {"key": "eq", "label": "EQ"},
    {"key": "gain", "label": "Gain"},
    {"key": "filter", "label": "Filter"},
    {"key": "normalize", "label": "Normalize"},
    {"key": "noisegate", "label": "Noisegate"},
    {"key": "timestretch", "label": "Time stretch"},
    {"key": "paulstretch", "label": "Paulstretch"},
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
    {"token": "gain:+5db", "type": "gain", "label": "+5db", "command": "increase_sample_gain", "params": {"db": 5}},
    {"token": "gain:+10db", "type": "gain", "label": "+10db", "command": "increase_sample_gain", "params": {"db": 10}},
    {"token": "gain:-2db", "type": "gain", "label": "-2db", "command": "decrease_sample_gain", "params": {"db": 2}},
    {"token": "gain:-5db", "type": "gain", "label": "-5db", "command": "decrease_sample_gain", "params": {"db": 5}},
    {"token": "gain:-10db", "type": "gain", "label": "-10db", "command": "decrease_sample_gain", "params": {"db": 10}},
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
    {"token": "normalize", "type": "normalize", "label": "Normalize", "command": "normalize_sample", "params": {}},
    {"token": "noisegate:fast", "type": "noisegate", "label": "Fast", "command": "noisegate_sample", "params": {"threshold_db": -30, "attack_ms": 2, "release_ms": 50}},
    {"token": "noisegate:medium", "type": "noisegate", "label": "Medium", "command": "noisegate_sample", "params": {"threshold_db": -30, "attack_ms": 8, "release_ms": 140}},
    {"token": "noisegate:slow", "type": "noisegate", "label": "Slow", "command": "noisegate_sample", "params": {"threshold_db": -30, "attack_ms": 20, "release_ms": 280}},
    {"token": "timestretch:50pct", "type": "timestretch", "label": "50%", "command": "stretch_sample", "params": {"stretch_factor": 2.0}},
    {"token": "timestretch:125pct", "type": "timestretch", "label": "125%", "command": "stretch_sample", "params": {"stretch_factor": 0.8}},
    {"token": "timestretch:175pct", "type": "timestretch", "label": "175%", "command": "stretch_sample", "params": {"stretch_factor": 0.5714286}},
    {"token": "timestretch:200pct", "type": "timestretch", "label": "200%", "command": "stretch_sample", "params": {"stretch_factor": 0.5}},
    {
        "token": "paulstretch:8x2.5s",
        "type": "paulstretch",
        "label": "8x 2.5s",
        "command": "paul_stretch_sample",
        "params": {"stretch": 8.0, "window_size": 2.5},
    },
    {
        "token": "paulstretch:12x4s",
        "type": "paulstretch",
        "label": "12x 4s",
        "command": "paul_stretch_sample",
        "params": {"stretch": 12.0, "window_size": 4.0},
    },
    {
        "token": "paulstretch:5x1s",
        "type": "paulstretch",
        "label": "5x 1s",
        "command": "paul_stretch_sample",
        "params": {"stretch": 5.0, "window_size": 1.0},
    },
    {"token": "pitch:+1", "type": "pitch", "label": "+1", "command": "pitch_shift_sample", "params": {"semitones": 1}},
    {"token": "pitch:+2", "type": "pitch", "label": "+2", "command": "pitch_shift_sample", "params": {"semitones": 2}},
    {"token": "pitch:+12", "type": "pitch", "label": "+12", "command": "pitch_shift_sample", "params": {"semitones": 12}},
    {"token": "pitch:-1", "type": "pitch", "label": "-1", "command": "pitch_shift_sample", "params": {"semitones": -1}},
    {"token": "pitch:-2", "type": "pitch", "label": "-2", "command": "pitch_shift_sample", "params": {"semitones": -2}},
    {"token": "pitch:-12", "type": "pitch", "label": "-12", "command": "pitch_shift_sample", "params": {"semitones": -12}},
    {
        "token": "pitch:-12resample",
        "type": "pitch",
        "label": "-12 resample",
        "command": "pitch_shift_transpose_sample",
        "params": {"semitones": -12},
    },
    {
        "token": "pitch:+12resample",
        "type": "pitch",
        "label": "+12 resample",
        "command": "pitch_shift_transpose_sample",
        "params": {"semitones": 12},
    },
    {
        "token": "pitch:-6resample",
        "type": "pitch",
        "label": "-6 resample",
        "command": "pitch_shift_transpose_sample",
        "params": {"semitones": -6},
    },
    {
        "token": "pitch:+6resample",
        "type": "pitch",
        "label": "+6 resample",
        "command": "pitch_shift_transpose_sample",
        "params": {"semitones": 6},
    },
    {"token": "reverb:bedroom", "type": "reverb", "label": "Bedroom", "command": "reverb_bedroom_sample", "params": {}},
    {"token": "reverb:classroom", "type": "reverb", "label": "Classroom", "command": "reverb_room_sample", "params": {}},
    {"token": "reverb:warehouse", "type": "reverb", "label": "Warehouse", "command": "reverb_hall_sample", "params": {}},
    {"token": "reverb:large", "type": "reverb", "label": "Large", "command": "reverb_large_sample", "params": {}},
    {"token": "reverb:ampitheatre", "type": "reverb", "label": "Ampitheatre", "command": "reverb_amphitheatre_sample", "params": {}},
    {"token": "reverb:space", "type": "reverb", "label": "Space", "command": "reverb_space_sample", "params": {}},
    {"token": "reverb:void", "type": "reverb", "label": "Void", "command": "reverb_void_sample", "params": {}},
    {"token": "reverb:distant", "type": "reverb", "label": "Distant (late bloom)", "command": "reverb_distant_sample", "params": {}},
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

# paulstretch:12x4s — stretch amount × window seconds
PAULSTRETCH_COMPACT_TOKEN_RE = re.compile(r"^paulstretch:(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)s$")
# Legacy: stretch:paul8s2.5w
PAUL_STRETCH_TOKEN_RE = re.compile(r"^stretch:paul(\d+(?:\.\d+)?)s(\d+(?:\.\d+)?)w$")

_LEGACY_PROCESSING_TOKEN_ACTIONS: dict[str, dict[str, object]] = {
    "stretch:50% speed": {
        "token": "stretch:50% speed",
        "type": "timestretch",
        "command": "stretch_sample",
        "params": {"stretch_factor": 2.0},
    },
    "stretch:125% speed": {
        "token": "stretch:125% speed",
        "type": "timestretch",
        "command": "stretch_sample",
        "params": {"stretch_factor": 0.8},
    },
    "stretch:175% speed": {
        "token": "stretch:175% speed",
        "type": "timestretch",
        "command": "stretch_sample",
        "params": {"stretch_factor": 0.5714286},
    },
    "stretch:200% speed": {
        "token": "stretch:200% speed",
        "type": "timestretch",
        "command": "stretch_sample",
        "params": {"stretch_factor": 0.5},
    },
}
PITCH_RESAMPLE_TOKEN_RE = re.compile(r"^pitch:([+-]?\d+(?:\.\d+)?)resample$")
PITCH_RAW_LEGACY_TOKEN_RE = re.compile(r"^pitch:([+-]?\d+(?:\.\d+)?)raw$")

BRACKET_PAIR_RE = re.compile(r"\[([^\]=]+)\s*=\s*([^\]]*)\]")

_PARAM_SCHEMAS_UI: dict[str, list[dict]] = {
    "cut": [
        {
            "key": "edge",
            "label": "Edge",
            "widget": "select",
            "options": [
                {"value": "start", "label": "Trim start (before playhead)"},
                {"value": "end", "label": "Trim end (after playhead)"},
            ],
            "default": "start",
        },
    ],
    "fade": [
        {
            "key": "style",
            "label": "Style",
            "widget": "select",
            "options": [{"value": "in", "label": "Fade in"}, {"value": "out", "label": "Fade out"}],
            "default": "in",
        },
        {
            "key": "duration_ms_ui",
            "label": "Duration",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.01,
            "default": 0.28,
            "maps_to": "duration_ms",
            "range_ms": [100, 8000],
        },
    ],
    "eq": [
        {
            "key": "band",
            "label": "Band",
            "widget": "select",
            "options": [
                {"value": "lows", "label": "Lows"},
                {"value": "mids", "label": "Mids"},
                {"value": "highs", "label": "Highs"},
            ],
            "default": "lows",
        },
        {
            "key": "db_ui",
            "label": "Gain (dB)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.55,
            "maps_to": "db",
            "range_db": [-12, 12],
        },
    ],
    "gain": [
        {
            "key": "db_ui",
            "label": "Gain change (dB)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.55,
            "maps_to": "db_signed",
            "range_db": [-12, 12],
        },
    ],
    "filter": [
        {
            "key": "kind",
            "label": "Kind",
            "widget": "select",
            "options": [
                {"value": "lpf", "label": "Low-pass"},
                {"value": "hpf", "label": "High-pass"},
                {"value": "bpf", "label": "Band-pass"},
            ],
            "default": "lpf",
        },
        {
            "key": "cutoff_hz_ui",
            "label": "Cutoff (Hz) LPF/HPF",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.005,
            "default": 0.55,
            "maps_to": "cutoff_hz",
            "range_hz": [80, 16000],
        },
        {
            "key": "low_hz_ui",
            "label": "BPF low (Hz)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.005,
            "default": 0.12,
            "maps_to": "low_hz",
            "range_hz": [40, 8000],
        },
        {
            "key": "high_hz_ui",
            "label": "BPF high (Hz)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.005,
            "default": 0.72,
            "maps_to": "high_hz",
            "range_hz": [200, 18000],
        },
    ],
    "normalize": [],
    "noisegate": [
        {
            "key": "threshold_db_ui",
            "label": "Threshold",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.45,
            "maps_to": "threshold_db",
            "range_db": [-55, -12],
        },
        {
            "key": "attack_ms_ui",
            "label": "Attack",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.25,
            "maps_to": "attack_ms",
            "range_ms": [1, 40],
        },
        {
            "key": "release_ms_ui",
            "label": "Release",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.4,
            "maps_to": "release_ms",
            "range_ms": [40, 400],
        },
        {
            "key": "ratio_ui",
            "label": "Ratio",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.45,
            "maps_to": "ratio",
            "range_linear": [2.0, 20.0],
        },
    ],
    "timestretch": [
        {
            "key": "stretch_factor_ui",
            "label": "Stretch factor",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.35,
            "maps_to": "stretch_factor",
            "range_linear": [2.5, 0.4],
            "hint": "Resampling stretch: >1 slows down (longer + lower pitch), <1 speeds up.",
        },
    ],
    "paulstretch": [
        {
            "key": "paul_stretch_ui",
            "label": "Stretch amount",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.35,
            "maps_to": "stretch_amount",
            "range_linear": [2.0, 16.0],
        },
        {
            "key": "paul_window_ui",
            "label": "Window (s)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.35,
            "maps_to": "window_size",
            "range_linear": [0.5, 6.0],
        },
    ],
    "pitch": [
        {
            "key": "mode",
            "label": "Mode",
            "widget": "select",
            "options": [{"value": "preserve", "label": "Preserve length"}, {"value": "resample", "label": "Resample"}],
            "default": "preserve",
        },
        {
            "key": "semitones_ui",
            "label": "Semitones",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.56,
            "maps_to": "semitones",
            "range_linear": [-24, 24],
        },
    ],
    "reverb": [
        {
            "key": "wet_level_ui",
            "label": "Wet",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.35,
            "maps_to": "wet_level",
            "range_linear": [0.05, 0.85],
        },
        {
            "key": "length_sec_ui",
            "label": "IR length (s)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.35,
            "maps_to": "length_sec",
            "range_linear": [0.15, 16.0],
        },
        {
            "key": "decay_sec_ui",
            "label": "Decay (s)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.35,
            "maps_to": "decay_sec",
            "range_linear": [0.1, 14.0],
        },
        {
            "key": "highpass_hz_ui",
            "label": "Reverb HPF (Hz)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.2,
            "maps_to": "highpass_hz",
            "range_linear": [30.0, 200.0],
        },
        {
            "key": "early_reflections_ui",
            "label": "Early reflections",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.05,
            "default": 0.45,
            "maps_to": "early_reflections",
            "range_linear": [2.0, 16.0],
        },
        {
            "key": "tail_diffusion_ui",
            "label": "Tail diffusion",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.75,
            "maps_to": "tail_diffusion",
            "range_linear": [0.5, 0.98],
        },
        {
            "key": "early_diffuse",
            "label": "Early diffuse",
            "widget": "select",
            "options": [{"value": "true", "label": "On"}, {"value": "false", "label": "Off"}],
            "default": "true",
        },
    ],
    "compress": [
        {
            "key": "threshold_db_ui",
            "label": "Threshold (dB)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.45,
            "maps_to": "threshold_db",
            "range_db": [-35, -8],
        },
        {
            "key": "ratio_ui",
            "label": "Ratio",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.45,
            "maps_to": "ratio",
            "range_linear": [1.5, 20.0],
        },
        {
            "key": "attack_ms_ui",
            "label": "Attack (ms)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.35,
            "maps_to": "attack_ms",
            "range_linear": [0.5, 40.0],
        },
        {
            "key": "release_ms_ui",
            "label": "Release (ms)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.45,
            "maps_to": "release_ms",
            "range_linear": [40.0, 250.0],
        },
    ],
    "overdrive": [
        {
            "key": "drive_db_ui",
            "label": "Drive (dB)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.45,
            "maps_to": "drive_db",
            "range_linear": [4.0, 26.0],
        },
        {
            "key": "highpass_hz_ui",
            "label": "High-pass (Hz)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.35,
            "maps_to": "highpass_hz",
            "range_linear": [80.0, 450.0],
        },
        {
            "key": "lowpass_hz_ui",
            "label": "Low-pass (Hz)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.55,
            "maps_to": "lowpass_hz",
            "range_linear": [2500.0, 7000.0],
        },
    ],
    "distort": [
        {
            "key": "drive_db_ui",
            "label": "Drive (dB)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.45,
            "maps_to": "drive_db",
            "range_linear": [1.0, 18.0],
        },
    ],
    "chorus": [
        {
            "key": "rate_hz_ui",
            "label": "Rate (Hz)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.45,
            "maps_to": "rate_hz",
            "range_linear": [0.3, 2.0],
        },
        {
            "key": "depth_ui",
            "label": "Depth",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.45,
            "maps_to": "depth",
            "range_linear": [0.05, 0.65],
        },
        {
            "key": "centre_delay_ms_ui",
            "label": "Centre delay (ms)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.45,
            "maps_to": "centre_delay_ms",
            "range_linear": [4.0, 14.0],
        },
        {
            "key": "feedback_ui",
            "label": "Feedback",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.25,
            "maps_to": "feedback",
            "range_linear": [0.0, 0.45],
        },
        {
            "key": "mix_ui",
            "label": "Mix",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.45,
            "maps_to": "mix",
            "range_linear": [0.15, 0.85],
        },
    ],
    "flanger": [
        {
            "key": "rate_hz_ui",
            "label": "Rate (Hz)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.45,
            "maps_to": "rate_hz",
            "range_linear": [0.2, 1.2],
        },
        {
            "key": "depth_ui",
            "label": "Depth",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.45,
            "maps_to": "depth",
            "range_linear": [0.1, 0.75],
        },
        {
            "key": "centre_delay_ms_ui",
            "label": "Centre delay (ms)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.35,
            "maps_to": "centre_delay_ms",
            "range_linear": [1.0, 4.0],
        },
        {
            "key": "feedback_ui",
            "label": "Feedback",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.45,
            "maps_to": "feedback",
            "range_linear": [0.05, 0.65],
        },
        {
            "key": "mix_ui",
            "label": "Mix",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.45,
            "maps_to": "mix",
            "range_linear": [0.2, 0.85],
        },
    ],
    "phaser": [
        {
            "key": "rate_hz_ui",
            "label": "Rate (Hz)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.45,
            "maps_to": "rate_hz",
            "range_linear": [0.4, 1.8],
        },
        {
            "key": "depth_ui",
            "label": "Depth",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.55,
            "maps_to": "depth",
            "range_linear": [0.35, 0.98],
        },
        {
            "key": "centre_frequency_hz_ui",
            "label": "Centre freq (Hz)",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.45,
            "maps_to": "centre_frequency_hz",
            "range_linear": [400.0, 2800.0],
        },
        {
            "key": "feedback_ui",
            "label": "Feedback",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.45,
            "maps_to": "feedback",
            "range_linear": [0.1, 0.85],
        },
        {
            "key": "mix_ui",
            "label": "Mix",
            "widget": "slider",
            "min": 0,
            "max": 1,
            "step": 0.02,
            "default": 0.45,
            "maps_to": "mix",
            "range_linear": [0.2, 0.85],
        },
    ],
}


def _split_spec_segments(spec: str) -> list[str]:
    depth = 0
    parts: list[str] = []
    buf: list[str] = []
    for ch in spec:
        if ch == "[":
            depth += 1
            buf.append(ch)
        elif ch == "]":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch in ",;" and depth == 0:
            seg = "".join(buf).strip()
            if seg:
                parts.append(seg)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_scalar(raw: str):
    s = raw.strip()
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if "." in s or "e" in low:
            return float(s)
        return int(s)
    except ValueError:
        return s


def _parse_bracket_params(rest: str) -> dict[str, object]:
    params: dict[str, object] = {}
    for m in BRACKET_PAIR_RE.finditer(rest):
        key = m.group(1).strip()
        params[key] = _parse_scalar(m.group(2))
    return params


def _is_bracket_segment(seg: str) -> bool:
    s = seg.strip()
    if re.match(r"^\w+\s*:\s*\[\s*\]\s*$", s):
        return True
    return "[" in s and BRACKET_PAIR_RE.search(s) is not None


def _fade_duration_ms(params: dict[str, object]) -> float:
    if "duration_ms" in params:
        return float(params["duration_ms"])
    if "duration_s" in params:
        return float(params["duration_s"]) * 1000.0
    if "duration" in params:
        v = float(params["duration"])
        # Values >= 500 are treated as milliseconds; smaller numbers as seconds (e.g. 2 → 2s).
        if v >= 500:
            return v
        return v * 1000.0
    raise ValueError("fade: requires duration_ms, duration_s, or duration")


def action_from_bracket_segment(seg: str) -> dict:
    seg_stripped = seg.strip()
    m = re.match(r"^(\w+)\s*:\s*(.*)$", seg_stripped, re.DOTALL)
    if not m:
        raise ValueError(f"Invalid bracket action: {seg_stripped!r}")
    ptype = m.group(1).lower()
    rest = m.group(2).strip()
    params = _parse_bracket_params(rest)

    if ptype == "normalize":
        return {"token": seg_stripped, "command": "normalize_sample", "params": {}}

    if ptype == "cut":
        edge = str(params.get("edge", "")).lower()
        if edge not in ("start", "end"):
            raise ValueError("cut: requires edge=start|end and seconds")
        sec = params.get("seconds")
        if sec is None:
            raise ValueError("cut: requires seconds")
        cmd = "trim_sample_start" if edge == "start" else "trim_sample_end"
        return {"token": seg_stripped, "command": cmd, "params": {"seconds": float(sec)}}

    if ptype == "fade":
        style = str(params.get("style", "")).lower()
        if style not in ("in", "out"):
            raise ValueError("fade: requires style=in|out")
        ms = _fade_duration_ms(params)
        seconds = ms / 1000.0
        cmd = "fade_sample_start" if style == "in" else "fade_sample_end"
        return {"token": seg_stripped, "command": cmd, "params": {"seconds": seconds}}

    if ptype == "eq":
        band = str(params.get("band", "")).lower()
        db = params.get("db")
        if band not in ("lows", "mids", "highs") or db is None:
            raise ValueError("eq: requires band=lows|mids|highs and db")
        cmd = {"lows": "eq_lows_sample", "mids": "eq_mids_sample", "highs": "eq_highs_sample"}[band]
        return {"token": seg_stripped, "command": cmd, "params": {"db": float(db)}}

    if ptype == "gain":
        db = params.get("db")
        if db is None:
            raise ValueError("gain: requires db (signed)")
        dbf = float(db)
        if dbf >= 0:
            return {"token": seg_stripped, "command": "increase_sample_gain", "params": {"db": dbf}}
        return {"token": seg_stripped, "command": "decrease_sample_gain", "params": {"db": abs(dbf)}}

    if ptype == "filter":
        kind = str(params.get("kind", "")).lower()
        if kind == "lpf":
            return {
                "token": seg_stripped,
                "command": "lpf_sample",
                "params": {"cutoff_hz": float(params["cutoff_hz"])} if "cutoff_hz" in params else {},
            }
        if kind == "hpf":
            return {
                "token": seg_stripped,
                "command": "hpf_sample",
                "params": {"cutoff_hz": float(params["cutoff_hz"])} if "cutoff_hz" in params else {},
            }
        if kind == "bpf":
            low = float(params.get("low_hz", 300))
            high = float(params.get("high_hz", 6000))
            return {"token": seg_stripped, "command": "bpf_sample", "params": {"low_hz": low, "high_hz": high}}
        raise ValueError("filter: requires kind=lpf|hpf|bpf")

    if ptype == "noisegate":
        gate_params = {
            "threshold_db": float(params.get("threshold_db", -30)),
            "attack_ms": float(params.get("attack_ms", 8)),
            "release_ms": float(params.get("release_ms", 140)),
        }
        if "ratio" in params:
            gate_params["ratio"] = float(params["ratio"])
        return {"token": seg_stripped, "command": "noisegate_sample", "params": gate_params}

    if ptype == "timestretch":
        return {
            "token": seg_stripped,
            "command": "stretch_sample",
            "params": {"stretch_factor": float(params.get("stretch_factor", 1.0))},
        }

    if ptype == "paulstretch":
        amt = params.get("stretch_amount")
        if amt is None:
            amt = params.get("stretch")
        return {
            "token": seg_stripped,
            "command": "paul_stretch_sample",
            "params": {
                "stretch": float(amt if amt is not None else 8.0),
                "window_size": float(params.get("window_size", 2.5)),
            },
        }

    if ptype == "stretch":
        mode = str(params.get("mode", "simple")).lower()
        if mode == "paul":
            return {
                "token": seg_stripped,
                "command": "paul_stretch_sample",
                "params": {
                    "stretch": float(params.get("stretch", 8.0)),
                    "window_size": float(params.get("window_size", 2.5)),
                },
            }
        return {
            "token": seg_stripped,
            "command": "stretch_sample",
            "params": {"stretch_factor": float(params.get("stretch_factor", 1.0))},
        }

    if ptype == "pitch":
        mode = str(params.get("mode", "preserve")).lower()
        sem = float(params.get("semitones", 0))
        if mode == "resample":
            return {
                "token": seg_stripped,
                "command": "pitch_shift_transpose_sample",
                "params": {"semitones": sem},
            }
        return {"token": seg_stripped, "command": "pitch_shift_sample", "params": {"semitones": sem}}

    if ptype == "reverb":
        out_params: dict[str, object] = {
            "wet_level": float(params.get("wet_level", 0.35)),
            "length_sec": float(params.get("length_sec", params.get("reverb_length_sec", 0.7))),
            "decay_sec": float(params.get("decay_sec", 0.5)),
            "highpass_hz": float(params.get("highpass_hz", params.get("reverb_highpass_hz", 100))),
        }
        if "early_reflections" in params:
            out_params["early_reflections"] = int(params["early_reflections"])
        if "tail_diffusion" in params:
            out_params["tail_diffusion"] = float(params["tail_diffusion"])
        if "early_diffuse" in params:
            ed = params["early_diffuse"]
            if isinstance(ed, str):
                out_params["early_diffuse"] = ed.lower() in ("true", "1", "yes")
            else:
                out_params["early_diffuse"] = bool(ed)
        return {"token": seg_stripped, "command": "reverb_sample", "params": out_params}

    if ptype == "compress":
        return {
            "token": seg_stripped,
            "command": "compress_sample",
            "params": {
                "threshold_db": float(params.get("threshold_db", -20)),
                "ratio": float(params.get("ratio", 10)),
                "attack_ms": float(params.get("attack_ms", 3)),
                "release_ms": float(params.get("release_ms", 80)),
            },
        }

    if ptype == "overdrive":
        return {
            "token": seg_stripped,
            "command": "overdrive_mids_sample",
            "params": {
                "drive_db": float(params.get("drive_db", 14)),
                "highpass_hz": float(params.get("highpass_hz", 200)),
                "lowpass_hz": float(params.get("lowpass_hz", 4000)),
            },
        }

    if ptype == "distort":
        return {
            "token": seg_stripped,
            "command": "distort_sample",
            "params": {"drive_db": float(params.get("drive_db", 6))},
        }

    if ptype == "chorus":
        return {
            "token": seg_stripped,
            "command": "chorus_sample",
            "params": {
                "rate_hz": float(params.get("rate_hz", 1.0)),
                "depth": float(params.get("depth", 0.25)),
                "centre_delay_ms": float(params.get("centre_delay_ms", 7.0)),
                "feedback": float(params.get("feedback", 0.0)),
                "mix": float(params.get("mix", 0.5)),
            },
        }

    if ptype == "flanger":
        return {
            "token": seg_stripped,
            "command": "flanger_sample",
            "params": {
                "rate_hz": float(params.get("rate_hz", 0.5)),
                "depth": float(params.get("depth", 0.4)),
                "centre_delay_ms": float(params.get("centre_delay_ms", 2.0)),
                "feedback": float(params.get("feedback", 0.3)),
                "mix": float(params.get("mix", 0.5)),
            },
        }

    if ptype == "phaser":
        return {
            "token": seg_stripped,
            "command": "phaser_sample",
            "params": {
                "rate_hz": float(params.get("rate_hz", 1.0)),
                "depth": float(params.get("depth", 0.7)),
                "centre_frequency_hz": float(params.get("centre_frequency_hz", 1000)),
                "feedback": float(params.get("feedback", 0.5)),
                "mix": float(params.get("mix", 0.5)),
            },
        }

    raise ValueError(f"Unknown bracket action type: {ptype}")


def _legacy_action_to_spec(action: dict) -> str:
    cmd = action.get("command")
    p = action.get("params") or {}
    if cmd == "fade_sample_start":
        ms = int(round(float(p["seconds"]) * 1000))
        return f"fade:[style=in][duration_ms={ms}]"
    if cmd == "fade_sample_end":
        ms = int(round(float(p["seconds"]) * 1000))
        return f"fade:[style=out][duration_ms={ms}]"
    if cmd == "eq_lows_sample":
        return f"eq:[band=lows][db={p['db']}]"
    if cmd == "eq_mids_sample":
        return f"eq:[band=mids][db={p['db']}]"
    if cmd == "eq_highs_sample":
        return f"eq:[band=highs][db={p['db']}]"
    if cmd == "increase_sample_gain":
        return f"gain:[db={p['db']}]"
    if cmd == "decrease_sample_gain":
        return f"gain:[db=-{p['db']}]"
    if cmd == "lpf_sample":
        if not p:
            return "filter:[kind=lpf]"
        return f"filter:[kind=lpf][cutoff_hz={p['cutoff_hz']}]"
    if cmd == "hpf_sample":
        if not p:
            return "filter:[kind=hpf]"
        return f"filter:[kind=hpf][cutoff_hz={p['cutoff_hz']}]"
    if cmd == "bpf_sample":
        return f"filter:[kind=bpf][low_hz={p.get('low_hz', 300)}][high_hz={p.get('high_hz', 6000)}]"
    if cmd == "normalize_sample":
        return "normalize:[]"
    if cmd == "noisegate_sample":
        parts = [
            f"[threshold_db={p.get('threshold_db', -30)}]",
            f"[attack_ms={p.get('attack_ms', 8)}]",
            f"[release_ms={p.get('release_ms', 140)}]",
        ]
        if "ratio" in p:
            parts.append(f"[ratio={p['ratio']}]")
        return "noisegate:" + "".join(parts)
    if cmd == "stretch_sample":
        return f"timestretch:[stretch_factor={p.get('stretch_factor', 1)}]"
    if cmd == "paul_stretch_sample":
        return (
            f"paulstretch:[stretch_amount={p.get('stretch', 8)}]"
            f"[window_size={p.get('window_size', 2.5)}]"
        )
    if cmd == "pitch_shift_sample":
        return f"pitch:[mode=preserve][semitones={p.get('semitones', 0)}]"
    if cmd == "pitch_shift_transpose_sample":
        return f"pitch:[mode=resample][semitones={p.get('semitones', 0)}]"
    rev_map = {
        "reverb_bedroom_sample": (
            "reverb:[wet_level=0.18][length_sec=0.25][decay_sec=0.18][highpass_hz=120]"
        ),
        "reverb_room_sample": (
            "reverb:[wet_level=0.3][length_sec=0.4][decay_sec=0.3][highpass_hz=100]"
        ),
        "reverb_hall_sample": (
            "reverb:[wet_level=0.35][length_sec=1][decay_sec=0.8][highpass_hz=100]"
        ),
        "reverb_large_sample": (
            "reverb:[wet_level=0.4][length_sec=1.8][decay_sec=1.5][highpass_hz=100]"
        ),
        "reverb_amphitheatre_sample": (
            "reverb:[wet_level=0.48][length_sec=3.5][decay_sec=3][highpass_hz=80]"
        ),
        "reverb_space_sample": (
            "reverb:[wet_level=0.62][length_sec=8][decay_sec=7][highpass_hz=50]"
        ),
        "reverb_void_sample": (
            "reverb:[wet_level=0.74][length_sec=14][decay_sec=12][highpass_hz=38]"
            "[early_reflections=11][tail_diffusion=0.93][early_diffuse=true]"
        ),
        "reverb_distant_sample": (
            "reverb:[wet_level=0.53][length_sec=11.5][decay_sec=10.5][highpass_hz=62]"
            "[early_reflections=3][tail_diffusion=0.95][early_diffuse=false]"
        ),
    }
    if cmd in rev_map:
        return rev_map[cmd]
    if cmd == "compress_mild_sample":
        return "compress:[threshold_db=-14][ratio=2.5][attack_ms=10][release_ms=140]"
    if cmd == "compress_medium_sample":
        return "compress:[threshold_db=-20][ratio=10][attack_ms=3][release_ms=80]"
    if cmd == "compress_heavy_sample":
        return "compress:[threshold_db=-28][ratio=16][attack_ms=1.5][release_ms=60]"
    if cmd == "overdrive_mild_sample":
        return "overdrive:[drive_db=8][highpass_hz=150][lowpass_hz=5500]"
    if cmd == "overdrive_medium_sample":
        return "overdrive:[drive_db=14][highpass_hz=200][lowpass_hz=4000]"
    if cmd == "overdrive_heavy_sample":
        return "overdrive:[drive_db=20][highpass_hz=300][lowpass_hz=3000]"
    if cmd == "distort_mild_sample":
        return "distort:[drive_db=3]"
    if cmd == "distort_medium_sample":
        return "distort:[drive_db=6]"
    if cmd == "distort_heavy_sample":
        return "distort:[drive_db=11]"
    if cmd == "chorus_mild_sample":
        return (
            "chorus:[rate_hz=0.7][depth=0.15][centre_delay_ms=8][feedback=0][mix=0.3]"
        )
    if cmd == "chorus_medium_sample":
        return (
            "chorus:[rate_hz=1][depth=0.25][centre_delay_ms=7][feedback=0][mix=0.5]"
        )
    if cmd == "chorus_heavy_sample":
        return (
            "chorus:[rate_hz=1.4][depth=0.45][centre_delay_ms=6][feedback=0.2][mix=0.75]"
        )
    if cmd == "flanger_mild_sample":
        return (
            "flanger:[rate_hz=0.35][depth=0.2][centre_delay_ms=2.5][feedback=0.15][mix=0.35]"
        )
    if cmd == "flanger_medium_sample":
        return (
            "flanger:[rate_hz=0.5][depth=0.4][centre_delay_ms=2][feedback=0.3][mix=0.5]"
        )
    if cmd == "flanger_heavy_sample":
        return (
            "flanger:[rate_hz=0.9][depth=0.65][centre_delay_ms=1.2][feedback=0.55][mix=0.75]"
        )
    if cmd == "phaser_mild_sample":
        return (
            "phaser:[rate_hz=0.7][depth=0.45][centre_frequency_hz=900]"
            "[feedback=0.25][mix=0.35]"
        )
    if cmd == "phaser_medium_sample":
        return (
            "phaser:[rate_hz=1][depth=0.7][centre_frequency_hz=1000]"
            "[feedback=0.5][mix=0.5]"
        )
    if cmd == "phaser_heavy_sample":
        return (
            "phaser:[rate_hz=1.4][depth=0.95][centre_frequency_hz=1300]"
            "[feedback=0.7][mix=0.75]"
        )
    tok = action.get("token")
    if isinstance(tok, str):
        return tok
    return str(cmd)


def _enriched_actions_and_presets():
    actions_out = []
    presets_out = []
    for a in PROCESSING_ACTIONS:
        row = dict(a)
        row["spec"] = _legacy_action_to_spec(a)
        actions_out.append(row)
        presets_out.append(
            {
                "type": a["type"],
                "label": a["label"],
                "token": a["token"],
                "spec": row["spec"],
            }
        )
    return actions_out, presets_out


_ENRICHED_ACTIONS, PROCESSING_PRESETS = _enriched_actions_and_presets()


def get_processing_actions_payload() -> dict:
    actions, presets = _enriched_actions_and_presets()
    return {
        "types": copy.deepcopy(PROCESSING_TYPES),
        "actions": actions,
        "paramSchemas": copy.deepcopy(_PARAM_SCHEMAS_UI),
        "presets": presets,
    }


def parse_single_processing_spec(spec: str) -> dict:
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("Empty processing_spec")
    actions = parse_post_processing_spec(spec)
    if len(actions) != 1:
        raise ValueError("processing_spec must contain exactly one action")
    return actions[0]


def parse_post_processing_spec(spec: str | None) -> list[dict]:
    if not spec:
        return []
    segments = _split_spec_segments(spec.strip())
    actions: list[dict] = []
    unknown: list[str] = []
    for seg in segments:
        seg_trim = seg.strip()
        if not seg_trim:
            continue
        if _is_bracket_segment(seg_trim):
            try:
                actions.append(action_from_bracket_segment(seg_trim))
            except ValueError as e:
                unknown.append(f"{seg_trim} ({e})")
            continue
        token = seg_trim.lower()
        action = PROCESSING_ACTIONS_BY_TOKEN.get(token)
        if action:
            actions.append(copy.deepcopy(action))
            continue
        legacy_action = _LEGACY_PROCESSING_TOKEN_ACTIONS.get(token)
        if legacy_action:
            actions.append(copy.deepcopy(legacy_action))
            continue
        paul_compact = PAULSTRETCH_COMPACT_TOKEN_RE.match(token)
        if paul_compact:
            actions.append(
                {
                    "token": token,
                    "command": "paul_stretch_sample",
                    "params": {
                        "stretch": float(paul_compact.group(1)),
                        "window_size": float(paul_compact.group(2)),
                    },
                }
            )
            continue
        paul_m = PAUL_STRETCH_TOKEN_RE.match(token)
        if paul_m:
            actions.append(
                {
                    "token": token,
                    "command": "paul_stretch_sample",
                    "params": {
                        "stretch": float(paul_m.group(1)),
                        "window_size": float(paul_m.group(2)),
                    },
                }
            )
            continue
        pr_sm = PITCH_RESAMPLE_TOKEN_RE.match(token)
        if pr_sm:
            actions.append(
                {
                    "token": token,
                    "command": "pitch_shift_transpose_sample",
                    "params": {"semitones": float(pr_sm.group(1))},
                }
            )
            continue
        pr_legacy = PITCH_RAW_LEGACY_TOKEN_RE.match(token)
        if pr_legacy:
            actions.append(
                {
                    "token": token,
                    "command": "pitch_shift_transpose_sample",
                    "params": {"semitones": float(pr_legacy.group(1))},
                }
            )
            continue
        unknown.append(token)
    if unknown:
        raise ValueError(
            "Unknown post-processing action(s): "
            + ", ".join(unknown)
            + ". Use legacy tokens or bracket syntax type:[k=v][...]."
        )
    return actions


def apply_processing_command(file_path: str, command: str, params: dict | None = None) -> None:
    p = params or {}
    match command:
        case "trim_sample_start":
            trim_sample_start(file_path, p["seconds"])
        case "trim_sample_end":
            trim_sample_end(file_path, p["seconds"])
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
        case "pitch_shift_transpose_sample":
            apply_transpose_pitch_by_resampling_inplace(
                file_path, p.get("semitones", 0)
            )
        case "paul_stretch_sample":
            apply_paulstretch_to_sample(
                file_path,
                stretch=p.get("stretch", 8.0),
                window_size=p.get("window_size", 2.5),
            )
        case "reverb_sample":
            kw = {
                "wet_level": float(p.get("wet_level", 0.35)),
                "reverb_length_sec": float(p.get("length_sec", p.get("reverb_length_sec", 0.7))),
                "decay_sec": float(p.get("decay_sec", 0.5)),
                "reverb_highpass_hz": float(p.get("highpass_hz", p.get("reverb_highpass_hz", 100))),
            }
            if "early_reflections" in p:
                kw["early_reflections"] = int(p["early_reflections"])
            if "tail_diffusion" in p:
                kw["tail_diffusion"] = float(p["tail_diffusion"])
            if "early_diffuse" in p:
                kw["early_diffuse"] = bool(p["early_diffuse"])
            apply_reverb_to_sample(file_path, **kw)
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
        case "reverb_void_sample":
            apply_reverb_void_to_sample(file_path)
        case "reverb_distant_sample":
            apply_reverb_distant_to_sample(file_path)
        case "compress_sample":
            apply_compress_to_sample(
                file_path,
                threshold_db=float(p.get("threshold_db", -20)),
                ratio=float(p.get("ratio", 10)),
                attack_ms=float(p.get("attack_ms", 3)),
                release_ms=float(p.get("release_ms", 80)),
            )
        case "compress_mild_sample":
            apply_compress_mild_to_sample(file_path)
        case "compress_medium_sample":
            apply_compress_medium_to_sample(file_path)
        case "compress_heavy_sample":
            apply_compress_heavy_to_sample(file_path)
        case "overdrive_mids_sample":
            apply_overdrive_mids_to_sample(
                file_path,
                drive_db=float(p.get("drive_db", 14)),
                highpass_hz=float(p.get("highpass_hz", 200)),
                lowpass_hz=float(p.get("lowpass_hz", 4000)),
            )
        case "overdrive_mild_sample":
            apply_overdrive_mild_to_sample(file_path)
        case "overdrive_medium_sample":
            apply_overdrive_medium_to_sample(file_path)
        case "overdrive_heavy_sample":
            apply_overdrive_heavy_to_sample(file_path)
        case "distort_sample":
            apply_distortion_to_sample(file_path, drive_db=float(p.get("drive_db", 6)))
        case "distort_mild_sample":
            apply_distortion_mild_to_sample(file_path)
        case "distort_medium_sample":
            apply_distortion_medium_to_sample(file_path)
        case "distort_heavy_sample":
            apply_distortion_heavy_to_sample(file_path)
        case "chorus_sample":
            apply_chorus_to_sample(
                file_path,
                rate_hz=float(p.get("rate_hz", 1.0)),
                depth=float(p.get("depth", 0.25)),
                centre_delay_ms=float(p.get("centre_delay_ms", 7.0)),
                feedback=float(p.get("feedback", 0.0)),
                mix=float(p.get("mix", 0.5)),
            )
        case "chorus_mild_sample":
            apply_chorus_mild_to_sample(file_path)
        case "chorus_medium_sample":
            apply_chorus_medium_to_sample(file_path)
        case "chorus_heavy_sample":
            apply_chorus_heavy_to_sample(file_path)
        case "flanger_sample":
            apply_flanger_to_sample(
                file_path,
                rate_hz=float(p.get("rate_hz", 0.5)),
                depth=float(p.get("depth", 0.4)),
                centre_delay_ms=float(p.get("centre_delay_ms", 2.0)),
                feedback=float(p.get("feedback", 0.3)),
                mix=float(p.get("mix", 0.5)),
            )
        case "flanger_mild_sample":
            apply_flanger_mild_to_sample(file_path)
        case "flanger_medium_sample":
            apply_flanger_medium_to_sample(file_path)
        case "flanger_heavy_sample":
            apply_flanger_heavy_to_sample(file_path)
        case "phaser_sample":
            apply_phaser_to_sample(
                file_path,
                rate_hz=float(p.get("rate_hz", 1.0)),
                depth=float(p.get("depth", 0.7)),
                centre_frequency_hz=float(p.get("centre_frequency_hz", 1000)),
                feedback=float(p.get("feedback", 0.5)),
                mix=float(p.get("mix", 0.5)),
            )
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
        case "normalize_sample":
            normalize_sample(file_path)
        case "noisegate_sample":
            apply_noise_gate_to_sample(
                file_path,
                threshold_db=float(p.get("threshold_db", -30)),
                ratio=float(p.get("ratio", 8.0)),
                attack_ms=float(p.get("attack_ms", 8)),
                release_ms=float(p.get("release_ms", 140)),
            )
        case "eq_lows_sample":
            apply_eq_lows_to_sample(file_path, p.get("db", 0))
        case "eq_mids_sample":
            apply_eq_mids_to_sample(file_path, p.get("db", 0))
        case "eq_highs_sample":
            apply_eq_highs_to_sample(file_path, p.get("db", 0))
        case _:
            raise ValueError(f"Unsupported processing command: {command}")


def actions_without_normalize(actions: list[dict]) -> list[dict]:
    """Drop user-supplied normalization; we always finalize exports with normalization once."""
    return [a for a in actions if (a.get("command") or "") != "normalize_sample"]


def apply_post_processing_actions(
    file_path: str,
    actions: list[dict],
    *,
    on_before_chain_step=None,
    on_before_finalize_normalize=None,
) -> None:
    if not actions:
        return
    chain = actions_without_normalize(actions)
    for i, action in enumerate(chain, start=1):
        if on_before_chain_step is not None:
            on_before_chain_step(i, len(chain), action)
        apply_processing_command(
            file_path,
            action.get("command", ""),
            action.get("params", {}),
        )
    if on_before_finalize_normalize is not None:
        on_before_finalize_normalize()
    normalize_sample(file_path)
