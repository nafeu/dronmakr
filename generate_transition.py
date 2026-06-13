"""Transition sound generation (sweeps, washes, crashes) using offline DSP."""

import random
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, lfilter, lfilter_zi

from pedalboard import Chorus, Delay, Pedalboard, Phaser

import dsp
from settings import get_setting, parse_escaped_csv

SAMPLE_RATE = dsp.SAMPLE_RATE
BLOCK_SIZE = 256
OUTPUT_TARGET_PEAK = (
    0.9  # normalize generated outputs to this peak for consistent levels
)


def _normalize_audio(
    audio: np.ndarray, target_peak: float = OUTPUT_TARGET_PEAK
) -> np.ndarray:
    """Scale audio so peak equals target_peak (0.9 by default for headroom)."""
    peak = np.max(np.abs(audio)) + 1e-12
    return (audio * (target_peak / peak)).astype(np.float32)


# Default ranges for randomization (when params not passed)
CUTOFF_LOW_RANGE = (300, 700)
CUTOFF_HIGH_RANGE = (10000, 18000)
PEAK_POS_RANGE = (0.25, 0.75)
NOISE_LEVEL_RANGE = (0.45, 0.8)
FILTER_ORDERS = (2, 4, 6)
FILTER_SWEEP_TYPES: tuple[str, ...] = ("lpf", "hpf", "bpf", "bsf", "none")
FILTER_LFO_WIDTH_RANGE = (0.0, 0.25)
FILTER_LFO_RATE_MIN_RANGE = (0.1, 1.5)
FILTER_LFO_RATE_PEAK_RANGE = (5.0, 25.0)
BUILD_SHAPES: tuple[str, ...] = (
    "easeInSine",
    "easeOutSine",
    "easeInQuad",
    "easeOutQuad",
    "easeInCubic",
    "easeOutCubic",
    "easeInQuart",
    "easeOutQuart",
    "easeInQuint",
    "easeOutQuint",
    "easeInExpo",
    "easeOutExpo",
    "easeInCirc",
    "easeOutCirc",
    "easeInBack",
    "easeOutBack",
)

_BUILD_SHAPE_ALIASES: dict[str, str] = {
    "ease_in": "easeInCubic",
    "easein": "easeInCubic",
    "ease_out": "easeOutCubic",
    "easeout": "easeOutCubic",
    "ease_in_out": "easeInCubic",
    "easeinout": "easeInCubic",
    "linear": "linear",
}

_EASE_BACK_C1 = 1.70158


def normalize_build_shape(value: str | None) -> str | None:
    """Map CLI / legacy curve names to canonical build shapes. None => random."""
    if value is None or _is_random(value):
        return None
    raw = value.strip()
    if not raw:
        return None
    lowered_key = raw.lower().replace("-", "").replace("_", "")
    alias_key = raw.lower().replace("-", "_")
    if alias_key in _BUILD_SHAPE_ALIASES:
        return _BUILD_SHAPE_ALIASES[alias_key]
    for alias, target in _BUILD_SHAPE_ALIASES.items():
        if alias.replace("_", "") == lowered_key:
            return target
    if raw.startswith("easeInOut") and len(raw) > 9:
        in_candidate = "easeIn" + raw[9:]
        canonical = {shape.lower(): shape for shape in BUILD_SHAPES}
        mapped = canonical.get(in_candidate.lower())
        if mapped:
            return mapped
    canonical = {shape.lower(): shape for shape in BUILD_SHAPES}
    direct = canonical.get(raw.lower())
    if direct:
        return direct
    camelish = canonical.get(lowered_key)
    if camelish:
        return camelish
    return None


def _resolve_build_shape(value: str | None) -> str:
    normalized = normalize_build_shape(value)
    if normalized == "linear":
        return "linear"
    if normalized in BUILD_SHAPES:
        return normalized
    return random.choice(BUILD_SHAPES)
NOISE_TYPES: tuple[Literal["white", "pink", "brown", "blue"], ...] = (
    "white",
    "pink",
    "brown",
    "blue",
)
VOICE_TYPES: tuple[Literal["noise", "sine", "saw", "tri", "square"], ...] = (
    "noise",
    "sine",
    "saw",
    "tri",
    "square",
)
SWEEP_VOICE_CHOICES: tuple[str, ...] = (
    "whitenoise",
    "pinknoise",
    "brownnoise",
    "bluenoise",
    "sine",
    "saw",
    "tri",
    "square",
)
_NOISE_VOICE_MAP: dict[str, Literal["white", "pink", "brown", "blue"]] = {
    "whitenoise": "white",
    "pinknoise": "pink",
    "brownnoise": "brown",
    "bluenoise": "blue",
}
GAIN_MIN_RANGE = (0.0, 0.2)
GAIN_MAX_RANGE = (0.75, 1.0)
OSC_FREQ_LOW_RANGE = (55, 220)  # A1 to A3
OSC_FREQ_HIGH_RANGE = (880, 4000)  # A5 to ~B7 (used only when freq_high explicit)
OSC_OCTAVE_SPAN_RANGE = (
    3.0,
    5.0,
)  # octaves for sweep when randomizing (3–5 octaves), capped by C4 as top
OSC_LEVEL_RANGE = (0.3, 0.8)
PULSE_WIDTH_RANGE = (0.2, 0.8)
PWM_SWEEP_RANGE = (0.15, 0.5)  # stronger PWM modulation by default
DETUNE_CENTS_RANGE = (8, 35)  # more noticeable detune
DETUNE_MIX_RANGE = (0.4, 0.9)  # detuned layer more prominent
RESONANCE_Q_RANGE = (1.0, 10.0)  # more pronounced filter peaks
TREMOLO_DEPTH_RANGE = (0.7, 1.0)  # tremolo is usually strong
TREMOLO_RATE_MIN_RANGE = (0.05, 0.5)
TREMOLO_RATE_MAX_RANGE = (10.0, 25.0)

# Modulation effect ranges (phaser, chorus, flanger) for randomization
PHASER_RATE_RANGE = (0.2, 2.0)
PHASER_DEPTH_RANGE = (0.4, 0.9)
PHASER_CENTRE_RANGE = (500, 2500)
PHASER_FEEDBACK_RANGE = (0.2, 0.6)
PHASER_MIX_RANGE = (0.3, 0.7)
CHORUS_RATE_RANGE = (0.3, 1.5)
CHORUS_DEPTH_RANGE = (0.15, 0.4)
CHORUS_DELAY_RANGE = (5.0, 10.0)
CHORUS_MIX_RANGE = (0.3, 0.6)
FLANGER_RATE_RANGE = (0.2, 0.8)
FLANGER_DEPTH_RANGE = (0.3, 0.5)
FLANGER_DELAY_RANGE = (1.0, 3.0)
FLANGER_FEEDBACK_RANGE = (0.2, 0.5)
FLANGER_MIX_RANGE = (0.4, 0.6)


def _parse_param_string(s: str | None) -> dict[str, str]:
    """Parse 'key1:val1;key2:val2' into dict. Empty or None returns {}."""
    if not s or not s.strip():
        return {}
    out: dict[str, str] = {}
    for part in s.split(";"):
        part = part.strip()
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip().lower()] = v.strip()
        elif part:
            out[part.lower()] = ""
    return out


def _is_random(val: str | None) -> bool:
    """Value is _ or empty means use random."""
    return val is None or val == "" or val.lower() == "_"


def _resolve_float(
    parsed: dict[str, str],
    key: str,
    default_range: tuple[float, float],
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    val = parsed.get(key)
    if _is_random(val):
        v = random.uniform(*default_range)
    else:
        try:
            v = float(val)
        except (TypeError, ValueError):
            v = random.uniform(*default_range)
    if min_val is not None:
        v = max(min_val, v)
    if max_val is not None:
        v = min(max_val, v)
    return v


def _resolve_int(
    parsed: dict[str, str],
    key: str,
    default_range: tuple[int, int],
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    val = parsed.get(key)
    if _is_random(val):
        v = random.randint(*default_range)
    else:
        try:
            v = int(float(val))
        except (TypeError, ValueError):
            v = random.randint(*default_range)
    if min_val is not None:
        v = max(min_val, v)
    if max_val is not None:
        v = min(max_val, v)
    return v


def _resolve_choice(parsed: dict[str, str], key: str, choices: tuple[str, ...]) -> str:
    val = parsed.get(key)
    if _is_random(val) or (val and val.lower() not in {c.lower() for c in choices}):
        return random.choice(choices)
    return next(c for c in choices if c.lower() == val.lower())


def _resolve_optional_float(
    value: float | None,
    default_range: tuple[float, float],
    min_val: float | None = None,
    max_val: float | None = None,
) -> float:
    if value is None:
        v = random.uniform(*default_range)
    else:
        v = float(value)
    if min_val is not None:
        v = max(min_val, v)
    if max_val is not None:
        v = min(max_val, v)
    return v


def _resolve_optional_int(
    value: int | None,
    default_range: tuple[int, int],
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    if value is None:
        v = random.randint(*default_range)
    else:
        v = int(value)
    if min_val is not None:
        v = max(min_val, v)
    if max_val is not None:
        v = min(max_val, v)
    return v


def _resolve_optional_choice(
    value: str | None, choices: tuple[str, ...]
) -> str:
    if value is None or _is_random(value):
        return random.choice(choices)
    lowered = {c.lower(): c for c in choices}
    return lowered.get(value.lower(), random.choice(choices))


def _parse_disable_set(disable: str | None) -> set[str]:
    disabled: set[str] = set()
    if not disable:
        return disabled
    for d in re.split(r"[\s,]+", disable.strip().lower()):
        d = d.strip()
        if d == "fx":
            disabled.update(("filter", "tremolo", "phaser", "chorus", "flanger", "gain"))
        elif d in ("filter", "tremolo", "phaser", "chorus", "flanger", "gain"):
            disabled.add(d)
    return disabled


def _resolve_effect_enabled(
    explicit: bool | None,
    disabled: set[str],
    name: str,
    *,
    default_when_unset: bool,
) -> bool:
    if name in disabled:
        return False
    if explicit is not None:
        return explicit
    return default_when_unset


def _resolve_osc_voice_params(voice: Literal["sine", "saw", "tri", "square"]) -> dict[str, Any]:
    """Randomize oscillator sweep params for a fixed waveform voice."""
    C5_HZ = 523.25
    freq_high = C5_HZ
    octaves = random.uniform(*OSC_OCTAVE_SPAN_RANGE)
    min_low_allowed = max(20.0, freq_high / (2.0**5))
    freq_low = max(freq_high / (2.0**octaves), min_low_allowed)
    if freq_low >= freq_high:
        freq_low = max(20.0, freq_high / 2.0)
    max_span_low = max(freq_high / (2.0**5), 20.0)
    if freq_low < max_span_low:
        freq_low = max_span_low
    return {
        "voice": voice,
        "osc_freq_low": freq_low,
        "osc_freq_high": freq_high,
        "osc_level": random.uniform(*OSC_LEVEL_RANGE),
        "pulse_width": random.uniform(*PULSE_WIDTH_RANGE),
        "pwm_sweep": random.uniform(*PWM_SWEEP_RANGE),
        "detune_cents": random.uniform(*DETUNE_CENTS_RANGE),
        "detune_mix": random.uniform(*DETUNE_MIX_RANGE),
        "resonance": random.uniform(0, 1),
    }


def _resolve_sweep_voice_params(
    voice: str | None,
    pitch_min: float | None = None,
    pitch_max: float | None = None,
) -> dict[str, Any]:
    """Resolve --voice (whitenoise, pinknoise, …, sine, saw, tri, square)."""
    picked = _resolve_optional_choice(voice, SWEEP_VOICE_CHOICES)
    if picked in _NOISE_VOICE_MAP:
        return {
            "voice": "noise",
            "sweep_voice": picked,
            "noise_type": _NOISE_VOICE_MAP[picked],
            "noise_level": random.uniform(*NOISE_LEVEL_RANGE),
        }
    osc_voice = picked  # type: ignore[assignment]
    osc = _resolve_osc_voice_params(osc_voice)
    osc["sweep_voice"] = picked
    if pitch_min is not None:
        osc["osc_freq_low"] = max(20.0, min(20000.0, float(pitch_min)))
    if pitch_max is not None:
        osc["osc_freq_high"] = max(40.0, min(20000.0, float(pitch_max)))
    osc["osc_freq_high"] = max(float(osc["osc_freq_high"]), float(osc["osc_freq_low"]) + 50.0)
    if pitch_min is not None and float(osc["osc_freq_low"]) > float(osc["osc_freq_high"]) - 50.0:
        osc["osc_freq_low"] = max(20.0, float(osc["osc_freq_high"]) - 50.0)
    return osc


def _resolve_phaser_params_from_explicit(
    *,
    rate_min: float | None,
    rate_max: float | None,
    depth: float | None,
    centre: float | None,
    feedback: float | None,
    mix: float | None,
) -> dict[str, float]:
    resolved_rate_min = _resolve_optional_float(rate_min, PHASER_RATE_RANGE, 0.1, 10)
    resolved_rate_max = _resolve_optional_float(rate_max, PHASER_RATE_RANGE, 0.1, 10)
    resolved_rate_max = max(resolved_rate_max, resolved_rate_min + 0.05)
    return {
        "rate_min_hz": resolved_rate_min,
        "rate_max_hz": resolved_rate_max,
        "depth": _resolve_optional_float(depth, PHASER_DEPTH_RANGE, 0, 1),
        "centre_frequency_hz": _resolve_optional_float(
            centre, PHASER_CENTRE_RANGE, 100, 10000
        ),
        "feedback": _resolve_optional_float(feedback, PHASER_FEEDBACK_RANGE, 0, 1),
        "mix": _resolve_optional_float(mix, PHASER_MIX_RANGE, 0, 1),
    }


def _resolve_chorus_params_from_explicit(
    *,
    rate_min: float | None,
    rate_max: float | None,
    depth: float | None,
    delay: float | None,
    mix: float | None,
) -> dict[str, float]:
    resolved_rate_min = _resolve_optional_float(rate_min, CHORUS_RATE_RANGE, 0.1, 10)
    resolved_rate_max = _resolve_optional_float(rate_max, CHORUS_RATE_RANGE, 0.1, 10)
    resolved_rate_max = max(resolved_rate_max, resolved_rate_min + 0.05)
    return {
        "rate_min_hz": resolved_rate_min,
        "rate_max_hz": resolved_rate_max,
        "depth": _resolve_optional_float(depth, CHORUS_DEPTH_RANGE, 0, 1),
        "centre_delay_ms": _resolve_optional_float(delay, CHORUS_DELAY_RANGE, 1, 20),
        "feedback": 0.0,
        "mix": _resolve_optional_float(mix, CHORUS_MIX_RANGE, 0, 1),
    }


def _resolve_flanger_params_from_explicit(
    *,
    rate_min: float | None,
    rate_max: float | None,
    depth: float | None,
    delay: float | None,
    feedback: float | None,
    mix: float | None,
) -> dict[str, float]:
    resolved_rate_min = _resolve_optional_float(rate_min, FLANGER_RATE_RANGE, 0.1, 5)
    resolved_rate_max = _resolve_optional_float(rate_max, FLANGER_RATE_RANGE, 0.1, 5)
    resolved_rate_max = max(resolved_rate_max, resolved_rate_min + 0.05)
    return {
        "rate_min_hz": resolved_rate_min,
        "rate_max_hz": resolved_rate_max,
        "depth": _resolve_optional_float(depth, FLANGER_DEPTH_RANGE, 0, 1),
        "centre_delay_ms": _resolve_optional_float(delay, FLANGER_DELAY_RANGE, 0.5, 10),
        "feedback": _resolve_optional_float(feedback, FLANGER_FEEDBACK_RANGE, 0, 0.8),
        "mix": _resolve_optional_float(mix, FLANGER_MIX_RANGE, 0, 1),
    }


def _resolve_phaser_params(parsed: dict[str, str]) -> dict[str, float]:
    rate = _resolve_float(parsed, "rate_hz", PHASER_RATE_RANGE, 0.1, 10)
    return _resolve_phaser_params_from_explicit(
        rate_min=rate,
        rate_max=rate,
        depth=_resolve_float(parsed, "depth", PHASER_DEPTH_RANGE, 0, 1),
        centre=_resolve_float(parsed, "centre_frequency_hz", PHASER_CENTRE_RANGE, 100, 10000),
        feedback=_resolve_float(parsed, "feedback", PHASER_FEEDBACK_RANGE, 0, 1),
        mix=_resolve_float(parsed, "mix", PHASER_MIX_RANGE, 0, 1),
    )


def _resolve_chorus_params(parsed: dict[str, str]) -> dict[str, float]:
    rate = _resolve_float(parsed, "rate_hz", CHORUS_RATE_RANGE, 0.1, 10)
    return _resolve_chorus_params_from_explicit(
        rate_min=rate,
        rate_max=rate,
        depth=_resolve_float(parsed, "depth", CHORUS_DEPTH_RANGE, 0, 1),
        delay=_resolve_float(parsed, "centre_delay_ms", CHORUS_DELAY_RANGE, 1, 20),
        mix=_resolve_float(parsed, "mix", CHORUS_MIX_RANGE, 0, 1),
    )


def _resolve_flanger_params(parsed: dict[str, str]) -> dict[str, float]:
    rate = _resolve_float(parsed, "rate_hz", FLANGER_RATE_RANGE, 0.1, 5)
    return _resolve_flanger_params_from_explicit(
        rate_min=rate,
        rate_max=rate,
        depth=_resolve_float(parsed, "depth", FLANGER_DEPTH_RANGE, 0, 1),
        delay=_resolve_float(parsed, "centre_delay_ms", FLANGER_DELAY_RANGE, 0.5, 10),
        feedback=_resolve_float(parsed, "feedback", FLANGER_FEEDBACK_RANGE, 0, 0.8),
        mix=_resolve_float(parsed, "mix", FLANGER_MIX_RANGE, 0, 1),
    )


def parse_sweep_config(
    *,
    voice: str | None = None,
    pitch_min: float | None = None,
    pitch_max: float | None = None,
    curve_shape: str | None = None,
    curve_peak_position: float | None = None,
    filter_enabled: bool | None = None,
    filter_type: str | None = None,
    filter_cutoff_low: int | None = None,
    filter_cutoff_high: int | None = None,
    tremolo_enabled: bool | None = None,
    tremolo_rate_min: float | None = None,
    tremolo_rate_max: float | None = None,
    tremolo_depth: float | None = None,
    phaser_enabled: bool | None = None,
    phaser_rate_min: float | None = None,
    phaser_rate_max: float | None = None,
    phaser_depth: float | None = None,
    phaser_centre: float | None = None,
    phaser_feedback: float | None = None,
    phaser_mix: float | None = None,
    chorus_enabled: bool | None = None,
    chorus_rate_min: float | None = None,
    chorus_rate_max: float | None = None,
    chorus_depth: float | None = None,
    chorus_delay: float | None = None,
    chorus_mix: float | None = None,
    flanger_enabled: bool | None = None,
    flanger_rate_min: float | None = None,
    flanger_rate_max: float | None = None,
    flanger_depth: float | None = None,
    flanger_delay: float | None = None,
    flanger_feedback: float | None = None,
    flanger_mix: float | None = None,
    gain_enabled: bool | None = None,
    gain_min: float | None = None,
    gain_max: float | None = None,
    disable: str | None = None,
) -> dict[str, Any]:
    """Resolve sweep config from explicit CLI / API options (omit for random)."""
    disabled = _parse_disable_set(disable)
    resolved_pitch_min = (
        _resolve_optional_float(pitch_min, OSC_FREQ_LOW_RANGE, 20, 20000)
        if pitch_min is not None
        else None
    )
    resolved_pitch_max = (
        _resolve_optional_float(pitch_max, OSC_FREQ_HIGH_RANGE, 40, 20000)
        if pitch_max is not None
        else None
    )
    if resolved_pitch_min is not None and resolved_pitch_max is not None:
        if resolved_pitch_min > resolved_pitch_max - 50.0:
            resolved_pitch_min = max(20.0, resolved_pitch_max - 50.0)
    voice_params = _resolve_sweep_voice_params(
        voice,
        pitch_min=resolved_pitch_min,
        pitch_max=resolved_pitch_max,
    )

    resolved_tremolo_rate_min = _resolve_optional_float(
        tremolo_rate_min, TREMOLO_RATE_MIN_RANGE, 0.05, 20
    )
    resolved_tremolo_rate_max = _resolve_optional_float(
        tremolo_rate_max, TREMOLO_RATE_MAX_RANGE, 2, 50
    )
    resolved_tremolo_rate_max = max(
        resolved_tremolo_rate_max, resolved_tremolo_rate_min + 1.0
    )

    use_filter = _resolve_effect_enabled(
        filter_enabled, disabled, "filter", default_when_unset=True
    )
    use_tremolo = _resolve_effect_enabled(
        tremolo_enabled, disabled, "tremolo", default_when_unset=True
    )
    phaser_explicit = any(
        v is not None
        for v in (
            phaser_rate_min,
            phaser_rate_max,
            phaser_depth,
            phaser_centre,
            phaser_feedback,
            phaser_mix,
        )
    )
    chorus_explicit = any(
        v is not None
        for v in (chorus_rate_min, chorus_rate_max, chorus_depth, chorus_delay, chorus_mix)
    )
    flanger_explicit = any(
        v is not None
        for v in (
            flanger_rate_min,
            flanger_rate_max,
            flanger_depth,
            flanger_delay,
            flanger_feedback,
            flanger_mix,
        )
    )
    use_phaser = _resolve_effect_enabled(
        phaser_enabled,
        disabled,
        "phaser",
        default_when_unset=phaser_explicit or random.random() < 0.5,
    )
    use_chorus = _resolve_effect_enabled(
        chorus_enabled,
        disabled,
        "chorus",
        default_when_unset=chorus_explicit or random.random() < 0.5,
    )
    use_flanger = _resolve_effect_enabled(
        flanger_enabled,
        disabled,
        "flanger",
        default_when_unset=flanger_explicit or random.random() < 0.5,
    )
    use_gain = _resolve_effect_enabled(
        gain_enabled, disabled, "gain", default_when_unset=True
    )

    cutoff_low = _resolve_optional_int(filter_cutoff_low, CUTOFF_LOW_RANGE, 50, 5000)
    cutoff_high = _resolve_optional_int(filter_cutoff_high, CUTOFF_HIGH_RANGE, 1000, 20000)
    cutoff_high = max(cutoff_high, cutoff_low + 500)

    return {
        **voice_params,
        "build_shape": _resolve_build_shape(curve_shape),
        "peak_pos": _resolve_optional_float(
            curve_peak_position, PEAK_POS_RANGE, 0.05, 1.0
        ),
        "filter_enabled": use_filter,
        "cutoff_low": cutoff_low,
        "cutoff_high": cutoff_high,
        "filter_sweep_type": (
            "none"
            if not use_filter
            else _resolve_optional_choice(filter_type, FILTER_SWEEP_TYPES)
        ),
        "filter_lfo_width": 0.0,
        "filter_lfo_rate_min": _resolve_optional_float(
            None, FILTER_LFO_RATE_MIN_RANGE, 0.05, 10
        ),
        "filter_lfo_rate_peak": _resolve_optional_float(
            None, FILTER_LFO_RATE_PEAK_RANGE, 2, 50
        ),
        "tremolo_enabled": use_tremolo,
        "tremolo_depth": (
            0.0
            if not use_tremolo
            else _resolve_optional_float(tremolo_depth, TREMOLO_DEPTH_RANGE, 0, 1)
        ),
        "tremolo_rate_min": resolved_tremolo_rate_min,
        "tremolo_rate_max": resolved_tremolo_rate_max,
        "phaser_enabled": use_phaser,
        "phaser_params": (
            _resolve_phaser_params_from_explicit(
                rate_min=phaser_rate_min,
                rate_max=phaser_rate_max,
                depth=phaser_depth,
                centre=phaser_centre,
                feedback=phaser_feedback,
                mix=phaser_mix,
            )
            if use_phaser
            else {}
        ),
        "chorus_enabled": use_chorus,
        "chorus_params": (
            _resolve_chorus_params_from_explicit(
                rate_min=chorus_rate_min,
                rate_max=chorus_rate_max,
                depth=chorus_depth,
                delay=chorus_delay,
                mix=chorus_mix,
            )
            if use_chorus
            else {}
        ),
        "flanger_enabled": use_flanger,
        "flanger_params": (
            _resolve_flanger_params_from_explicit(
                rate_min=flanger_rate_min,
                rate_max=flanger_rate_max,
                depth=flanger_depth,
                delay=flanger_delay,
                feedback=flanger_feedback,
                mix=flanger_mix,
            )
            if use_flanger
            else {}
        ),
        "gain_enabled": use_gain,
        "gain_min": _resolve_optional_float(gain_min, GAIN_MIN_RANGE, 0, 1),
        "gain_max": _resolve_optional_float(gain_max, GAIN_MAX_RANGE, 0, 1),
        "pitch_min_hz": resolved_pitch_min,
        "pitch_max_hz": resolved_pitch_max,
    }


def _generate_noise(
    num_samples: int,
    noise_type: Literal["white", "pink", "brown", "blue"],
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate colored noise. white=flat, pink=1/f, brown=1/f², blue=rising with freq."""
    white = rng.standard_normal(num_samples).astype(np.float64)

    if noise_type == "white":
        return white

    if noise_type == "brown":
        # Integrate white → 1/f² (brown/red noise), rumbly and smooth
        brown = np.cumsum(white)
        brown = brown - np.linspace(brown[0], brown[-1], num_samples)
        return brown / (np.max(np.abs(brown)) + 1e-9) * (np.std(white) or 1.0)

    if noise_type == "pink":
        # FFT method: magnitude ~ 1/sqrt(f) for 1/f spectrum
        n = num_samples
        freqs = np.fft.rfftfreq(n)
        freqs[0] = freqs[1]  # avoid div by zero at DC
        magnitude = 1.0 / np.sqrt(np.abs(freqs))
        phase = rng.uniform(0, 2 * np.pi, len(freqs))
        spectrum = magnitude * np.exp(1j * phase)
        pink = np.fft.irfft(spectrum, n=n).astype(np.float64)
        return pink / (np.std(pink) + 1e-9) * (np.std(white) or 1.0)

    if noise_type == "blue":
        # Differentiate white → power ∝ f² (blue noise), bright and hissy
        blue = np.diff(white, prepend=0.0)
        return blue / (np.std(blue) + 1e-9) * (np.std(white) or 1.0)

    return white


def _oscillator_sample(
    phase: np.ndarray,
    voice: Literal["sine", "saw", "tri", "square"],
    pulse_width: np.ndarray | float,
) -> np.ndarray:
    """Single oscillator output. pulse_width: scalar or per-sample array."""
    if np.isscalar(pulse_width):
        pw = np.full(phase.shape, float(pulse_width))
    else:
        pw = np.asarray(pulse_width, dtype=np.float64)
        if pw.size != phase.size:
            pw = np.broadcast_to(pw.flat[0], phase.shape)

    if voice == "sine":
        return np.sin(phase)
    if voice == "saw":
        return 1.0 - 2.0 * np.mod(phase, 2 * np.pi) / (2 * np.pi)
    if voice == "tri":
        p = np.mod(phase, 2 * np.pi)
        tri = np.where(p < np.pi, p / np.pi, 2 - p / np.pi)
        return 2.0 * tri - 1.0
    if voice == "square":
        p = np.mod(phase, 2 * np.pi)
        thresh = np.clip(pw, 0.05, 0.95) * 2 * np.pi
        return np.where(p < thresh, 1.0, -1.0)
    return np.sin(phase)


def _generate_oscillator_sweep(
    num_samples: int,
    freq_sweep_hz: np.ndarray,
    voice: Literal["sine", "saw", "tri", "square"],
    level: float = 0.5,
    pulse_width: float = 0.5,
    pwm_sweep: float = 0.0,
    detune_cents: float = 0.0,
    detune_mix: float = 0.0,
    cutoff_frac: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generate oscillator output with pitch sweep. Optional: pwm_sweep (pulse width modulation),
    detune (unison layer), both following cutoff_frac (build curve) for accelerating modulation.
    """
    phase_inc = 2 * np.pi * freq_sweep_hz / SAMPLE_RATE
    phase = np.cumsum(phase_inc)

    # PWM sweep: pulse width varies with build curve (only for square)
    if voice == "square" and pwm_sweep > 0 and cutoff_frac is not None:
        pw = np.clip(pulse_width + pwm_sweep * (cutoff_frac - 0.5) * 2, 0.1, 0.9)
    else:
        pw = pulse_width

    main = _oscillator_sample(phase, voice, pw)

    # Detune: second oscillator, mix amount follows build curve (thicker toward peak)
    if detune_cents > 0 and detune_mix > 0 and cutoff_frac is not None:
        detune_ratio = 2 ** (detune_cents / 1200)
        phase_det = np.cumsum(phase_inc * detune_ratio)
        detuned = _oscillator_sample(phase_det, voice, pw)
        mix = detune_mix * np.clip(cutoff_frac, 0, 1)
        main = main * (1 - mix) + detuned * mix

    return (main * level).astype(np.float64)


def _apply_build_shape(t_norm: np.ndarray, shape: str) -> np.ndarray:
    """Map normalized 0..1 through an easing curve (f(0)=0, f(1)=1)."""
    t = np.clip(t_norm.astype(np.float64, copy=False), 0.0, 1.0)
    canonical = normalize_build_shape(shape) or shape

    if canonical == "linear":
        return t
    if canonical == "easeInSine":
        return 1.0 - np.cos((t * np.pi) / 2.0)
    if canonical == "easeOutSine":
        return np.sin((t * np.pi) / 2.0)
    if canonical == "easeInQuad":
        return t * t
    if canonical == "easeOutQuad":
        return 1.0 - np.power(1.0 - t, 2.0)
    if canonical == "easeInCubic":
        return t * t * t
    if canonical == "easeOutCubic":
        return 1.0 - np.power(1.0 - t, 3.0)
    if canonical == "easeInQuart":
        return np.power(t, 4.0)
    if canonical == "easeOutQuart":
        return 1.0 - np.power(1.0 - t, 4.0)
    if canonical == "easeInQuint":
        return np.power(t, 5.0)
    if canonical == "easeOutQuint":
        return 1.0 - np.power(1.0 - t, 5.0)
    if canonical == "easeInExpo":
        return np.where(t == 0.0, 0.0, np.power(2.0, 10.0 * t - 10.0))
    if canonical == "easeOutExpo":
        return np.where(t == 1.0, 1.0, 1.0 - np.power(2.0, -10.0 * t))
    if canonical == "easeInCirc":
        return 1.0 - np.sqrt(1.0 - np.power(t, 2.0))
    if canonical == "easeOutCirc":
        return np.sqrt(1.0 - np.power(t - 1.0, 2.0))
    if canonical == "easeInBack":
        return (_EASE_BACK_C1 + 1.0) * np.power(t, 3.0) - _EASE_BACK_C1 * np.power(t, 2.0)
    if canonical == "easeOutBack":
        return (
            1.0
            + (_EASE_BACK_C1 + 1.0) * np.power(t - 1.0, 3.0)
            + _EASE_BACK_C1 * np.power(t - 1.0, 2.0)
        )
    return t


def _riser_build_curve(
    normalized_t: np.ndarray,
    peak_pos: float,
    build_shape: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep envelope: one easing curve from 0 to 1 on the rise, then the same curve
    reversed in time after the peak (not a different ease-out). At peak_pos=0.5 the
    waveform is perfectly time-symmetric.
    """
    x = np.clip(normalized_t, 0, 1)
    build = x <= peak_pos
    release = ~build

    t_norm = np.empty_like(x)
    t_norm[build] = x[build] / peak_pos if peak_pos > 0 else np.ones(np.sum(build))
    t_norm[release] = (
        (1.0 - x[release]) / (1.0 - peak_pos)
        if peak_pos < 1
        else np.zeros(np.sum(release))
    )
    envelope = _apply_build_shape(t_norm, build_shape)

    cutoff_frac = envelope
    lfo_rate_frac = envelope

    return cutoff_frac, envelope, lfo_rate_frac


def _curve_rate_hz(
    lfo_rate_frac: np.ndarray, index: int, rate_min: float, rate_max: float
) -> float:
    shaped = float(np.clip(lfo_rate_frac[index], 0.0, 1.0) ** 3)
    return rate_min + (rate_max - rate_min) * shaped


def _apply_modulation_effects_with_curve(
    audio: np.ndarray,
    lfo_rate_frac: np.ndarray,
    *,
    phaser_enabled: bool,
    chorus_enabled: bool,
    flanger_enabled: bool,
    phaser_params: dict,
    chorus_params: dict,
    flanger_params: dict,
) -> tuple[np.ndarray, dict]:
    """Apply phaser/chorus/flanger in blocks with LFO rate following the sweep curve."""
    out = audio.astype(np.float32, copy=True)
    mod_params: dict = {}
    num_samples = len(out)

    if phaser_enabled and phaser_params:
        mod_params["phaser"] = phaser_params
        p_min = float(phaser_params["rate_min_hz"])
        p_max = float(phaser_params["rate_max_hz"])
        for start in range(0, num_samples, BLOCK_SIZE):
            end = min(start + BLOCK_SIZE, num_samples)
            mid = start + (end - start) // 2
            rate = _curve_rate_hz(lfo_rate_frac, mid, p_min, p_max)
            board = Pedalboard(
                [
                    Phaser(
                        rate_hz=rate,
                        depth=float(phaser_params["depth"]),
                        centre_frequency_hz=float(phaser_params["centre_frequency_hz"]),
                        feedback=float(phaser_params["feedback"]),
                        mix=float(phaser_params["mix"]),
                    )
                ]
            )
            block = out[start:end].reshape(1, -1)
            processed = board(block, SAMPLE_RATE)
            out[start:end] = (
                np.mean(processed, axis=0) if processed.shape[0] == 2 else processed[0]
            )

    if chorus_enabled and chorus_params:
        mod_params["chorus"] = chorus_params
        c_min = float(chorus_params["rate_min_hz"])
        c_max = float(chorus_params["rate_max_hz"])
        for start in range(0, num_samples, BLOCK_SIZE):
            end = min(start + BLOCK_SIZE, num_samples)
            mid = start + (end - start) // 2
            rate = _curve_rate_hz(lfo_rate_frac, mid, c_min, c_max)
            board = Pedalboard(
                [
                    Chorus(
                        rate_hz=rate,
                        depth=float(chorus_params["depth"]),
                        centre_delay_ms=float(chorus_params["centre_delay_ms"]),
                        feedback=float(chorus_params.get("feedback", 0.0)),
                        mix=float(chorus_params["mix"]),
                    )
                ]
            )
            block = out[start:end].reshape(1, -1)
            processed = board(block, SAMPLE_RATE)
            out[start:end] = (
                np.mean(processed, axis=0) if processed.shape[0] == 2 else processed[0]
            )

    if flanger_enabled and flanger_params:
        mod_params["flanger"] = flanger_params
        f_min = float(flanger_params["rate_min_hz"])
        f_max = float(flanger_params["rate_max_hz"])
        for start in range(0, num_samples, BLOCK_SIZE):
            end = min(start + BLOCK_SIZE, num_samples)
            mid = start + (end - start) // 2
            rate = _curve_rate_hz(lfo_rate_frac, mid, f_min, f_max)
            board = Pedalboard(
                [
                    Chorus(
                        rate_hz=rate,
                        depth=float(flanger_params["depth"]),
                        centre_delay_ms=float(flanger_params["centre_delay_ms"]),
                        feedback=float(flanger_params["feedback"]),
                        mix=float(flanger_params["mix"]),
                    )
                ]
            )
            block = out[start:end].reshape(1, -1)
            processed = board(block, SAMPLE_RATE)
            out[start:end] = (
                np.mean(processed, axis=0) if processed.shape[0] == 2 else processed[0]
            )

    return out, mod_params


def _compute_tremolo_gain(
    num_samples: int,
    lfo_rate_frac: np.ndarray,
    rate_min_hz: float,
    rate_max_hz: float,
    depth: float,
) -> np.ndarray:
    """Gain modulation: sine LFO, rate follows lfo_rate_frac. depth=0 disables."""
    if depth <= 0:
        return np.ones(num_samples, dtype=np.float64)

    # Strongly ease-in the tremolo speed so it starts very slow and
    # accelerates sharply into the peak.
    shaped = np.clip(lfo_rate_frac, 0.0, 1.0) ** 3
    rate_hz = rate_min_hz + (rate_max_hz - rate_min_hz) * shaped
    phase_inc = 2 * np.pi * rate_hz / SAMPLE_RATE
    phase = np.cumsum(phase_inc)
    # Also ease-in the tremolo depth so modulation is minimal at the
    # start of the sweep and strongest near the peak.
    depth_env = depth * shaped
    # Gain: 1 at peak of sine, 1-depth at trough. (1 + sin) / 2 gives 0..1
    gain = 1.0 - depth_env * (1.0 + np.sin(phase)) / 2.0
    return np.clip(gain, 0.01, 1.0)


def generate_sweep_sample(
    tempo: int = 120,
    bars: int = 2,
    output: str = "sweep.wav",
    config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Generate a noise riser with LFO-modulated filter cutoff.
    Pass config from parse_sweep_config().
    """
    if config is None:
        config = parse_sweep_config()

    cfg = config
    voice = cfg.get("voice") or random.choice(VOICE_TYPES)
    cutoff_low = cfg.get("cutoff_low") or random.randint(*CUTOFF_LOW_RANGE)
    cutoff_high = cfg.get("cutoff_high") or random.randint(*CUTOFF_HIGH_RANGE)
    cutoff_high = max(cutoff_high, cutoff_low + 500)
    peak_pos = cfg.get("peak_pos") or random.uniform(*PEAK_POS_RANGE)
    peak_pos = max(0.05, min(1.0, peak_pos))
    noise_level = cfg.get("noise_level") or random.uniform(*NOISE_LEVEL_RANGE)
    noise_level = max(0.2, min(1.0, noise_level))
    noise_type = cfg.get("noise_type") or random.choice(NOISE_TYPES)
    osc_freq_low = cfg.get("osc_freq_low") or random.uniform(*OSC_FREQ_LOW_RANGE)
    osc_freq_high = cfg.get("osc_freq_high")
    if osc_freq_high is None:
        octaves = random.uniform(*OSC_OCTAVE_SPAN_RANGE)
        osc_freq_high = osc_freq_low * (2.0**octaves)
        osc_freq_high = min(osc_freq_high, SAMPLE_RATE / 2 - 100)
    if voice != "noise":
        pitch_min_hz = cfg.get("pitch_min_hz")
        pitch_max_hz = cfg.get("pitch_max_hz")
        if pitch_min_hz is not None:
            osc_freq_low = max(20.0, min(20000.0, float(pitch_min_hz)))
        if pitch_max_hz is not None:
            osc_freq_high = max(40.0, min(20000.0, float(pitch_max_hz)))
    osc_freq_high = max(osc_freq_high, osc_freq_low + 50)
    osc_level = cfg.get("osc_level") or random.uniform(*OSC_LEVEL_RANGE)
    pulse_width = cfg.get("pulse_width") or random.uniform(*PULSE_WIDTH_RANGE)
    pwm_sweep = cfg.get("pwm_sweep", 0.0)
    detune_cents = cfg.get("detune_cents", 0.0)
    detune_mix = cfg.get("detune_mix", 0.0)
    resonance = cfg.get("resonance", 0.0)
    filter_order = cfg.get("filter_order") or random.choice(FILTER_ORDERS)
    filter_order = 2 if filter_order == 2 else (4 if filter_order == 4 else 6)
    build_shape = _resolve_build_shape(cfg.get("build_shape"))
    tremolo_depth = cfg.get("tremolo_depth")
    if tremolo_depth is None:
        tremolo_depth = random.uniform(*TREMOLO_DEPTH_RANGE)
    tremolo_depth = max(0.0, min(1.0, tremolo_depth))
    tremolo_rate_min = cfg.get("tremolo_rate_min") or random.uniform(
        *TREMOLO_RATE_MIN_RANGE
    )
    tremolo_rate_max = cfg.get("tremolo_rate_max") or random.uniform(
        *TREMOLO_RATE_MAX_RANGE
    )
    tremolo_rate_max = max(tremolo_rate_max, tremolo_rate_min + 1.0)
    phaser_enabled = cfg.get("phaser_enabled", random.random() < 0.5)
    chorus_enabled = cfg.get("chorus_enabled", random.random() < 0.5)
    flanger_enabled = cfg.get("flanger_enabled", random.random() < 0.5)
    phaser_params = cfg.get("phaser_params", _resolve_phaser_params({}))
    chorus_params = cfg.get("chorus_params", _resolve_chorus_params({}))
    flanger_params = cfg.get("flanger_params", _resolve_flanger_params({}))

    beats_per_bar = 4
    total_beats = bars * beats_per_bar
    duration_sec = (60.0 / tempo) * total_beats
    num_samples = int(duration_sec * SAMPLE_RATE)

    t = np.arange(num_samples, dtype=np.float64) / num_samples
    cutoff_frac, envelope, lfo_rate_frac = _riser_build_curve(
        t, peak_pos, build_shape
    )
    gain_enabled = cfg.get("gain_enabled", True)
    gain_min = float(cfg.get("gain_min", 0.0))
    gain_max = float(cfg.get("gain_max", 1.0))
    if gain_max < gain_min:
        gain_min, gain_max = gain_max, gain_min
    if gain_enabled:
        envelope = gain_min + envelope * (gain_max - gain_min)
    else:
        envelope = np.ones_like(envelope, dtype=np.float64)

    base_cutoffs_hz = cutoff_low + cutoff_frac * (cutoff_high - cutoff_low)

    filter_sweep_type = (cfg.get("filter_sweep_type") or "lpf").lower()
    filter_lfo_width = cfg.get("filter_lfo_width", 0.0)
    filter_lfo_rate_min = cfg.get("filter_lfo_rate_min") or 0.2
    filter_lfo_rate_peak = cfg.get("filter_lfo_rate_peak") or 12.0
    filter_lfo_rate_peak = max(filter_lfo_rate_peak, filter_lfo_rate_min + 0.5)
    sweep_range = cutoff_high - cutoff_low
    # Strong ease-in for filter LFO rate so modulation is very slow at the
    # start and accelerates into the peak.
    lfo_shape = np.clip(lfo_rate_frac, 0.0, 1.0) ** 3
    lfo_rate_hz = filter_lfo_rate_min + lfo_shape * (
        filter_lfo_rate_peak - filter_lfo_rate_min
    )
    lfo_phase_inc = 2 * np.pi * lfo_rate_hz / SAMPLE_RATE
    lfo_phase = np.cumsum(lfo_phase_inc)
    lfo_mod = np.sin(lfo_phase) * filter_lfo_width * 0.5 * sweep_range
    cutoffs_hz = np.clip(base_cutoffs_hz + lfo_mod, 20, SAMPLE_RATE / 2 - 10).astype(
        np.float64
    )

    rng = np.random.default_rng()
    if voice == "noise":
        source = _generate_noise(num_samples, noise_type, rng) * noise_level
    else:
        freq_sweep = osc_freq_low + cutoff_frac * (osc_freq_high - osc_freq_low)
        source = _generate_oscillator_sweep(
            num_samples,
            freq_sweep,
            voice,
            level=osc_level,
            pulse_width=pulse_width,
            pwm_sweep=pwm_sweep,
            detune_cents=detune_cents,
            detune_mix=detune_mix,
            cutoff_frac=cutoff_frac,
        )
    tremolo_gain = _compute_tremolo_gain(
        num_samples, lfo_rate_frac, tremolo_rate_min, tremolo_rate_max, tremolo_depth
    )

    out = np.zeros(num_samples, dtype=np.float32)
    nyq = 0.5 * SAMPLE_RATE
    zi = None
    use_resonant = voice != "noise" and resonance > 0 and filter_sweep_type == "lpf"

    if filter_sweep_type == "none":
        for start in range(0, num_samples, BLOCK_SIZE):
            end = min(start + BLOCK_SIZE, num_samples)
            block = source[start:end].copy()
            env_block = envelope[start:end]
            trem_block = tremolo_gain[start:end]
            out[start:end] = (block * env_block * trem_block).astype(np.float32)
    else:
        for start in range(0, num_samples, BLOCK_SIZE):
            end = min(start + BLOCK_SIZE, num_samples)
            block = source[start:end].copy()
            mid_idx = start + (end - start) // 2
            cutoff = float(cutoffs_hz[mid_idx])

            if filter_sweep_type == "lpf":
                if use_resonant:
                    q_val = 0.5 + resonance * float(cutoff_frac[mid_idx]) * (
                        RESONANCE_Q_RANGE[1] - 0.5
                    )
                    b, a = dsp.resonant_lowpass_biquad_coeffs(cutoff, q_val)
                else:
                    b, a = butter(filter_order, cutoff / nyq, btype="low")
                zi_use = lfilter_zi(b, a) * block[0] if zi is None else zi
                filtered, zi = lfilter(b, a, block, zi=zi_use)
            elif filter_sweep_type == "hpf":
                fc = np.clip(cutoff / nyq, 0.01, 0.99)
                b, a = butter(filter_order, fc, btype="high")
                zi_use = lfilter_zi(b, a) * block[0] if zi is None else zi
                filtered, zi = lfilter(b, a, block, zi=zi_use)
            elif filter_sweep_type == "bpf":
                bw_oct = 1.0
                low_fc = cutoff / (2 ** (bw_oct / 2))
                high_fc = np.clip(cutoff * (2 ** (bw_oct / 2)), low_fc + 50, nyq - 10)
                low_n, high_n = low_fc / nyq, high_fc / nyq
                low_n, high_n = np.clip(low_n, 0.01, 0.98), np.clip(
                    high_n, low_n + 0.01, 0.99
                )
                b, a = butter(4, [low_n, high_n], btype="band")
                zi_use = lfilter_zi(b, a) * block[0] if zi is None else zi
                filtered, zi = lfilter(b, a, block, zi=zi_use)
            elif filter_sweep_type == "bsf":
                bw_oct = 1.0
                low_fc = cutoff / (2 ** (bw_oct / 2))
                high_fc = np.clip(cutoff * (2 ** (bw_oct / 2)), low_fc + 50, nyq - 10)
                low_n, high_n = low_fc / nyq, high_fc / nyq
                low_n, high_n = np.clip(low_n, 0.01, 0.98), np.clip(
                    high_n, low_n + 0.01, 0.99
                )
                b, a = butter(4, [low_n, high_n], btype="bandstop")
                zi_use = lfilter_zi(b, a) * block[0] if zi is None else zi
                filtered, zi = lfilter(b, a, block, zi=zi_use)
            else:
                filtered = block

            env_block = envelope[start:end]
            trem_block = tremolo_gain[start:end]
            out[start:end] = (filtered * env_block * trem_block).astype(np.float32)

    out, mod_params = _apply_modulation_effects_with_curve(
        out,
        lfo_rate_frac,
        phaser_enabled=phaser_enabled,
        chorus_enabled=chorus_enabled,
        flanger_enabled=flanger_enabled,
        phaser_params=phaser_params,
        chorus_params=chorus_params,
        flanger_params=flanger_params,
    )

    sf.write(output, _normalize_audio(out), SAMPLE_RATE, subtype="PCM_16")

    params_used = {
        "voice": voice,
        "cutoff_low": cutoff_low,
        "cutoff_high": cutoff_high,
        "peak_pos": peak_pos,
        "noise_level": noise_level,
        "noise_type": noise_type,
        "osc_freq_low": osc_freq_low,
        "osc_freq_high": osc_freq_high,
        "osc_level": osc_level,
        "pulse_width": pulse_width,
        "pwm_sweep": pwm_sweep,
        "detune_cents": detune_cents,
        "detune_mix": detune_mix,
        "resonance": resonance,
        "filter_order": filter_order,
        "build_shape": build_shape,
        "tremolo_depth": tremolo_depth,
        "tremolo_rate_min": tremolo_rate_min,
        "tremolo_rate_max": tremolo_rate_max,
        "phaser": phaser_enabled,
        "chorus": chorus_enabled,
        "flanger": flanger_enabled,
        "modulation_params": mod_params,
        "filter_sweep_type": filter_sweep_type,
        "filter_lfo_width": filter_lfo_width,
        "filter_lfo_rate_min": filter_lfo_rate_min,
        "filter_lfo_rate_peak": filter_lfo_rate_peak,
        "sweep_voice": cfg.get("sweep_voice"),
        "gain_enabled": gain_enabled,
        "gain_min": gain_min,
        "gain_max": gain_max,
    }
    return output, params_used


# --- Closh: washed clap with long reverb (and optional tempo-synced delay) ---

# Convolution reverb: higher quality than algorithmic. Param ranges for randomization.
CLOSH_REVERB_WET_RANGE = (0.4, 0.6)  # 40–60% wet (dry/wet 0.0–1.0)
CLOSH_REVERB_LENGTH_SEC_RANGE = (4.0, 6.0)  # long IR for stadium-style tail
CLOSH_REVERB_DECAY_SEC_RANGE = (2.5, 5.0)  # slow, smooth decay
CLOSH_REVERB_EARLY_REFLECTIONS_RANGE = (20, 40)  # dense early field
CLOSH_REVERB_HIGHPASS_RANGE = (70.0, 120.0)  # avoid low-end mud
CLOSH_REVERB_TAIL_DIFFUSION_RANGE = (0.65, 0.9)  # high = smoother, stadium-like

# Delay param ranges (tempo-synced when enabled)
CLOSH_DELAY_DIVISIONS: tuple[str, ...] = ("1/4", "1/8", "1/8d", "1/16", "1/16d", "1/32")
CLOSH_DELAY_FEEDBACK_RANGE = (0.2, 0.55)
CLOSH_DELAY_MIX_RANGE = (0.15, 0.5)

# Output tail length (seconds) so reverb tail is not truncated
CLOSH_TAIL_SEC = 12.0


def _get_random_kick_path() -> Path:
    """Pick a random .wav from DRUM_KICK_PATHS. Supports comma-separated roots."""
    paths_str = get_setting("DRUM_KICK_PATHS", "")
    roots = parse_escaped_csv(paths_str)
    if not roots:
        raise ValueError(
            "DRUM_KICK_PATHS is not configured. Add paths to kick samples in settings."
        )
    root = Path(random.choice(roots)).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"DRUM_KICK_PATHS root does not exist or is not a dir: {root}")
    candidates = [
        f for f in root.iterdir() if f.is_file() and f.suffix.lower() == ".wav"
    ]
    if not candidates:
        raise ValueError(f"No .wav files found in {root}")
    return random.choice(candidates)


def _get_random_clap_path() -> Path:
    """Pick a random .wav from DRUM_CLAP_PATHS. Supports comma-separated roots."""
    paths_str = get_setting("DRUM_CLAP_PATHS", "")
    roots = parse_escaped_csv(paths_str)
    if not roots:
        raise ValueError(
            "DRUM_CLAP_PATHS is not configured. Add paths to clap samples in settings."
        )
    root = Path(random.choice(roots)).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"DRUM_CLAP_PATHS root does not exist or is not a dir: {root}")
    candidates = [
        f for f in root.iterdir() if f.is_file() and f.suffix.lower() == ".wav"
    ]
    if not candidates:
        raise ValueError(f"No .wav files found in {root}")
    return random.choice(candidates)


def _get_random_cymbal_path() -> Path:
    """Pick a random .wav from DRUM_CYMBAL_PATHS. Supports comma-separated roots."""
    paths_str = get_setting("DRUM_CYMBAL_PATHS", "")
    roots = parse_escaped_csv(paths_str)
    if not roots:
        raise ValueError(
            "DRUM_CYMBAL_PATHS is not configured. Add paths to cymbal samples in settings."
        )
    root = Path(random.choice(roots)).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(
            f"DRUM_CYMBAL_PATHS root does not exist or is not a dir: {root}"
        )
    candidates = [
        f for f in root.iterdir() if f.is_file() and f.suffix.lower() == ".wav"
    ]
    if not candidates:
        raise ValueError(f"No .wav files found in {root}")
    return random.choice(candidates)


def parse_closh_config(
    reverb: str | None = None,
    delay: str | None = None,
) -> dict[str, Any]:
    """
    Parse closh param strings into resolved config.
    --reverb: wet_level, length_sec, decay_sec, early_reflections, highpass_hz (use _ for random)
    --delay: division (1/4|1/8|1/8d|1/16|1/16d|1/32), feedback, mix. Omit or 'off' to disable.
    """
    r = _parse_param_string(reverb)
    d = _parse_param_string(delay)
    use_delay = delay is not None and delay.strip().lower() not in (
        "",
        "off",
        "0",
        "false",
    )
    return {
        "reverb_wet_level": _resolve_float(
            r, "wet_level", CLOSH_REVERB_WET_RANGE, 0.0, 1.0
        ),
        "reverb_length_sec": _resolve_float(
            r, "length_sec", CLOSH_REVERB_LENGTH_SEC_RANGE, 2.0, 10.0
        ),
        "reverb_decay_sec": _resolve_float(
            r, "decay_sec", CLOSH_REVERB_DECAY_SEC_RANGE, 1.5, 8.0
        ),
        "reverb_early_reflections": _resolve_int(
            r, "early_reflections", CLOSH_REVERB_EARLY_REFLECTIONS_RANGE, 10, 60
        ),
        "reverb_highpass_hz": _resolve_float(
            r, "highpass_hz", CLOSH_REVERB_HIGHPASS_RANGE, 40, 180
        ),
        "reverb_tail_diffusion": _resolve_float(
            r, "tail_diffusion", CLOSH_REVERB_TAIL_DIFFUSION_RANGE, 0.3, 0.95
        ),
        "delay_enabled": use_delay,
        "delay_division": _resolve_choice(d, "division", CLOSH_DELAY_DIVISIONS),
        "delay_feedback": _resolve_float(
            d, "feedback", CLOSH_DELAY_FEEDBACK_RANGE, 0, 0.8
        ),
        "delay_mix": _resolve_float(d, "mix", CLOSH_DELAY_MIX_RANGE, 0, 0.8),
    }


def generate_closh_sample(
    tempo: int = 120,
    bars: int = 2,
    output: str = "closh.wav",
    config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Generate a washed clap transition: random clap from DRUM_CLAP_PATHS with
    long reverb and optional tempo-synced delay.
    """
    if config is None:
        config = parse_closh_config()

    clap_path = _get_random_clap_path()
    audio, sr = sf.read(str(clap_path), dtype="float32")

    # Pedalboard expects (channels, samples); soundfile returns (samples, channels)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    else:
        audio = np.ascontiguousarray(audio.T)

    # Resample if needed; librosa expects (samples, channels)
    if sr != SAMPLE_RATE:
        import librosa

        audio_lr = audio.T
        resampled = librosa.resample(audio_lr, orig_sr=sr, target_sr=SAMPLE_RATE)
        if resampled.ndim == 1:
            audio = resampled[np.newaxis, :]
        else:
            audio = np.ascontiguousarray(resampled.T)

    n_samples = audio.shape[1]
    tail_samples = int(CLOSH_TAIL_SEC * SAMPLE_RATE)
    total_len = n_samples + tail_samples
    padded = np.zeros((audio.shape[0], total_len), dtype=np.float32)
    padded[:, :n_samples] = audio

    # Validate input has signal
    input_peak = np.max(np.abs(audio))
    if input_peak < 1e-6:
        raise ValueError(f"Clap sample is effectively silent: {clap_path}")

    # High-quality convolution reverb (stadium-style: diffuse early field, smooth tail)
    ir = dsp.make_reverb_ir(
        SAMPLE_RATE,
        length_sec=config["reverb_length_sec"],
        decay_sec=config["reverb_decay_sec"],
        early_reflections=config["reverb_early_reflections"],
        highpass_cutoff_hz=config["reverb_highpass_hz"],
        seed=random.randint(0, 2**31 - 1),
        tail_diffusion=config["reverb_tail_diffusion"],
        early_diffuse=True,
    )
    wet_level = config["reverb_wet_level"]
    n_channels = padded.shape[0]
    wet = np.zeros((n_channels, padded.shape[1] + len(ir) - 1), dtype=np.float32)
    for ch in range(n_channels):
        wet[ch] = fftconvolve(padded[ch], ir, mode="full")
    dry_extended = np.zeros_like(wet, dtype=np.float32)
    dry_extended[:, : padded.shape[1]] = padded
    wet_peak = np.max(np.abs(wet)) + 1e-12
    dry_peak = np.max(np.abs(padded)) + 1e-12
    wet = wet * (dry_peak / wet_peak)
    processed = wet_level * wet + (1.0 - wet_level) * dry_extended

    # Optionally apply tempo-synced delay
    if config["delay_enabled"]:
        delay_sec = dsp.delay_division_to_seconds(config["delay_division"], float(tempo))
        delay_plugin = Delay(
            delay_seconds=delay_sec,
            feedback=config["delay_feedback"],
            mix=config["delay_mix"],
        )
        board = Pedalboard([delay_plugin])
        chunk_samples = 44100
        proc_len = processed.shape[1]
        n_chunks = (proc_len + chunk_samples - 1) // chunk_samples
        delay_chunks = []
        for i in range(n_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, proc_len)
            chunk = processed[:, start:end]
            out_chunk = board(chunk, SAMPLE_RATE, buffer_size=4096, reset=(i == 0))
            delay_chunks.append(out_chunk)
        processed = np.concatenate(delay_chunks, axis=1)

    # Trim to desired length (bars)
    duration_sec = bars * 4 * (60.0 / tempo)
    max_samples = int(duration_sec * SAMPLE_RATE)
    out = processed[:, :max_samples]
    if out.shape[0] == 2:
        out_mono = np.mean(out, axis=0)
    else:
        out_mono = out[0]
    sf.write(output, _normalize_audio(out_mono), SAMPLE_RATE, subtype="PCM_16")

    params_used = {
        "clap_path": str(clap_path),
        "reverb_wet_level": config["reverb_wet_level"],
        "reverb_length_sec": config["reverb_length_sec"],
        "reverb_decay_sec": config["reverb_decay_sec"],
        "reverb_early_reflections": config["reverb_early_reflections"],
        "reverb_highpass_hz": config["reverb_highpass_hz"],
        "reverb_tail_diffusion": config["reverb_tail_diffusion"],
        "delay_enabled": config["delay_enabled"],
        "delay_division": config["delay_division"],
        "delay_feedback": config["delay_feedback"],
        "delay_mix": config["delay_mix"],
    }
    return output, params_used


def generate_kickboom_sample(
    tempo: int = 120,
    bars: int = 2,
    output: str = "kickboom.wav",
    config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Generate a washed kick transition: random kick from DRUM_KICK_PATHS with
    long reverb and optional tempo-synced delay. Same config interface as closh.
    """
    if config is None:
        config = parse_closh_config()

    kick_path = _get_random_kick_path()
    audio, sr = sf.read(str(kick_path), dtype="float32")

    # Pedalboard expects (channels, samples); soundfile returns (samples, channels)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    else:
        audio = np.ascontiguousarray(audio.T)

    # Resample if needed
    if sr != SAMPLE_RATE:
        import librosa

        audio_lr = audio.T
        resampled = librosa.resample(audio_lr, orig_sr=sr, target_sr=SAMPLE_RATE)
        if resampled.ndim == 1:
            audio = resampled[np.newaxis, :]
        else:
            audio = np.ascontiguousarray(resampled.T)

    n_samples = audio.shape[1]
    tail_samples = int(CLOSH_TAIL_SEC * SAMPLE_RATE)
    total_len = n_samples + tail_samples
    padded = np.zeros((audio.shape[0], total_len), dtype=np.float32)
    padded[:, :n_samples] = audio

    input_peak = np.max(np.abs(audio))
    if input_peak < 1e-6:
        raise ValueError(f"Kick sample is effectively silent: {kick_path}")

    ir = dsp.make_reverb_ir(
        SAMPLE_RATE,
        length_sec=config["reverb_length_sec"],
        decay_sec=config["reverb_decay_sec"],
        early_reflections=config["reverb_early_reflections"],
        highpass_cutoff_hz=config["reverb_highpass_hz"],
        seed=random.randint(0, 2**31 - 1),
        tail_diffusion=config["reverb_tail_diffusion"],
        early_diffuse=True,
    )
    wet_level = config["reverb_wet_level"]
    n_channels = padded.shape[0]
    wet = np.zeros((n_channels, padded.shape[1] + len(ir) - 1), dtype=np.float32)
    for ch in range(n_channels):
        wet[ch] = fftconvolve(padded[ch], ir, mode="full")
    dry_extended = np.zeros_like(wet, dtype=np.float32)
    dry_extended[:, : padded.shape[1]] = padded
    wet_peak = np.max(np.abs(wet)) + 1e-12
    dry_peak = np.max(np.abs(padded)) + 1e-12
    wet = wet * (dry_peak / wet_peak)
    processed = wet_level * wet + (1.0 - wet_level) * dry_extended

    if config["delay_enabled"]:
        delay_sec = dsp.delay_division_to_seconds(config["delay_division"], float(tempo))
        delay_plugin = Delay(
            delay_seconds=delay_sec,
            feedback=config["delay_feedback"],
            mix=config["delay_mix"],
        )
        board = Pedalboard([delay_plugin])
        chunk_samples = 44100
        proc_len = processed.shape[1]
        n_chunks = (proc_len + chunk_samples - 1) // chunk_samples
        delay_chunks = []
        for i in range(n_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, proc_len)
            chunk = processed[:, start:end]
            out_chunk = board(chunk, SAMPLE_RATE, buffer_size=4096, reset=(i == 0))
            delay_chunks.append(out_chunk)
        processed = np.concatenate(delay_chunks, axis=1)

    duration_sec = bars * 4 * (60.0 / tempo)
    max_samples = int(duration_sec * SAMPLE_RATE)
    out = processed[:, :max_samples]
    if out.shape[0] == 2:
        out_mono = np.mean(out, axis=0)
    else:
        out_mono = out[0]
    sf.write(output, _normalize_audio(out_mono), SAMPLE_RATE, subtype="PCM_16")

    params_used = {
        "kick_path": str(kick_path),
        "reverb_wet_level": config["reverb_wet_level"],
        "reverb_length_sec": config["reverb_length_sec"],
        "reverb_decay_sec": config["reverb_decay_sec"],
        "reverb_early_reflections": config["reverb_early_reflections"],
        "reverb_highpass_hz": config["reverb_highpass_hz"],
        "reverb_tail_diffusion": config["reverb_tail_diffusion"],
        "delay_enabled": config["delay_enabled"],
        "delay_division": config["delay_division"],
        "delay_feedback": config["delay_feedback"],
        "delay_mix": config["delay_mix"],
    }
    return output, params_used


def generate_longcrash_sample(
    tempo: int = 120,
    bars: int = 8,
    output: str = "longcrash.wav",
    config: dict[str, Any] | None = None,
    stretch: float = 3.0,
    window_size: float = 0.25,
) -> tuple[str, dict[str, Any]]:
    """
    Generate a long crash transition:
    - random cymbal from DRUM_CYMBAL_PATHS
    - long convolution reverb (same engine as closh/kickboom)
    - optional tempo-synced delay
    - Paulstretch applied after reverb for an elongated, evolving tail.
    """
    from paulstretch import paulstretch as paulstretch_fn

    if config is None:
        config = parse_closh_config()

    cym_path = _get_random_cymbal_path()
    audio, sr = sf.read(str(cym_path), dtype="float32")

    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    else:
        audio = np.ascontiguousarray(audio.T)

    # Resample cymbal to internal SAMPLE_RATE if needed
    if sr != SAMPLE_RATE:
        import librosa

        audio_lr = audio.T
        resampled = librosa.resample(audio_lr, orig_sr=sr, target_sr=SAMPLE_RATE)
        if resampled.ndim == 1:
            audio = resampled[np.newaxis, :]
        else:
            audio = np.ascontiguousarray(resampled.T)

    n_samples = audio.shape[1]
    tail_samples = int(CLOSH_TAIL_SEC * SAMPLE_RATE)
    total_len = n_samples + tail_samples
    padded = np.zeros((audio.shape[0], total_len), dtype=np.float32)
    padded[:, :n_samples] = audio

    input_peak = np.max(np.abs(audio))
    if input_peak < 1e-6:
        raise ValueError(f"Cymbal sample is effectively silent: {cym_path}")

    ir = dsp.make_reverb_ir(
        SAMPLE_RATE,
        length_sec=config["reverb_length_sec"],
        decay_sec=config["reverb_decay_sec"],
        early_reflections=config["reverb_early_reflections"],
        highpass_cutoff_hz=config["reverb_highpass_hz"],
        seed=random.randint(0, 2**31 - 1),
        tail_diffusion=config["reverb_tail_diffusion"],
        early_diffuse=True,
    )
    wet_level = config["reverb_wet_level"]
    n_channels = padded.shape[0]
    wet = np.zeros((n_channels, padded.shape[1] + len(ir) - 1), dtype=np.float32)
    for ch in range(n_channels):
        wet[ch] = fftconvolve(padded[ch], ir, mode="full")
    dry_extended = np.zeros_like(wet, dtype=np.float32)
    dry_extended[:, : padded.shape[1]] = padded
    wet_peak = np.max(np.abs(wet)) + 1e-12
    dry_peak = np.max(np.abs(padded)) + 1e-12
    wet = wet * (dry_peak / wet_peak)
    processed = wet_level * wet + (1.0 - wet_level) * dry_extended

    # Optional tempo-synced delay (same as closh/kickboom)
    if config["delay_enabled"]:
        delay_sec = dsp.delay_division_to_seconds(config["delay_division"], float(tempo))
        delay_plugin = Delay(
            delay_seconds=delay_sec,
            feedback=config["delay_feedback"],
            mix=config["delay_mix"],
        )
        board = Pedalboard([delay_plugin])
        chunk_samples = 44100
        proc_len = processed.shape[1]
        n_chunks = (proc_len + chunk_samples - 1) // chunk_samples
        delay_chunks = []
        for i in range(n_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, proc_len)
            chunk = processed[:, start:end]
            out_chunk = board(chunk, SAMPLE_RATE, buffer_size=4096, reset=(i == 0))
            delay_chunks.append(out_chunk)
        processed = np.concatenate(delay_chunks, axis=1)

    # Use temp files for intermediate processing (only final output is kept)
    duration_sec = bars * 4 * (60.0 / tempo)
    max_samples = int(duration_sec * SAMPLE_RATE)
    pre_stretch = processed[:, :max_samples]
    if pre_stretch.shape[0] == 2:
        pre_mono = np.mean(pre_stretch, axis=0)
    else:
        pre_mono = pre_stretch[0]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        pre_path = f.name
    try:
        sf.write(pre_path, _normalize_audio(pre_mono), SAMPLE_RATE, subtype="PCM_16")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            stretched_path = f.name
        try:
            paulstretch_fn(
                pre_path,
                stretched_path,
                stretch=stretch,
                window_size=window_size,
                show_logs=False,
            )

            stretched_audio, sr_out = sf.read(stretched_path, dtype="float32")
        finally:
            Path(stretched_path).unlink(missing_ok=True)
    finally:
        Path(pre_path).unlink(missing_ok=True)

    if sr_out != SAMPLE_RATE:
        import librosa

        stretched_audio = librosa.resample(
            stretched_audio.T, orig_sr=sr_out, target_sr=SAMPLE_RATE
        ).T
        sr_out = SAMPLE_RATE

    if stretched_audio.ndim > 1:
        stretched_mono = np.mean(stretched_audio, axis=1)
    else:
        stretched_mono = stretched_audio

    target_samples = int(duration_sec * SAMPLE_RATE)
    if stretched_mono.shape[0] >= target_samples:
        final_audio = stretched_mono[:target_samples]
    else:
        pad = target_samples - stretched_mono.shape[0]
        final_audio = np.pad(stretched_mono, (0, pad))

    sf.write(output, _normalize_audio(final_audio), SAMPLE_RATE, subtype="PCM_16")

    params_used = {
        "cymbal_path": str(cym_path),
        "reverb_wet_level": config["reverb_wet_level"],
        "reverb_length_sec": config["reverb_length_sec"],
        "reverb_decay_sec": config["reverb_decay_sec"],
        "reverb_early_reflections": config["reverb_early_reflections"],
        "reverb_highpass_hz": config["reverb_highpass_hz"],
        "reverb_tail_diffusion": config["reverb_tail_diffusion"],
        "delay_enabled": config["delay_enabled"],
        "delay_division": config["delay_division"],
        "delay_feedback": config["delay_feedback"],
        "delay_mix": config["delay_mix"],
        "stretch": stretch,
        "window_size": window_size,
    }
    return output, params_used
