"""Transition sound generation (sweeps, risers, etc.) using offline DSP."""

import random
import re
from typing import Any, Literal

import numpy as np
import soundfile as sf
from pedalboard import Chorus, Pedalboard, Phaser
from scipy.signal import butter, lfilter, lfilter_zi

SAMPLE_RATE = 44100
BLOCK_SIZE = 256

# Default ranges for randomization (when params not passed)
CUTOFF_LOW_RANGE = (300, 700)
CUTOFF_HIGH_RANGE = (10000, 18000)
DECAY_RATE_RANGE = (2.0, 6.0)
PEAK_POS_RANGE = (0.4, 0.6)
NOISE_LEVEL_RANGE = (0.45, 0.8)
FILTER_ORDERS = (2, 4, 6)
BUILD_SHAPES: tuple[Literal["ease_in", "linear", "ease_out"], ...] = (
    "ease_in",
    "linear",
    "ease_out",
)
NOISE_TYPES: tuple[Literal["white", "pink", "brown", "blue"], ...] = (
    "white",
    "pink",
    "brown",
    "blue",
)
TREMOLO_DEPTH_RANGE = (0.4, 0.9)
TREMOLO_RATE_MIN_RANGE = (1.0, 4.0)
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


def _resolve_phaser_params(parsed: dict[str, str]) -> dict[str, float]:
    return {
        "rate_hz": _resolve_float(parsed, "rate_hz", PHASER_RATE_RANGE, 0.1, 10),
        "depth": _resolve_float(parsed, "depth", PHASER_DEPTH_RANGE, 0, 1),
        "centre_frequency_hz": _resolve_float(parsed, "centre_frequency_hz", PHASER_CENTRE_RANGE, 100, 10000),
        "feedback": _resolve_float(parsed, "feedback", PHASER_FEEDBACK_RANGE, 0, 1),
        "mix": _resolve_float(parsed, "mix", PHASER_MIX_RANGE, 0, 1),
    }


def _resolve_chorus_params(parsed: dict[str, str]) -> dict[str, float]:
    return {
        "rate_hz": _resolve_float(parsed, "rate_hz", CHORUS_RATE_RANGE, 0.1, 10),
        "depth": _resolve_float(parsed, "depth", CHORUS_DEPTH_RANGE, 0, 1),
        "centre_delay_ms": _resolve_float(parsed, "centre_delay_ms", CHORUS_DELAY_RANGE, 1, 20),
        "feedback": _resolve_float(parsed, "feedback", (0, 0.5), 0, 0.5),
        "mix": _resolve_float(parsed, "mix", CHORUS_MIX_RANGE, 0, 1),
    }


def _resolve_flanger_params(parsed: dict[str, str]) -> dict[str, float]:
    return {
        "rate_hz": _resolve_float(parsed, "rate_hz", FLANGER_RATE_RANGE, 0.1, 5),
        "depth": _resolve_float(parsed, "depth", FLANGER_DEPTH_RANGE, 0, 1),
        "centre_delay_ms": _resolve_float(parsed, "centre_delay_ms", FLANGER_DELAY_RANGE, 0.5, 10),
        "feedback": _resolve_float(parsed, "feedback", FLANGER_FEEDBACK_RANGE, 0, 0.8),
        "mix": _resolve_float(parsed, "mix", FLANGER_MIX_RANGE, 0, 1),
    }


def parse_sweep_config(
    noise: str | None = None,
    curve: str | None = None,
    filter_str: str | None = None,
    tremolo: str | None = None,
    phaser: str | None = None,
    chorus: str | None = None,
    flanger: str | None = None,
    disable: str | None = None,
) -> dict[str, Any]:
    """
    Parse encoded param strings into a resolved config dict.
    Syntax: key:value;key2:value2. Use _ or empty for random. E.g. --noise type:white;noise_level:0.6
    """
    disabled = set()
    if disable:
        for d in re.split(r"[\s,]+", disable.strip().lower()):
            d = d.strip()
            if d == "fx":
                disabled.update(("tremolo", "phaser", "chorus", "flanger"))
            elif d in ("tremolo", "phaser", "chorus", "flanger"):
                disabled.add(d)

    n = _parse_param_string(noise)
    c = _parse_param_string(curve)
    f = _parse_param_string(filter_str)
    t = _parse_param_string(tremolo)
    ph = _parse_param_string(phaser)
    ch = _parse_param_string(chorus)
    fl = _parse_param_string(flanger)

    tremolo_rate_min = _resolve_float(t, "rate_min_hz", TREMOLO_RATE_MIN_RANGE, 0.5, 20)
    tremolo_rate_max = _resolve_float(t, "rate_max_hz", TREMOLO_RATE_MAX_RANGE, 2, 50)
    tremolo_rate_max = max(tremolo_rate_max, tremolo_rate_min + 1.0)

    return {
        "noise_type": _resolve_choice(n, "type", NOISE_TYPES),
        "noise_level": _resolve_float(n, "noise_level", NOISE_LEVEL_RANGE, 0.2, 1.0),
        "build_shape": _resolve_choice(c, "shape", BUILD_SHAPES),
        "decay_rate": _resolve_float(c, "decay_rate", DECAY_RATE_RANGE, 0.5, 15.0),
        "peak_pos": _resolve_float(c, "peak_pos", PEAK_POS_RANGE, 0.15, 0.85),
        "cutoff_low": _resolve_int(f, "cutoff_low", CUTOFF_LOW_RANGE, 50, 5000),
        "cutoff_high": _resolve_int(f, "cutoff_high", CUTOFF_HIGH_RANGE, 1000, 20000),
        "tremolo_depth": 0.0 if "tremolo" in disabled else _resolve_float(t, "depth", TREMOLO_DEPTH_RANGE, 0, 1),
        "tremolo_rate_min": tremolo_rate_min,
        "tremolo_rate_max": tremolo_rate_max,
        "phaser_enabled": "phaser" not in disabled and (phaser is not None or random.random() < 0.5),
        "phaser_params": _resolve_phaser_params(ph) if "phaser" not in disabled else {},
        "chorus_enabled": "chorus" not in disabled and (chorus is not None or random.random() < 0.5),
        "chorus_params": _resolve_chorus_params(ch) if "chorus" not in disabled else {},
        "flanger_enabled": "flanger" not in disabled and (flanger is not None or random.random() < 0.5),
        "flanger_params": _resolve_flanger_params(fl) if "flanger" not in disabled else {},
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


def _apply_build_shape(t_norm: np.ndarray, shape: Literal["ease_in", "linear", "ease_out"]) -> np.ndarray:
    """Map normalized 0..1 to build curve. ease_in: slow start; linear: constant; ease_out: slow end."""
    t = np.clip(t_norm, 0, 1)
    if shape == "linear":
        return t
    if shape == "ease_in":
        return t**2
    if shape == "ease_out":
        return 1 - (1 - t) ** 2
    return t


def _riser_build_curve(
    normalized_t: np.ndarray,
    peak_pos: float,
    decay_rate: float,
    build_shape: Literal["ease_in", "linear", "ease_out"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Riser curves: cutoff and envelope build from 0 to 1 by peak_pos,
    then cutoff stays open while envelope does a smooth exponential decay.
    Also returns lfo_rate_frac (0->1 during build, 1->0 during release) for tremolo.
    """
    x = np.clip(normalized_t, 0, 1)
    build = x <= peak_pos
    release = ~build

    t_build = x[build] / peak_pos if peak_pos > 0 else np.ones(np.sum(build))
    build_val = _apply_build_shape(t_build, build_shape)

    t_release = (x[release] - peak_pos) / (1 - peak_pos) if peak_pos < 1 else np.zeros(np.sum(release))
    decay_val = np.exp(-decay_rate * t_release)

    cutoff_frac = np.empty_like(x)
    cutoff_frac[build] = build_val
    cutoff_frac[release] = 1.0

    envelope = np.empty_like(x)
    envelope[build] = build_val
    envelope[release] = decay_val

    # LFO rate curve: increases during build, decreases during release (for tremolo)
    lfo_rate_frac = np.empty_like(x)
    lfo_rate_frac[build] = build_val
    lfo_rate_frac[release] = decay_val

    return cutoff_frac, envelope, lfo_rate_frac


def _build_modulation_board_from_config(
    phaser_enabled: bool,
    chorus_enabled: bool,
    flanger_enabled: bool,
    phaser_params: dict,
    chorus_params: dict,
    flanger_params: dict,
) -> tuple[Pedalboard | None, dict]:
    """Build a Pedalboard from config. Params dicts already have resolved values."""
    plugins = []
    mod_params: dict = {}
    if phaser_enabled and phaser_params:
        mod_params["phaser"] = phaser_params
        plugins.append(Phaser(**phaser_params))
    if chorus_enabled and chorus_params:
        mod_params["chorus"] = chorus_params
        plugins.append(Chorus(**chorus_params))
    if flanger_enabled and flanger_params:
        mod_params["flanger"] = flanger_params
        plugins.append(Chorus(**flanger_params))
    if not plugins:
        return None, {}
    return Pedalboard(plugins), mod_params


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

    rate_hz = rate_min_hz + (rate_max_hz - rate_min_hz) * lfo_rate_frac
    phase_inc = 2 * np.pi * rate_hz / SAMPLE_RATE
    phase = np.cumsum(phase_inc)
    # Gain: 1 at peak of sine, 1-depth at trough. (1 + sin) / 2 gives 0..1, so gain = 1 - depth * (1 + sin) / 2
    gain = 1.0 - depth * (1.0 + np.sin(phase)) / 2.0
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
    cutoff_low = cfg.get("cutoff_low") or random.randint(*CUTOFF_LOW_RANGE)
    cutoff_high = cfg.get("cutoff_high") or random.randint(*CUTOFF_HIGH_RANGE)
    cutoff_high = max(cutoff_high, cutoff_low + 500)
    decay_rate = cfg.get("decay_rate") or random.uniform(*DECAY_RATE_RANGE)
    peak_pos = cfg.get("peak_pos") or random.uniform(*PEAK_POS_RANGE)
    peak_pos = max(0.15, min(0.85, peak_pos))
    noise_level = cfg.get("noise_level") or random.uniform(*NOISE_LEVEL_RANGE)
    noise_level = max(0.2, min(1.0, noise_level))
    noise_type = cfg.get("noise_type") or random.choice(NOISE_TYPES)
    filter_order = cfg.get("filter_order") or random.choice(FILTER_ORDERS)
    filter_order = 2 if filter_order == 2 else (4 if filter_order == 4 else 6)
    build_shape = cfg.get("build_shape") or random.choice(BUILD_SHAPES)
    tremolo_depth = cfg.get("tremolo_depth")
    if tremolo_depth is None:
        tremolo_depth = random.uniform(*TREMOLO_DEPTH_RANGE)
    tremolo_depth = max(0.0, min(1.0, tremolo_depth))
    tremolo_rate_min = cfg.get("tremolo_rate_min") or random.uniform(*TREMOLO_RATE_MIN_RANGE)
    tremolo_rate_max = cfg.get("tremolo_rate_max") or random.uniform(*TREMOLO_RATE_MAX_RANGE)
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

    rng = np.random.default_rng()
    noise = _generate_noise(num_samples, noise_type, rng) * noise_level

    t = np.arange(num_samples, dtype=np.float64) / num_samples

    cutoff_frac, envelope, lfo_rate_frac = _riser_build_curve(t, peak_pos, decay_rate, build_shape)
    cutoffs_hz = cutoff_low + cutoff_frac * (cutoff_high - cutoff_low)
    tremolo_gain = _compute_tremolo_gain(
        num_samples, lfo_rate_frac, tremolo_rate_min, tremolo_rate_max, tremolo_depth
    )

    out = np.zeros(num_samples, dtype=np.float32)
    nyq = 0.5 * SAMPLE_RATE
    zi = None

    for start in range(0, num_samples, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, num_samples)
        block = noise[start:end].copy()

        mid_idx = start + (end - start) // 2
        cutoff = float(np.clip(cutoffs_hz[mid_idx], 20, nyq - 1))

        b, a = butter(filter_order, cutoff / nyq, btype="low")
        zi_use = lfilter_zi(b, a) * block[0] if zi is None else zi
        filtered, zi = lfilter(b, a, block, zi=zi_use)

        env_block = envelope[start:end]
        trem_block = tremolo_gain[start:end]
        out[start:end] = (filtered * env_block * trem_block).astype(np.float32)

    # Normalize
    peak = np.max(np.abs(out))
    if peak > 0.9:
        out = out * (0.9 / peak)

    # Apply modulation effects (phaser, chorus, flanger) via pedalboard
    mod_board, mod_params = _build_modulation_board_from_config(
        phaser_enabled, chorus_enabled, flanger_enabled,
        phaser_params, chorus_params, flanger_params,
    )
    if mod_board is not None:
        audio_2d = out.reshape(1, -1).astype(np.float32)
        processed = mod_board(audio_2d, SAMPLE_RATE)
        if processed.shape[0] == 2:
            out = np.mean(processed, axis=0).astype(np.float32)
        else:
            out = processed[0].astype(np.float32)
        peak = np.max(np.abs(out))
        if peak > 0.9:
            out = out * (0.9 / peak)

    sf.write(output, out, SAMPLE_RATE, subtype="PCM_16")

    params_used = {
        "cutoff_low": cutoff_low,
        "cutoff_high": cutoff_high,
        "decay_rate": decay_rate,
        "peak_pos": peak_pos,
        "noise_level": noise_level,
        "noise_type": noise_type,
        "filter_order": filter_order,
        "build_shape": build_shape,
        "tremolo_depth": tremolo_depth,
        "tremolo_rate_min": tremolo_rate_min,
        "tremolo_rate_max": tremolo_rate_max,
        "phaser": phaser_enabled,
        "chorus": chorus_enabled,
        "flanger": flanger_enabled,
        "modulation_params": mod_params,
    }
    return output, params_used
