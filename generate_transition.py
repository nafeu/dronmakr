"""Transition sound generation (sweeps, risers, etc.) using offline DSP."""

import random
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, lfilter, lfilter_zi

from pedalboard import Chorus, Delay, Pedalboard, Phaser

import dsp
from settings import get_setting

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
DECAY_RATE_RANGE = (2.0, 6.0)
PEAK_POS_RANGE = (0.4, 0.6)
NOISE_LEVEL_RANGE = (0.45, 0.8)
FILTER_ORDERS = (2, 4, 6)
FILTER_SWEEP_TYPES: tuple[str, ...] = ("lpf", "hpf", "bpf", "bsf", "none")
FILTER_LFO_WIDTH_RANGE = (0.0, 0.25)
FILTER_LFO_RATE_MIN_RANGE = (0.5, 3.0)
FILTER_LFO_RATE_PEAK_RANGE = (5.0, 25.0)
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
VOICE_TYPES: tuple[Literal["noise", "sine", "saw", "tri", "square"], ...] = (
    "noise",
    "sine",
    "saw",
    "tri",
    "square",
)
OSC_FREQ_LOW_RANGE = (55, 220)  # A1 to A3
OSC_FREQ_HIGH_RANGE = (880, 4000)  # A5 to ~B7 (used only when freq_high explicit)
OSC_OCTAVE_SPAN_RANGE = (
    2.5,
    4.5,
)  # octaves for sweep when randomizing (2.5–4.5 ≈ 3–4 octaves)
OSC_LEVEL_RANGE = (0.3, 0.8)
PULSE_WIDTH_RANGE = (0.2, 0.8)
PWM_SWEEP_RANGE = (0.0, 0.35)  # how much pulse width modulates (0=static)
DETUNE_CENTS_RANGE = (5, 25)  # cents for unison detune
DETUNE_MIX_RANGE = (0.2, 0.6)  # max mix of detuned layer (follows build curve)
RESONANCE_Q_RANGE = (0.5, 8.0)  # filter Q at peak (0.5=flat, 8=acid scream)
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
        "centre_frequency_hz": _resolve_float(
            parsed, "centre_frequency_hz", PHASER_CENTRE_RANGE, 100, 10000
        ),
        "feedback": _resolve_float(parsed, "feedback", PHASER_FEEDBACK_RANGE, 0, 1),
        "mix": _resolve_float(parsed, "mix", PHASER_MIX_RANGE, 0, 1),
    }


def _resolve_chorus_params(parsed: dict[str, str]) -> dict[str, float]:
    return {
        "rate_hz": _resolve_float(parsed, "rate_hz", CHORUS_RATE_RANGE, 0.1, 10),
        "depth": _resolve_float(parsed, "depth", CHORUS_DEPTH_RANGE, 0, 1),
        "centre_delay_ms": _resolve_float(
            parsed, "centre_delay_ms", CHORUS_DELAY_RANGE, 1, 20
        ),
        "feedback": _resolve_float(parsed, "feedback", (0, 0.5), 0, 0.5),
        "mix": _resolve_float(parsed, "mix", CHORUS_MIX_RANGE, 0, 1),
    }


def _resolve_flanger_params(parsed: dict[str, str]) -> dict[str, float]:
    return {
        "rate_hz": _resolve_float(parsed, "rate_hz", FLANGER_RATE_RANGE, 0.1, 5),
        "depth": _resolve_float(parsed, "depth", FLANGER_DEPTH_RANGE, 0, 1),
        "centre_delay_ms": _resolve_float(
            parsed, "centre_delay_ms", FLANGER_DELAY_RANGE, 0.5, 10
        ),
        "feedback": _resolve_float(parsed, "feedback", FLANGER_FEEDBACK_RANGE, 0, 0.8),
        "mix": _resolve_float(parsed, "mix", FLANGER_MIX_RANGE, 0, 1),
    }


def _resolve_sound_params(parsed: dict[str, str]) -> dict[str, Any]:
    """Resolve voice-specific params from parsed --sound string."""
    voice = _resolve_choice(parsed, "voice", VOICE_TYPES)

    result: dict[str, Any] = {"voice": voice}

    if voice == "noise":
        result["noise_type"] = _resolve_choice(parsed, "type", NOISE_TYPES)
        result["noise_level"] = _resolve_float(
            parsed, "noise_level", NOISE_LEVEL_RANGE, 0.2, 1.0
        )
    else:
        # Oscillator: sine, saw, tri, square
        # Use octaves to span (default 2.5–4.5 octaves); or freq_high if user specifies it
        freq_low = _resolve_float(parsed, "freq_low", OSC_FREQ_LOW_RANGE, 20, 2000)
        octaves = _resolve_float(parsed, "octaves", OSC_OCTAVE_SPAN_RANGE, 1, 6)
        freq_high_explicit = parsed.get("freq_high") and not _is_random(
            parsed.get("freq_high")
        )
        if freq_high_explicit:
            freq_high = _resolve_float(
                parsed, "freq_high", OSC_FREQ_HIGH_RANGE, 200, 12000
            )
            freq_high = max(freq_high, freq_low + 50)
        else:
            freq_high = freq_low * (2.0**octaves)
            freq_high = min(freq_high, SAMPLE_RATE / 2 - 100)
            freq_high = max(freq_high, freq_low + 50)
        result["osc_freq_low"] = freq_low
        result["osc_freq_high"] = freq_high
        result["osc_level"] = _resolve_float(parsed, "level", OSC_LEVEL_RANGE, 0.1, 1.0)
        result["pulse_width"] = _resolve_float(
            parsed, "pulse_width", PULSE_WIDTH_RANGE, 0.1, 0.9
        )
        result["pwm_sweep"] = _resolve_float(
            parsed, "pwm_sweep", PWM_SWEEP_RANGE, 0, 0.5
        )
        result["detune_cents"] = _resolve_float(
            parsed, "detune_cents", DETUNE_CENTS_RANGE, 1, 50
        )
        result["detune_mix"] = _resolve_float(
            parsed, "detune_mix", DETUNE_MIX_RANGE, 0, 1
        )
        result["resonance"] = _resolve_float(parsed, "resonance", (0, 1), 0, 1)

    return result


def parse_sweep_config(
    sound: str | None = None,
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
    --sound: voice (noise|sine|saw|tri|square), then voice-specific params.
    E.g. --sound "voice:noise;type:white;noise_level:0.6" or --sound "voice:square;freq_low:110;octaves:3" or --sound "voice:saw;freq_low:55;freq_high:880"
    """
    disabled = set()
    if disable:
        for d in re.split(r"[\s,]+", disable.strip().lower()):
            d = d.strip()
            if d == "fx":
                disabled.update(("tremolo", "phaser", "chorus", "flanger"))
            elif d in ("tremolo", "phaser", "chorus", "flanger"):
                disabled.add(d)

    s = _parse_param_string(sound)
    c = _parse_param_string(curve)
    f = _parse_param_string(filter_str)
    t = _parse_param_string(tremolo)
    ph = _parse_param_string(phaser)
    ch = _parse_param_string(chorus)
    fl = _parse_param_string(flanger)

    tremolo_rate_min = _resolve_float(t, "rate_min_hz", TREMOLO_RATE_MIN_RANGE, 0.5, 20)
    tremolo_rate_max = _resolve_float(t, "rate_max_hz", TREMOLO_RATE_MAX_RANGE, 2, 50)
    tremolo_rate_max = max(tremolo_rate_max, tremolo_rate_min + 1.0)

    sound_params = _resolve_sound_params(s)

    return {
        **sound_params,
        "build_shape": _resolve_choice(c, "shape", BUILD_SHAPES),
        "decay_rate": _resolve_float(c, "decay_rate", DECAY_RATE_RANGE, 0.5, 15.0),
        "peak_pos": _resolve_float(c, "peak_pos", PEAK_POS_RANGE, 0.15, 0.85),
        "cutoff_low": _resolve_int(f, "cutoff_low", CUTOFF_LOW_RANGE, 50, 5000),
        "cutoff_high": _resolve_int(f, "cutoff_high", CUTOFF_HIGH_RANGE, 1000, 20000),
        "filter_sweep_type": (
            _resolve_choice(f, "type", FILTER_SWEEP_TYPES)
            if f and "type" in f
            else "lpf"
        ),
        "filter_lfo_width": _resolve_float(
            f, "lfo_width", FILTER_LFO_WIDTH_RANGE, 0, 0.5
        ),
        "filter_lfo_rate_min": _resolve_float(
            f, "lfo_rate_min", FILTER_LFO_RATE_MIN_RANGE, 0.2, 10
        ),
        "filter_lfo_rate_peak": _resolve_float(
            f, "lfo_rate_peak", FILTER_LFO_RATE_PEAK_RANGE, 2, 50
        ),
        "tremolo_depth": (
            0.0
            if "tremolo" in disabled
            else _resolve_float(t, "depth", TREMOLO_DEPTH_RANGE, 0, 1)
        ),
        "tremolo_rate_min": tremolo_rate_min,
        "tremolo_rate_max": tremolo_rate_max,
        "phaser_enabled": "phaser" not in disabled
        and (phaser is not None or random.random() < 0.5),
        "phaser_params": _resolve_phaser_params(ph) if "phaser" not in disabled else {},
        "chorus_enabled": "chorus" not in disabled
        and (chorus is not None or random.random() < 0.5),
        "chorus_params": _resolve_chorus_params(ch) if "chorus" not in disabled else {},
        "flanger_enabled": "flanger" not in disabled
        and (flanger is not None or random.random() < 0.5),
        "flanger_params": (
            _resolve_flanger_params(fl) if "flanger" not in disabled else {}
        ),
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


def _apply_build_shape(
    t_norm: np.ndarray, shape: Literal["ease_in", "linear", "ease_out"]
) -> np.ndarray:
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
    Sweep curves: build from 0 to 1 by peak_pos, then sweep back down to 0 over the
    remaining time. Cutoff, envelope, and LFO rate all follow this up-then-down shape
    so the sweep returns to its initial state by the end.
    """
    x = np.clip(normalized_t, 0, 1)
    build = x <= peak_pos
    release = ~build

    t_build = x[build] / peak_pos if peak_pos > 0 else np.ones(np.sum(build))
    build_val = _apply_build_shape(t_build, build_shape)

    # Release: use remaining time (1 - peak_pos) to sweep back down to 0
    t_release = (
        (x[release] - peak_pos) / (1 - peak_pos)
        if peak_pos < 1
        else np.zeros(np.sum(release))
    )
    # Mirror the build shape for symmetric sweep-down
    release_val = 1.0 - _apply_build_shape(t_release, build_shape)

    cutoff_frac = np.empty_like(x)
    cutoff_frac[build] = build_val
    cutoff_frac[release] = release_val

    envelope = np.empty_like(x)
    envelope[build] = build_val
    envelope[release] = release_val

    # LFO rate for tremolo: same up-then-down shape
    lfo_rate_frac = np.empty_like(x)
    lfo_rate_frac[build] = build_val
    lfo_rate_frac[release] = release_val

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
    voice = cfg.get("voice") or random.choice(VOICE_TYPES)
    cutoff_low = cfg.get("cutoff_low") or random.randint(*CUTOFF_LOW_RANGE)
    cutoff_high = cfg.get("cutoff_high") or random.randint(*CUTOFF_HIGH_RANGE)
    cutoff_high = max(cutoff_high, cutoff_low + 500)
    decay_rate = cfg.get("decay_rate") or random.uniform(*DECAY_RATE_RANGE)
    peak_pos = cfg.get("peak_pos") or random.uniform(*PEAK_POS_RANGE)
    peak_pos = max(0.15, min(0.85, peak_pos))
    noise_level = cfg.get("noise_level") or random.uniform(*NOISE_LEVEL_RANGE)
    noise_level = max(0.2, min(1.0, noise_level))
    noise_type = cfg.get("noise_type") or random.choice(NOISE_TYPES)
    osc_freq_low = cfg.get("osc_freq_low") or random.uniform(*OSC_FREQ_LOW_RANGE)
    osc_freq_high = cfg.get("osc_freq_high")
    if osc_freq_high is None:
        octaves = random.uniform(*OSC_OCTAVE_SPAN_RANGE)
        osc_freq_high = osc_freq_low * (2.0**octaves)
        osc_freq_high = min(osc_freq_high, SAMPLE_RATE / 2 - 100)
    osc_freq_high = max(osc_freq_high, osc_freq_low + 50)
    osc_level = cfg.get("osc_level") or random.uniform(*OSC_LEVEL_RANGE)
    pulse_width = cfg.get("pulse_width") or random.uniform(*PULSE_WIDTH_RANGE)
    pwm_sweep = cfg.get("pwm_sweep", 0.0)
    detune_cents = cfg.get("detune_cents", 0.0)
    detune_mix = cfg.get("detune_mix", 0.0)
    resonance = cfg.get("resonance", 0.0)
    filter_order = cfg.get("filter_order") or random.choice(FILTER_ORDERS)
    filter_order = 2 if filter_order == 2 else (4 if filter_order == 4 else 6)
    build_shape = cfg.get("build_shape") or random.choice(BUILD_SHAPES)
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
        t, peak_pos, decay_rate, build_shape
    )
    base_cutoffs_hz = cutoff_low + cutoff_frac * (cutoff_high - cutoff_low)

    filter_sweep_type = (cfg.get("filter_sweep_type") or "lpf").lower()
    filter_lfo_width = cfg.get("filter_lfo_width", 0.0)
    filter_lfo_rate_min = cfg.get("filter_lfo_rate_min") or 1.0
    filter_lfo_rate_peak = cfg.get("filter_lfo_rate_peak") or 12.0
    filter_lfo_rate_peak = max(filter_lfo_rate_peak, filter_lfo_rate_min + 0.5)
    sweep_range = cutoff_high - cutoff_low
    lfo_rate_hz = filter_lfo_rate_min + lfo_rate_frac * (
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

    # Apply modulation effects (phaser, chorus, flanger) via pedalboard
    mod_board, mod_params = _build_modulation_board_from_config(
        phaser_enabled,
        chorus_enabled,
        flanger_enabled,
        phaser_params,
        chorus_params,
        flanger_params,
    )
    if mod_board is not None:
        audio_2d = out.reshape(1, -1).astype(np.float32)
        processed = mod_board(audio_2d, SAMPLE_RATE)
        if processed.shape[0] == 2:
            out = np.mean(processed, axis=0).astype(np.float32)
        else:
            out = processed[0].astype(np.float32)

    sf.write(output, _normalize_audio(out), SAMPLE_RATE, subtype="PCM_16")

    params_used = {
        "voice": voice,
        "cutoff_low": cutoff_low,
        "cutoff_high": cutoff_high,
        "decay_rate": decay_rate,
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


def _get_random_clap_path() -> Path:
    """Pick a random .wav from DRUM_CLAP_PATHS. Supports comma-separated roots."""
    paths_str = get_setting("DRUM_CLAP_PATHS", "")
    roots = [p.strip() for p in paths_str.split(",") if p.strip()]
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


def _delay_division_to_seconds(division: str, tempo: int) -> float:
    """Convert musical division to delay time in seconds. 1/8d = dotted 1/8."""
    beat_sec = 60.0 / tempo
    if division == "1/4":
        return beat_sec
    if division == "1/8":
        return beat_sec / 2
    if division == "1/8d":
        return beat_sec * 0.75
    if division == "1/16":
        return beat_sec / 4
    if division == "1/16d":
        return beat_sec * 0.375
    if division == "1/32":
        return beat_sec / 8
    return beat_sec / 2


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
        delay_sec = _delay_division_to_seconds(config["delay_division"], tempo)
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
        delay_sec = _delay_division_to_seconds(config["delay_division"], tempo)
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
