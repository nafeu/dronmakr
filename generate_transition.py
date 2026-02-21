"""Transition sound generation (sweeps, risers, etc.) using offline DSP."""

import random
from typing import Literal

import numpy as np
import soundfile as sf
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
    cutoff_low: int | None = None,
    cutoff_high: int | None = None,
    decay_rate: float | None = None,
    peak_pos: float | None = None,
    noise_level: float | None = None,
    noise_type: Literal["white", "pink", "brown", "blue"] | None = None,
    filter_order: int | None = None,
    build_shape: Literal["ease_in", "linear", "ease_out"] | None = None,
    tremolo_depth: float | None = None,
    tremolo_rate_min: float | None = None,
    tremolo_rate_max: float | None = None,
) -> str:
    """
    Generate a noise riser with LFO-modulated filter cutoff.
    Builds from muffled/quiet to bright/loud, then smooth decay.
    Trance/house style - high-end sweeping noise, no low thump.
    """
    # Randomize params when not provided
    cutoff_low = cutoff_low if cutoff_low is not None else random.randint(*CUTOFF_LOW_RANGE)
    cutoff_high = cutoff_high if cutoff_high is not None else random.randint(*CUTOFF_HIGH_RANGE)
    cutoff_high = max(cutoff_high, cutoff_low + 500)  # ensure valid sweep range
    decay_rate = decay_rate if decay_rate is not None else random.uniform(*DECAY_RATE_RANGE)
    peak_pos = peak_pos if peak_pos is not None else random.uniform(*PEAK_POS_RANGE)
    peak_pos = max(0.15, min(0.85, peak_pos))  # avoid edge cases
    noise_level = noise_level if noise_level is not None else random.uniform(*NOISE_LEVEL_RANGE)
    noise_level = max(0.2, min(1.0, noise_level))
    noise_type = noise_type if noise_type is not None else random.choice(NOISE_TYPES)
    filter_order = filter_order if filter_order is not None else random.choice(FILTER_ORDERS)
    filter_order = 2 if filter_order == 2 else (4 if filter_order == 4 else 6)
    build_shape = build_shape if build_shape is not None else random.choice(BUILD_SHAPES)
    tremolo_depth = tremolo_depth if tremolo_depth is not None else random.uniform(*TREMOLO_DEPTH_RANGE)
    tremolo_depth = max(0.0, min(1.0, tremolo_depth))
    tremolo_rate_min = tremolo_rate_min if tremolo_rate_min is not None else random.uniform(*TREMOLO_RATE_MIN_RANGE)
    tremolo_rate_min = max(0.5, min(8.0, tremolo_rate_min))
    tremolo_rate_max = tremolo_rate_max if tremolo_rate_max is not None else random.uniform(*TREMOLO_RATE_MAX_RANGE)
    tremolo_rate_max = max(tremolo_rate_min + 2.0, min(40.0, tremolo_rate_max))

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
    }
    return output, params_used
