"""Transition sound generation (sweeps, risers, etc.) using offline DSP."""

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, lfilter_zi

SAMPLE_RATE = 44100
CUTOFF_LOW_HZ = 400
CUTOFF_HIGH_HZ = 14000
FILTER_ORDER = 4
BLOCK_SIZE = 256


def _riser_build_curve(normalized_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Riser curves: cutoff and envelope both build from 0 to 1 by the middle,
    then cutoff stays open (bright) while envelope does a smooth exponential decay.
    This avoids the "kick" from closing the filter and a sharp envelope drop.
    """
    x = np.clip(normalized_t, 0, 1)
    mid = 0.5
    build = x <= mid
    release = ~build

    # Build phase: smooth ease-in rise 0 -> 1 (quadratic)
    t_build = x[build] / mid
    build_val = t_build**2

    # Release phase: exponential decay so no sharp transient
    t_release = (x[release] - mid) / (1 - mid)
    decay_val = np.exp(-4.0 * t_release)

    cutoff_frac = np.empty_like(x)
    cutoff_frac[build] = build_val
    cutoff_frac[release] = 1.0  # keep filter open - stay bright

    envelope = np.empty_like(x)
    envelope[build] = build_val
    envelope[release] = decay_val

    return cutoff_frac, envelope


def generate_sweep_sample(
    tempo: int = 120,
    bars: int = 2,
    output: str = "sweep.wav",
) -> str:
    """
    Generate a white noise riser with LFO-modulated filter cutoff.
    Builds from muffled/quiet to bright/loud by the middle, then smooth decay.
    Trance/house style - high-end sweeping noise, no low thump.
    """
    beats_per_bar = 4
    total_beats = bars * beats_per_bar
    duration_sec = (60.0 / tempo) * total_beats
    num_samples = int(duration_sec * SAMPLE_RATE)

    # White noise - slightly more level for a fuller sweep
    rng = np.random.default_rng()
    noise = rng.standard_normal(num_samples).astype(np.float64) * 0.6

    # Normalized time 0..1
    t = np.arange(num_samples, dtype=np.float64) / num_samples

    # Cutoff and envelope: build to peak at middle, no closing/thump
    cutoff_frac, envelope = _riser_build_curve(t)
    cutoffs_hz = CUTOFF_LOW_HZ + cutoff_frac * (CUTOFF_HIGH_HZ - CUTOFF_LOW_HZ)

    # Process in blocks with time-varying lowpass (preserve filter state)
    out = np.zeros(num_samples, dtype=np.float32)
    nyq = 0.5 * SAMPLE_RATE
    zi = None

    for start in range(0, num_samples, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, num_samples)
        block = noise[start:end].copy()

        mid_idx = start + (end - start) // 2
        cutoff = float(np.clip(cutoffs_hz[mid_idx], 20, nyq - 1))

        b, a = butter(FILTER_ORDER, cutoff / nyq, btype="low")
        zi_use = lfilter_zi(b, a) * block[0] if zi is None else zi
        filtered, zi = lfilter(b, a, block, zi=zi_use)

        env_block = envelope[start:end]
        out[start:end] = (filtered * env_block).astype(np.float32)

    # Normalize
    peak = np.max(np.abs(out))
    if peak > 0.9:
        out = out * (0.9 / peak)

    sf.write(output, out, SAMPLE_RATE, subtype="PCM_16")
    return output
