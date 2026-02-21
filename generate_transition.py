"""Transition sound generation (sweeps, risers, etc.) using offline DSP."""

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, lfilter_zi

SAMPLE_RATE = 44100
CUTOFF_LOW_HZ = 200
CUTOFF_HIGH_HZ = 8000
FILTER_ORDER = 4
BLOCK_SIZE = 256


def _sweep_lfo(normalized_t: np.ndarray) -> np.ndarray:
    """
    LFO curve that accelerates into the middle and decelerates out.
    Returns values 0..1, peaking at the midpoint (steepest change in the middle).
    """
    x = np.clip(normalized_t, 0, 1)
    mid = 0.5
    left = x <= mid
    right = ~left
    out = np.empty_like(x)
    # First half: parabola from 0 to 0.5 (accelerating)
    t_left = x[left] / mid
    out[left] = 2 * t_left**2
    # Second half: parabola from 0.5 to 1 (decelerating)
    t_right = (x[right] - mid) / (1 - mid)
    out[right] = 1 - 2 * (1 - t_right) ** 2
    return out


def generate_sweep_sample(
    tempo: int = 120,
    bars: int = 2,
    output: str = "sweep.wav",
) -> str:
    """
    Generate a white noise sweep with LFO-modulated filter cutoff.
    The modulation accelerates into the middle (end of first bar for 2 bars)
    and decelerates out, producing a techno/trance-style riser/sweep.
    """
    beats_per_bar = 4
    total_beats = bars * beats_per_bar
    duration_sec = (60.0 / tempo) * total_beats
    num_samples = int(duration_sec * SAMPLE_RATE)

    # White noise
    rng = np.random.default_rng()
    noise = rng.standard_normal(num_samples).astype(np.float32) * 0.4

    # Normalized time 0..1
    t = np.arange(num_samples, dtype=np.float64) / num_samples

    # LFO: 0 at edges, 1 at middle (accelerating then decelerating)
    lfo = _sweep_lfo(t)

    # Amplitude envelope: same shape, so everything peaks in the middle
    envelope = _sweep_lfo(t)

    # Cutoff follows LFO: low -> high -> low
    cutoffs_hz = CUTOFF_LOW_HZ + lfo * (CUTOFF_HIGH_HZ - CUTOFF_LOW_HZ)

    # Process in blocks with time-varying lowpass (preserve filter state across blocks)
    out = np.zeros(num_samples, dtype=np.float32)
    nyq = 0.5 * SAMPLE_RATE
    zi = None

    for start in range(0, num_samples, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, num_samples)
        block = noise[start:end].copy()

        # Use cutoff at block center
        mid_idx = start + (end - start) // 2
        cutoff = float(cutoffs_hz[mid_idx])
        cutoff = np.clip(cutoff, 20, nyq - 1)

        b, a = butter(FILTER_ORDER, cutoff / nyq, btype="low")
        zi_use = lfilter_zi(b, a) * block[0] if zi is None else zi
        filtered, zi = lfilter(b, a, block, zi=zi_use.astype(np.float64))

        # Apply envelope
        env_block = envelope[start:end]
        out[start:end] = (filtered * env_block).astype(np.float32)

    # Normalize to avoid clipping
    peak = np.max(np.abs(out))
    if peak > 0.9:
        out = out * (0.9 / peak)

    sf.write(output, out, SAMPLE_RATE, subtype="PCM_16")
    return output
