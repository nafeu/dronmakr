"""
Shared DSP utilities for generate_transition, process_sample, and related modules.
"""

import numpy as np
from scipy.signal import butter, filtfilt

SAMPLE_RATE = 44100


def resonant_lowpass_biquad_coeffs(
    cutoff_hz: float, q: float, sample_rate: float = SAMPLE_RATE
) -> tuple[np.ndarray, np.ndarray]:
    """RBJ lowpass biquad. Q controls resonance (0.5=flat, higher=peaky)."""
    w0 = 2 * np.pi * cutoff_hz / sample_rate
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / (2 * max(q, 0.5))
    b0 = (1 - cos_w0) / 2
    b1 = 1 - cos_w0
    b2 = (1 - cos_w0) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def highpass_ir(
    ir: np.ndarray, sample_rate: float, cutoff_hz: float, order: int = 2
) -> np.ndarray:
    """Apply highpass to an IR so reverb has no energy below cutoff_hz (avoids low-end phase clash)."""
    if cutoff_hz <= 0 or cutoff_hz >= sample_rate / 2.1:
        return ir
    nyq = 0.5 * sample_rate
    low = cutoff_hz / nyq
    b, a = butter(order, low, btype="high")
    return filtfilt(b, a, ir).astype(ir.dtype)


def make_reverb_ir(
    sample_rate: float,
    length_sec: float = 0.7,
    decay_sec: float = 0.5,
    early_reflections: int = 5,
    highpass_cutoff_hz: float = 0.0,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Build a high-quality reverb impulse response for offline convolution.
    Early reflections (discrete echoes) + dense tail (exponential decay of
    filtered noise). Optional highpass on the IR so reverb doesn't reflect
    low-end (set highpass_cutoff_hz > 0, e.g. 80–120 for drums).
    seed: RNG seed for IR variation (None = deterministic default).
    """
    n = int(sample_rate * length_sec)
    ir = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(seed if seed is not None else 42)
    for _ in range(early_reflections):
        idx = int(rng.uniform(0.002 * sample_rate, 0.04 * sample_rate))
        if idx < n:
            ir[idx] += rng.uniform(0.12, 0.35)
    tail_start = int(0.03 * sample_rate)
    tail_len = n - tail_start
    noise = rng.standard_normal(tail_len)
    alpha = 0.3
    for i in range(1, tail_len):
        noise[i] = alpha * noise[i - 1] + (1 - alpha) * noise[i]
    t = np.arange(tail_len, dtype=np.float64) / sample_rate
    decay = np.exp(-t * (3.0 / decay_sec))
    ir[tail_start:] = ir[tail_start:] + noise * decay
    if highpass_cutoff_hz > 0:
        ir = highpass_ir(ir, sample_rate, highpass_cutoff_hz)
    ir = ir / (np.max(np.abs(ir)) + 1e-12)
    return ir.astype(np.float32)


def apply_iir_per_channel(
    audio: np.ndarray, sample_rate: float, b: np.ndarray, a: np.ndarray
) -> np.ndarray:
    """Apply IIR filter (b, a) to each channel. audio shape (channels, samples)."""
    out = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        out[ch] = filtfilt(b, a, audio[ch])
    return out
