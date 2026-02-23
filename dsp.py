"""
Shared DSP utilities for generate_transition, process_sample, and related modules.
"""

import numpy as np
from scipy.signal import butter, cheby2, filtfilt, sosfiltfilt

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


def _allpass_diffusion(x: np.ndarray, delay_samples: int, g: float) -> np.ndarray:
    """Single allpass for diffusion. y[n] = -g*x[n] + x[n-D] + g*y[n-D]."""
    out = np.zeros_like(x)
    D = min(max(1, delay_samples), len(x) - 1)
    for i in range(D):
        out[i] = -g * x[i]
    for i in range(D, len(x)):
        out[i] = -g * x[i] + x[i - D] + g * out[i - D]
    return out


def make_reverb_ir(
    sample_rate: float,
    length_sec: float = 0.7,
    decay_sec: float = 0.5,
    early_reflections: int = 5,
    highpass_cutoff_hz: float = 0.0,
    seed: int | None = 42,
    tail_diffusion: float = 0.7,
    early_diffuse: bool = True,
) -> np.ndarray:
    """
    Build a high-quality reverb impulse response for offline convolution.
    Early reflections (soft diffuse bursts or discrete) + dense tail with
    optional allpass diffusion for stadium/hall quality.
    seed: RNG seed for IR variation.
    tail_diffusion: 0-1, higher = smoother tail (less grainy). 0.7+ for stadium.
    early_diffuse: if True, use soft bursts instead of spikes for natural early field.
    """
    n = int(sample_rate * length_sec)
    ir = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(seed if seed is not None else 42)
    early_start = int(0.005 * sample_rate)
    early_end = int(0.08 * sample_rate)

    for _ in range(early_reflections):
        center = int(rng.uniform(early_start, min(early_end, n - 50)))
        amp = rng.uniform(0.06, 0.22)
        if early_diffuse:
            burst_len = int(rng.uniform(0.001 * sample_rate, 0.004 * sample_rate))
            burst_len = min(burst_len, n - center - 1)
            if burst_len > 1:
                t = np.arange(burst_len, dtype=np.float64) / burst_len
                envelope = np.exp(-t * 3)
                ir[center : center + burst_len] += amp * envelope
            else:
                ir[center] += amp
        else:
            if center < n:
                ir[center] += amp

    tail_start = int(0.04 * sample_rate)
    tail_len = n - tail_start
    noise = rng.standard_normal(tail_len)
    alpha = 0.2 + 0.65 * np.clip(tail_diffusion, 0, 1)
    for i in range(1, tail_len):
        noise[i] = alpha * noise[i - 1] + (1 - alpha) * noise[i]
    t = np.arange(tail_len, dtype=np.float64) / sample_rate
    decay = np.exp(-t * (3.0 / decay_sec))
    ir[tail_start:] = ir[tail_start:] + noise * decay

    if tail_diffusion >= 0.5:
        primes = [347, 113, 37]
        for d, g in zip(primes, [0.6, 0.55, 0.5]):
            ir = _allpass_diffusion(ir, d, g)

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


def _apply_sos_per_channel(
    audio: np.ndarray, sos: np.ndarray
) -> np.ndarray:
    """Apply IIR filter in SOS form per channel. audio shape (channels, samples)."""
    out = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        out[ch] = sosfiltfilt(sos, audio[ch])
    return out


def apply_butter_filter(
    audio: np.ndarray,
    sample_rate: float,
    btype: str,
    cutoff_hz: float | tuple[float, float],
    order: int = 4,
) -> np.ndarray:
    """
    Apply Butterworth filter. btype: 'low', 'high', or 'band'.
    For 'band', cutoff_hz must be (low_hz, high_hz).
    """
    nyq = 0.5 * sample_rate
    if btype == "band":
        low, high = cutoff_hz
        low_n = max(0.01, min(low, high - 50) / nyq)
        high_n = min(0.99, high / nyq)
        if low_n >= high_n:
            low_n, high_n = 0.02, 0.98
        b, a = butter(order, [low_n, high_n], btype="band")
    else:
        fc = np.clip(cutoff_hz / nyq, 0.01, 0.99)
        b, a = butter(order, fc, btype=btype)
    return apply_iir_per_channel(audio, sample_rate, b, a)


def apply_steep_lowpass(audio: np.ndarray, sample_rate: float, cutoff_hz: float) -> np.ndarray:
    """
    Apply aggressive low-pass with steep cutoff (Chebyshev Type II, order 10).
    Truly cuts high frequencies rather than gently rolling off.
    """
    nyq = 0.5 * sample_rate
    fc = np.clip(cutoff_hz / nyq, 0.02, 0.98)
    sos = cheby2(N=10, rs=60, Wn=fc, btype="low", output="sos")
    return _apply_sos_per_channel(audio, sos)


def apply_steep_highpass(audio: np.ndarray, sample_rate: float, cutoff_hz: float) -> np.ndarray:
    """
    Apply aggressive high-pass with steep cutoff (Chebyshev Type II, order 10).
    Truly cuts low frequencies rather than gently rolling off.
    """
    nyq = 0.5 * sample_rate
    fc = np.clip(cutoff_hz / nyq, 0.02, 0.98)
    sos = cheby2(N=10, rs=60, Wn=fc, btype="high", output="sos")
    return _apply_sos_per_channel(audio, sos)
