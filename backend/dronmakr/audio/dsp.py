"""
Shared DSP utilities for generate_transition, process_sample, and related modules.
"""

from __future__ import annotations

import math

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


def _cheby2_order_for_steepness(steepness: float, *, base: int, span: int, cap: int) -> int:
    """Map 0..1 steepness to filter order; steepness=0 keeps legacy base order."""
    s = float(np.clip(steepness, 0.0, 1.0))
    if s <= 1e-9:
        return base
    return int(max(base, min(cap, round(base + s * span))))


def _apply_resonance_peak(
    audio: np.ndarray,
    sample_rate: float,
    centre_hz: float,
    resonance: float,
) -> np.ndarray:
    """Narrow peaking boost at cutoff / band centre (0..1 maps to modest DJ-style resonance)."""
    r = float(np.clip(resonance, 0.0, 1.0))
    if r <= 1e-6:
        return audio
    gain_db = min(18.0, (r**1.35) * 15.0)
    qpk = max(0.9, 38.0 / (r * 34.0 + 0.45))
    return apply_peaking_eq(audio, sample_rate, centre_hz, gain_db, qpk)


def apply_steep_lowpass(
    audio: np.ndarray,
    sample_rate: float,
    cutoff_hz: float,
    *,
    resonance: float = 0.0,
    steepness: float = 0.0,
) -> np.ndarray:
    """
    Low-pass with a sharp transition (Chebyshev type II by default order 10 when steepness=0).
    steepness in 0..1 increases order (stronger brick-wall feel). resonance 0..1 adds a peak at cutoff.
    """
    nyq = 0.5 * sample_rate
    fc = np.clip(cutoff_hz / nyq, 0.02, 0.98)
    n_order = _cheby2_order_for_steepness(steepness, base=10, span=24, cap=36)
    if n_order % 2:
        n_order = min(36, n_order + 1)
    rs = float(np.clip(52.0 + float(np.clip(resonance, 0.0, 1.0)) * 34.0, 48.0, 92.0))
    sos = cheby2(N=n_order, rs=rs, Wn=fc, btype="low", output="sos")
    out = _apply_sos_per_channel(audio, sos)
    return _apply_resonance_peak(out, sample_rate, float(cutoff_hz), resonance)


def apply_steep_highpass(
    audio: np.ndarray,
    sample_rate: float,
    cutoff_hz: float,
    *,
    resonance: float = 0.0,
    steepness: float = 0.0,
) -> np.ndarray:
    """
    High-pass with sharp transition; same steepness/resonance semantics as low-pass.
    """
    nyq = 0.5 * sample_rate
    fc = np.clip(cutoff_hz / nyq, 0.02, 0.98)
    n_order = _cheby2_order_for_steepness(steepness, base=10, span=24, cap=36)
    if n_order % 2:
        n_order = min(36, n_order + 1)
    rs = float(np.clip(52.0 + float(np.clip(resonance, 0.0, 1.0)) * 34.0, 48.0, 92.0))
    sos = cheby2(N=n_order, rs=rs, Wn=fc, btype="high", output="sos")
    out = _apply_sos_per_channel(audio, sos)
    return _apply_resonance_peak(out, sample_rate, float(cutoff_hz), resonance)


def apply_steep_bandpass(
    audio: np.ndarray,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
    *,
    resonance: float = 0.0,
    steepness: float = 0.0,
) -> np.ndarray:
    """
    Steep band-pass (Chebyshev type II). steepness 0 falls back to mild 4th-order Butterworth.
    """
    nyq = 0.5 * sample_rate
    low = float(np.clip(low_hz, 1.0, nyq * 0.99))
    high = float(np.clip(high_hz, low + 1.0, nyq * 0.995))
    lo_n = max(0.01, low / nyq)
    hi_n = min(0.99, high / nyq)
    if lo_n >= hi_n - 1e-7:
        lo_n, hi_n = 0.02, min(0.98, hi_n + 0.02)

    s = float(np.clip(steepness, 0.0, 1.0))
    if s <= 1e-9:
        return apply_butter_filter(audio, sample_rate, "band", (low, high), order=4)

    n_order = int(max(8, min(34, round(8 + s * 26))))
    if n_order % 2:
        n_order = min(34, n_order + 1)
    rs = float(np.clip(50.0 + float(np.clip(resonance, 0.0, 1.0)) * 32.0, 48.0, 90.0))
    sos = cheby2(N=n_order, rs=rs, Wn=[lo_n, hi_n], btype="band", output="sos")
    out = _apply_sos_per_channel(audio, sos)
    centre = float(np.sqrt(max(low * high, 1.0)))
    return _apply_resonance_peak(out, sample_rate, centre, resonance)


def peaking_biquad_coeffs(
    freq_hz: float, gain_db: float, q: float, sample_rate: float = SAMPLE_RATE
) -> tuple[np.ndarray, np.ndarray]:
    """RBJ peaking EQ biquad. gain_db: positive = boost, negative = cut."""
    w0 = 2 * np.pi * freq_hz / sample_rate
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / (2 * max(q, 0.1))
    A = 10.0 ** (gain_db / 40.0)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def apply_peaking_eq(
    audio: np.ndarray, sample_rate: float, freq_hz: float, gain_db: float, q: float = 1.0
) -> np.ndarray:
    """Apply peaking EQ (one band). audio shape (channels, samples)."""
    b, a = peaking_biquad_coeffs(freq_hz, gain_db, q, sample_rate)
    return apply_iir_per_channel(audio, sample_rate, b, a)


def apply_reese_post_eq(
    audio: np.ndarray, sample_rate: float = SAMPLE_RATE
) -> np.ndarray:
    """
    Post-EQ for Reese: highpass ~30 Hz, slight cut 200-400 Hz, boost 700-1500 Hz.
    audio shape (channels, samples).
    """
    nyq = 0.5 * sample_rate
    # Highpass 30 Hz
    fc = np.clip(30.0 / nyq, 0.001, 0.99)
    b, a = butter(2, fc, btype="high")
    out = apply_iir_per_channel(audio, sample_rate, b, a)
    # Broad cut around 300 Hz
    b, a = peaking_biquad_coeffs(300.0, -2.5, 0.8, sample_rate)
    out = apply_iir_per_channel(out, sample_rate, b, a)
    # Broad boost for growl presence
    b, a = peaking_biquad_coeffs(1100.0, 2.0, 0.7, sample_rate)
    out = apply_iir_per_channel(out, sample_rate, b, a)
    return out


def delay_division_to_seconds(division: str, bpm: float) -> float:
    """
    Note value → delay time in seconds (quarter note = one beat at ``bpm``).
    Supports dotted (suffix ``d``) and triplet (suffix ``t``) forms.

    Unknown divisions fall back to an eighth-note delay (half a beat), matching
    legacy ``generate_transition`` behaviour.
    """
    beat_sec = 60.0 / max(float(bpm), 1e-6)
    d = str(division).strip().lower().replace(" ", "")
    table: dict[str, float] = {
        # Beats (quarter = 1 beat)
        "1/64": 1 / 16,
        "1/32": 1 / 8,
        "1/32d": 3 / 16,
        "1/16": 1 / 4,
        "1/16d": 3 / 8,
        "1/16t": 1 / 6,
        "1/8": 1 / 2,
        "1/8d": 3 / 4,
        "1/8t": 1 / 3,
        "1/4": 1.0,
        "1/4d": 1.5,
        "1/4t": 2 / 3,
        "1/2": 2.0,
        "1/1": 4.0,
    }
    mul = table.get(d)
    if mul is None:
        mul = 0.5
    return beat_sec * mul


def _butter1_df1_coeffs(
    fc_hz: float, sample_rate: float, btype: str
) -> tuple[float, float, float]:
    """First-order Butterworth as DF1: y = b0*x + b1*xm1 - a1*ym1 (scipy ``a`` coeffs)."""
    nyq = 0.5 * sample_rate
    w = np.clip(float(fc_hz), 2.0, nyq * 0.97) / nyq
    b, a = butter(1, w, btype=btype)
    return float(b[0]), float(b[1]), float(a[1])


def _numba():
    import numba  # noqa: PLC0415 — lazy: DawDreamer must init LLVM before numba

    return numba


def _feedback_delay_mono_numba(
    dry: np.ndarray,
    out: np.ndarray,
    delay_samples_f: float,
    feedback: float,
    mix: float,
    b0_hp: float,
    b1_hp: float,
    a1_hp: float,
    b0_lp: float,
    b1_lp: float,
    a1_lp: float,
    do_hp: int,
    do_lp: int,
    buf: np.ndarray,
) -> None:
    n = dry.shape[0]
    cap = buf.shape[0]
    wi = 0
    x_hp_m1 = 0.0
    y_hp_m1 = 0.0
    x_lp_m1 = 0.0
    y_lp_m1 = 0.0
    dly = delay_samples_f
    if dly < 1.0:
        dly = 1.0
    if dly > float(cap - 3):
        dly = float(cap - 3)
    fb = feedback
    mx = mix
    if mx < 0.0:
        mx = 0.0
    if mx > 1.0:
        mx = 1.0

    for i in range(n):
        xi = dry[i]
        pos = float(wi) - dly
        while pos < 0.0:
            pos += float(cap)
        i0 = int(math.floor(pos))
        frac = pos - float(i0)
        i0 = int(i0 % cap)
        i1 = int((i0 + 1) % cap)
        tap = buf[i0] * (1.0 - frac) + buf[i1] * frac

        if do_hp != 0:
            y_hp = b0_hp * tap + b1_hp * x_hp_m1 - a1_hp * y_hp_m1
            x_hp_m1 = tap
            y_hp_m1 = y_hp
            src_lp = y_hp
        else:
            src_lp = tap

        if do_lp != 0:
            y_lp = b0_lp * src_lp + b1_lp * x_lp_m1 - a1_lp * y_lp_m1
            x_lp_m1 = src_lp
            y_lp_m1 = y_lp
            s_fb = y_lp
        else:
            s_fb = src_lp

        buf[wi] = xi + fb * s_fb
        out[i] = (1.0 - mx) * xi + mx * tap

        wi += 1
        if wi >= cap:
            wi = 0


_feedback_delay_mono_numba = _numba().njit(cache=True, fastmath=True)(_feedback_delay_mono_numba)


def _feedback_delay_stereo_numba(
    dry_l: np.ndarray,
    dry_r: np.ndarray,
    out_l: np.ndarray,
    out_r: np.ndarray,
    delay_l_f: float,
    delay_r_f: float,
    feedback: float,
    mix: float,
    ping_pong: int,
    stereo_width: float,
    crossfeed: float,
    b0_hp: float,
    b1_hp: float,
    a1_hp: float,
    b0_lp: float,
    b1_lp: float,
    a1_lp: float,
    do_hp: int,
    do_lp: int,
    buf_l: np.ndarray,
    buf_r: np.ndarray,
) -> None:
    n = dry_l.shape[0]
    cap = buf_l.shape[0]
    wi = 0

    x_hp_ml = 0.0
    y_hp_ml = 0.0
    x_lp_ml = 0.0
    y_lp_ml = 0.0
    x_hp_mr = 0.0
    y_hp_mr = 0.0
    x_lp_mr = 0.0
    y_lp_mr = 0.0

    d_l = delay_l_f
    d_r = delay_r_f
    if d_l < 1.0:
        d_l = 1.0
    if d_r < 1.0:
        d_r = 1.0
    max_d = d_l if d_l > d_r else d_r
    capf = float(cap - 3)
    if max_d > capf:
        scale = capf / max_d
        d_l *= scale
        d_r *= scale

    fb = feedback
    mx = mix
    if mx < 0.0:
        mx = 0.0
    if mx > 1.0:
        mx = 1.0

    sw = stereo_width
    if sw < 0.0:
        sw = 0.0
    if sw > 1.0:
        sw = 1.0

    cf = crossfeed
    if cf < 0.0:
        cf = 0.0
    if cf > 1.0:
        cf = 1.0

    for i in range(n):
        xl = dry_l[i]
        xr = dry_r[i]

        pos_l = float(wi) - d_l
        while pos_l < 0.0:
            pos_l += float(cap)
        i0l = int(math.floor(pos_l))
        fracl = pos_l - float(i0l)
        i0l = int(i0l % cap)
        i1l = int((i0l + 1) % cap)
        tap_l = buf_l[i0l] * (1.0 - fracl) + buf_l[i1l] * fracl

        pos_r = float(wi) - d_r
        while pos_r < 0.0:
            pos_r += float(cap)
        i0r = int(math.floor(pos_r))
        fracr = pos_r - float(i0r)
        i0r = int(i0r % cap)
        i1r = int((i0r + 1) % cap)
        tap_r = buf_r[i0r] * (1.0 - fracr) + buf_r[i1r] * fracr

        mid = 0.5 * (tap_l + tap_r)
        wet_l = mid + sw * (tap_l - mid)
        wet_r = mid + sw * (tap_r - mid)

        src_fb_l = tap_r if ping_pong != 0 else tap_l
        src_fb_r = tap_l if ping_pong != 0 else tap_r

        if do_hp != 0:
            yh_l = b0_hp * src_fb_l + b1_hp * x_hp_ml - a1_hp * y_hp_ml
            x_hp_ml = src_fb_l
            y_hp_ml = yh_l
            z_l = yh_l
            yh_r = b0_hp * src_fb_r + b1_hp * x_hp_mr - a1_hp * y_hp_mr
            x_hp_mr = src_fb_r
            y_hp_mr = yh_r
            z_r = yh_r
        else:
            z_l = src_fb_l
            z_r = src_fb_r

        if do_lp != 0:
            ylp_l = b0_lp * z_l + b1_lp * x_lp_ml - a1_lp * y_lp_ml
            x_lp_ml = z_l
            y_lp_ml = ylp_l
            f_l = ylp_l
            ylp_r = b0_lp * z_r + b1_lp * x_lp_mr - a1_lp * y_lp_mr
            x_lp_mr = z_r
            y_lp_mr = ylp_r
            f_r = ylp_r
        else:
            f_l = z_l
            f_r = z_r

        in_l = (1.0 - cf) * xl + cf * xr
        in_r = (1.0 - cf) * xr + cf * xl

        buf_l[wi] = in_l + fb * f_l
        buf_r[wi] = in_r + fb * f_r

        out_l[i] = (1.0 - mx) * xl + mx * wet_l
        out_r[i] = (1.0 - mx) * xr + mx * wet_r

        wi += 1
        if wi >= cap:
            wi = 0


_feedback_delay_stereo_numba = _numba().njit(cache=True, fastmath=True)(_feedback_delay_stereo_numba)


def apply_feedback_delay(
    audio: np.ndarray,
    sample_rate: float,
    *,
    time_mode: str = "sync",
    bpm: float = 120.0,
    division: str = "1/8",
    delay_ms: float = 250.0,
    delay_offset_ms: float = 0.0,
    stereo_spread_ms: float = 0.0,
    feedback: float = 0.42,
    mix: float = 0.35,
    ping_pong: bool = False,
    stereo_width: float = 1.0,
    input_crossfeed: float = 0.0,
    feedback_lowpass_hz: float = 12000.0,
    feedback_highpass_hz: float = 80.0,
    max_delay_sec: float = 8.0,
) -> np.ndarray:
    """
    Stereo feedback delay with fractional delay line, tempo sync or manual ms time,
    feedback damping (one-pole HP/LP), optional ping-pong cross-feedback, Haas-style
    spread (different delay times per channel), wet stereo width, and pre-delay input crossfeed.

    ``audio`` shape ``(channels, samples)``; float32/float64. Mono uses a single delay line
    (ping-pong, spread, and input crossfeed are ignored). More than two channels: independent mono
    delays per channel (same timing), no ping-pong.
    """
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("apply_feedback_delay expects shape (channels, samples)")
    n_ch, n_samp = x.shape
    if n_samp == 0:
        return x

    mode = str(time_mode).strip().lower()
    if mode == "sync":
        base_sec = delay_division_to_seconds(division, bpm) + float(delay_offset_ms) / 1000.0
    else:
        base_sec = float(delay_ms) / 1000.0 + float(delay_offset_ms) / 1000.0

    sr = float(sample_rate)
    half_spread_sec = float(stereo_spread_ms) / 2000.0
    if n_ch >= 2:
        d_l_sec = base_sec + half_spread_sec
        d_r_sec = base_sec - half_spread_sec
    else:
        d_l_sec = d_r_sec = base_sec
    min_sec = 1.0 / sr
    if d_l_sec < min_sec:
        d_l_sec = min_sec
    if d_r_sec < min_sec:
        d_r_sec = min_sec

    delay_l_f = float(d_l_sec * sr)
    delay_r_f = float(d_r_sec * sr)

    max_d_samp = max(delay_l_f, delay_r_f, 1.0)
    cap = int(min(max_delay_sec * sr, max(64.0, max_d_samp + 8.0)))
    cap = max(cap, int(max_d_samp) + 8)

    fb = float(np.clip(feedback, 0.0, 0.995))
    mx = float(np.clip(mix, 0.0, 1.0))
    sw = float(np.clip(stereo_width, 0.0, 1.0))
    cf = float(np.clip(input_crossfeed, 0.0, 1.0))

    nyq = 0.5 * sr
    f_hp = float(feedback_highpass_hz)
    f_lp = float(feedback_lowpass_hz)
    do_hp = 1 if f_hp > 25.0 else 0
    do_lp = 1 if f_lp < nyq * 0.92 else 0

    if do_hp:
        b0_hp, b1_hp, a1_hp = _butter1_df1_coeffs(f_hp, sr, "high")
    else:
        b0_hp = b1_hp = a1_hp = 0.0

    if do_lp:
        b0_lp, b1_lp, a1_lp = _butter1_df1_coeffs(f_lp, sr, "low")
    else:
        b0_lp = b1_lp = a1_lp = 0.0

    pp = 1 if (ping_pong and n_ch == 2) else 0

    out = np.zeros_like(x, dtype=np.float32)

    if n_ch == 1:
        buf = np.zeros(cap, dtype=np.float32)
        _feedback_delay_mono_numba(
            x[0],
            out[0],
            delay_l_f,
            fb,
            mx,
            b0_hp,
            b1_hp,
            a1_hp,
            b0_lp,
            b1_lp,
            a1_lp,
            do_hp,
            do_lp,
            buf,
        )
        return out

    if n_ch == 2:
        buf_l = np.zeros(cap, dtype=np.float32)
        buf_r = np.zeros(cap, dtype=np.float32)
        _feedback_delay_stereo_numba(
            x[0],
            x[1],
            out[0],
            out[1],
            delay_l_f,
            delay_r_f,
            fb,
            mx,
            pp,
            sw,
            cf if cf > 1e-7 else 0.0,
            b0_hp,
            b1_hp,
            a1_hp,
            b0_lp,
            b1_lp,
            a1_lp,
            do_hp,
            do_lp,
            buf_l,
            buf_r,
        )
        return out

    buf_tmp = np.zeros(cap, dtype=np.float32)
    for c in range(n_ch):
        _feedback_delay_mono_numba(
            x[c],
            out[c],
            delay_l_f,
            fb,
            mx,
            b0_hp,
            b1_hp,
            a1_hp,
            b0_lp,
            b1_lp,
            a1_lp,
            do_hp,
            do_lp,
            buf_tmp,
        )
    return out


def _ensure_channels_first(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        return np.stack([x, x], axis=0)
    if x.shape[0] > x.shape[1]:
        x = x.T
    if x.shape[0] == 1:
        x = np.vstack([x, x])
    return x


def apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    x = _ensure_channels_first(audio)
    return x * (10.0 ** (float(gain_db) / 20.0))


def apply_distortion(
    audio: np.ndarray, sample_rate: float, *, drive_db: float = 6.0
) -> np.ndarray:
    del sample_rate
    x = _ensure_channels_first(audio)
    drive = 10.0 ** (float(drive_db) / 20.0)
    return np.tanh(x * drive).astype(np.float32, copy=False)


def _envelope_follower(mono: np.ndarray, attack: float, release: float) -> np.ndarray:
    env = np.zeros_like(mono)
    state = 0.0
    for i, sample in enumerate(np.abs(mono)):
        coeff = attack if sample > state else release
        state = coeff * state + (1.0 - coeff) * sample
        env[i] = state
    return env


def apply_compressor(
    audio: np.ndarray,
    sample_rate: float,
    *,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
) -> np.ndarray:
    x = _ensure_channels_first(audio)
    sr = float(sample_rate)
    attack = math.exp(-1.0 / (sr * max(attack_ms, 0.1) / 1000.0))
    release = math.exp(-1.0 / (sr * max(release_ms, 0.1) / 1000.0))
    thresh = 10.0 ** (float(threshold_db) / 20.0)
    mono = np.mean(x, axis=0)
    env = _envelope_follower(mono, attack, release)
    gain = np.ones_like(env)
    over = env > thresh
    if np.any(over):
        gain[over] = (
            thresh
            + (env[over] - thresh) / max(float(ratio), 1.0)
        ) / (env[over] + 1e-12)
    return (x * gain).astype(np.float32, copy=False)


def apply_limiter(
    audio: np.ndarray,
    sample_rate: float,
    *,
    threshold_db: float = -4.0,
    release_ms: float = 250.0,
) -> np.ndarray:
    return apply_compressor(
        audio,
        sample_rate,
        threshold_db=threshold_db,
        ratio=20.0,
        attack_ms=1.0,
        release_ms=release_ms,
    )


def apply_noise_gate(
    audio: np.ndarray,
    sample_rate: float,
    *,
    threshold_db: float = -30.0,
    ratio: float = 8.0,
    attack_ms: float = 2.0,
    release_ms: float = 60.0,
) -> np.ndarray:
    x = _ensure_channels_first(audio)
    sr = float(sample_rate)
    attack = math.exp(-1.0 / (sr * max(attack_ms, 0.1) / 1000.0))
    release = math.exp(-1.0 / (sr * max(release_ms, 0.1) / 1000.0))
    thresh = 10.0 ** (float(threshold_db) / 20.0)
    mono = np.mean(x, axis=0)
    env = _envelope_follower(mono, attack, release)
    gate = np.ones_like(env)
    below = env < thresh
    gate[below] = 1.0 / max(float(ratio), 1.0)
    return (x * gate).astype(np.float32, copy=False)


def low_shelf_biquad_coeffs(
    freq_hz: float, gain_db: float, q: float, sample_rate: float = SAMPLE_RATE
) -> tuple[np.ndarray, np.ndarray]:
    w0 = 2 * np.pi * freq_hz / sample_rate
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / (2 * max(q, 0.1))
    A = 10.0 ** (gain_db / 40.0)
    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def high_shelf_biquad_coeffs(
    freq_hz: float, gain_db: float, q: float, sample_rate: float = SAMPLE_RATE
) -> tuple[np.ndarray, np.ndarray]:
    w0 = 2 * np.pi * freq_hz / sample_rate
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / (2 * max(q, 0.1))
    A = 10.0 ** (gain_db / 40.0)
    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return b, a


def apply_low_shelf(
    audio: np.ndarray, sample_rate: float, freq_hz: float, gain_db: float, q: float = 0.7
) -> np.ndarray:
    x = _ensure_channels_first(audio)
    b, a = low_shelf_biquad_coeffs(freq_hz, gain_db, q, sample_rate)
    return apply_iir_per_channel(x, sample_rate, b, a)


def apply_high_shelf(
    audio: np.ndarray, sample_rate: float, freq_hz: float, gain_db: float, q: float = 0.7
) -> np.ndarray:
    x = _ensure_channels_first(audio)
    b, a = high_shelf_biquad_coeffs(freq_hz, gain_db, q, sample_rate)
    return apply_iir_per_channel(x, sample_rate, b, a)


def apply_modulated_delay_effect(
    audio: np.ndarray,
    sample_rate: float,
    *,
    rate_hz: float = 1.0,
    depth: float = 0.25,
    centre_delay_ms: float = 7.0,
    feedback: float = 0.0,
    mix: float = 0.5,
) -> np.ndarray:
    """Chorus / flanger-style modulated delay per channel."""
    x = _ensure_channels_first(audio)
    n_ch, n_samp = x.shape
    sr = float(sample_rate)
    centre = max(1.0, float(centre_delay_ms) * sr / 1000.0)
    max_d = int(centre + depth * sr * 0.02 + 4)
    out = np.zeros_like(x)
    t = np.arange(n_samp, dtype=np.float64) / sr
    lfo = np.sin(2.0 * np.pi * float(rate_hz) * t)
    for ch in range(n_ch):
        buf = np.zeros(max_d + n_samp + 8, dtype=np.float32)
        y = 0.0
        for i in range(n_samp):
            delay = centre + depth * sr * 0.01 * (0.5 + 0.5 * lfo[i])
            idx = i - int(delay)
            tap = buf[idx] if idx >= 0 else 0.0
            xi = x[ch, i]
            buf[i] = xi + float(feedback) * tap
            y = (1.0 - mix) * xi + mix * tap
            out[ch, i] = y
    return out.astype(np.float32, copy=False)


def apply_phaser(
    audio: np.ndarray,
    sample_rate: float,
    *,
    rate_hz: float = 1.0,
    depth: float = 0.7,
    centre_frequency_hz: float = 1000.0,
    feedback: float = 0.5,
    mix: float = 0.5,
) -> np.ndarray:
    x = _ensure_channels_first(audio)
    n_ch, n_samp = x.shape
    sr = float(sample_rate)
    out = np.zeros_like(x)
    t = np.arange(n_samp, dtype=np.float64) / sr
    lfo = 0.5 + 0.5 * np.sin(2.0 * np.pi * float(rate_hz) * t)
    for ch in range(n_ch):
        z1 = 0.0
        prev = 0.0
        for i in range(n_samp):
            fc = float(centre_frequency_hz) * (0.5 + lfo[i] * float(depth))
            w0 = 2.0 * np.pi * fc / sr
            alpha = np.sin(w0) / 2.0
            b0, b1, a1 = 1.0 - alpha, -2.0 * np.cos(w0), -(1.0 - alpha)
            xi = x[ch, i]
            y = b0 * xi + b1 * prev + a1 * z1
            z1 = y
            prev = xi
            wet = xi + float(feedback) * y
            out[ch, i] = (1.0 - mix) * xi + mix * wet
    return out.astype(np.float32, copy=False)


def apply_pitch_shift_preserve_length(
    audio: np.ndarray, sample_rate: float, semitones: float
) -> np.ndarray:
    import librosa  # noqa: PLC0415

    x = _ensure_channels_first(audio)
    out_ch = []
    for ch in range(x.shape[0]):
        shifted = librosa.effects.pitch_shift(
            x[ch], sr=int(sample_rate), n_steps=float(semitones)
        )
        if shifted.shape[0] < x.shape[1]:
            shifted = np.pad(shifted, (0, x.shape[1] - shifted.shape[0]))
        else:
            shifted = shifted[: x.shape[1]]
        out_ch.append(shifted)
    return np.stack(out_ch, axis=0).astype(np.float32, copy=False)


def apply_resample_transpose(
    audio: np.ndarray, sample_rate: float, semitones: float
) -> np.ndarray:
    """Tape-style transpose: change pitch by resampling, keep output at input sample rate."""
    import librosa  # noqa: PLC0415

    x = _ensure_channels_first(audio)
    pitch_factor = 2 ** (float(semitones) / 12.0)
    if abs(pitch_factor - 1.0) < 1e-12:
        return x.astype(np.float32, copy=False)

    # Higher pitch => fewer samples at the same playback sample rate.
    resample_orig_sr = int(round(float(sample_rate) * pitch_factor))
    resample_target_sr = int(sample_rate)
    out_ch = []
    for ch in range(x.shape[0]):
        out_ch.append(
            librosa.resample(
                x[ch],
                orig_sr=resample_orig_sr,
                target_sr=resample_target_sr,
            )
        )
    min_len = min(len(c) for c in out_ch)
    stacked = np.stack([c[:min_len] for c in out_ch], axis=0)
    return stacked.astype(np.float32, copy=False)


def apply_master_normalization_chain(
    audio: np.ndarray, sample_rate: float
) -> np.ndarray:
    """HPF + LPF + gentle compression + limiter (replaces Pedalboard tail on apply_effect)."""
    x = _ensure_channels_first(audio)
    x = apply_steep_highpass(x, sample_rate, 40.0, steepness=0.0)
    x = apply_steep_lowpass(x, sample_rate, 18000.0, steepness=0.0)
    x = apply_compressor(
        x, sample_rate, threshold_db=-24.0, ratio=1.5, attack_ms=30.0, release_ms=200.0
    )
    x = apply_limiter(x, sample_rate, threshold_db=-4.0, release_ms=250.0)
    return x

