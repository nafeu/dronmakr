"""High-quality Reese bass generator for dronmakr. Root C1, 3-osc design, neuro movement."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, lfilter_zi

from pedalboard import Chorus, Phaser, Pedalboard

import dsp

SAMPLE_RATE = dsp.SAMPLE_RATE

# Root pitch: C1 (all iterations). Sub also at C1 for solid low-end.
C1_HZ = 32.7

# --- Ranges for randomization (root not randomized) ---------------------------

# Osc A/B detune: wider spread for classic full reese (less neurofunk-thin)
DETUNE_LEFT_RANGE = (-38.0, -10.0)   # cents
DETUNE_RIGHT_RANGE = (10.0, 38.0)

# Sub: 30% lower than before so reese can dominate
SUB_LEVEL_RANGE = (0.45, 0.75)
SUB_LEVEL_SCALE = 0.7  # additional 30% reduction
REESE_LEVEL_RANGE = (1.6, 2.6)

# Main filter: keep reese open (higher cutoffs) so it doesn't sound like a telephone
MAIN_CUTOFF_LOW_RANGE = (400.0, 1200.0)
MAIN_CUTOFF_HIGH_RANGE = (1800.0, 5500.0)
MAIN_RESONANCE_RANGE = (0.0, 0.12)
RESONANT_FILTER_PROB = 0.25

# Reese highpass: lower so more low-end body stays in the reese layer
REESE_HIGHPASS_RANGE = (35.0, 75.0)
# Parallel full-range "dry" reese path (no main filter) so bass stays full
REESE_DRY_MIX_RANGE = (0.35, 0.6)
BODY_MIX_RANGE = (0.2, 0.55)
SUB_LOWPASS_RANGE = (95.0, 145.0)

# LFO 1: filter movement – rate and depth vary a lot; sometimes almost no movement
LFO1_RATE_RANGE = (0.15, 5.0)
LFO1_DEPTH_RANGE = (0.0, 0.85)

# LFO 2: fine pitch – rate and amount vary
LFO2_RATE_RANGE = (0.05, 0.45)
LFO2_CENTS_RANGE = (0.5, 4.5)

# Distortion: much higher drive so reese is saturated and present, not thin/telephone
DIST_SOFT_DRIVE_RANGE = (2.5, 6.5)
DIST_HARD_DRIVE_RANGE = (2.0, 8.0)
DIST_HARD_MIX_RANGE = (0.25, 0.8)

# Post-EQ: gentle so we don't scoop the reese into telephone tone
POST_EQ_CUT_DB_RANGE = (-1.5, 0.5)
POST_EQ_BOOST_DB_RANGE = (0.0, 2.5)

# Oscillator wave types for the two root reese oscillators (randomized per osc)
WAVE_TYPES = ("saw", "tri", "square", "pulse")
PULSE_WIDTH_RANGE = (0.25, 0.75)

STEREO_WIDTH_RANGE = (0.2, 1.0)
HAAS_DELAY_MS_RANGE = (0.0, 22.0)
CHORUS_MIX_RANGE = (0.0, 0.55)
PHASER_MIX_RANGE = (0.0, 0.5)

BLOCK_SIZE = 512  # for LFO-modulated filter


def _parse_param_string(s: str | None) -> dict[str, str]:
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


def _resolve_choice(parsed: dict[str, str], key: str, choices: tuple[str, ...]) -> str:
    val = parsed.get(key)
    if _is_random(val):
        return random.choice(choices)
    v = val.lower()
    return v if v in choices else random.choice(choices)


def _normalize_audio(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    peak = float(np.max(np.abs(audio)) + 1e-12)
    return (audio * (target_peak / peak)).astype(np.float32)


# --- Config (root always C1) --------------------------------------------------


def parse_reese_config(
    sound: str | None = None,
    movement: str | None = None,
    distortion: str | None = None,
    fx: str | None = None,
    disable: str | None = None,
    sub_enabled: bool = False,
    neuro_eq: bool = False,
) -> dict[str, Any]:
    """Parse param strings. base_freq is always C1; other params randomized if _ or omitted.
    sub_enabled: if False, no sub (default); if True, sub is on unless disabled via disable=sub.
    neuro_eq: if True, apply neuro-style EQ/filter (highpass, LFO filter, body band, post-EQ). Default False = raw reese.
    """
    disabled = set()
    if disable:
        for d in disable.split(","):
            d = d.strip().lower()
            if d in ("sub", "fx", "movement", "distortion"):
                disabled.add(d)

    s = _parse_param_string(sound)
    m = _parse_param_string(movement)
    d = _parse_param_string(distortion)
    f = _parse_param_string(fx)

    # Sub: only when sub_enabled and not in disable
    sub_level = 0.0
    if sub_enabled and "sub" not in disabled:
        sub_level = _resolve_float(
            s, "sub_level", SUB_LEVEL_RANGE, 0.0, 1.0
        ) * SUB_LEVEL_SCALE
    reese_level = _resolve_float(s, "reese_level", REESE_LEVEL_RANGE, 0.8, 3.0)
    detune_left = _resolve_float(s, "detune_left", DETUNE_LEFT_RANGE, -40.0, 0.0)
    detune_right = _resolve_float(s, "detune_right", DETUNE_RIGHT_RANGE, 0.0, 40.0)
    wave_a = _resolve_choice(s, "wave_a", WAVE_TYPES)
    wave_b = _resolve_choice(s, "wave_b", WAVE_TYPES)
    pulse_width = _resolve_float(s, "pulse_width", PULSE_WIDTH_RANGE, 0.1, 0.9)

    # Filter: wide cutoff; resonance often low; sometimes plain LPF (fuller)
    main_cutoff_low = _resolve_float(
        m, "filter_cutoff_low", MAIN_CUTOFF_LOW_RANGE, 60.0, 800.0
    )
    main_cutoff_high = _resolve_float(
        m, "filter_cutoff_high", MAIN_CUTOFF_HIGH_RANGE, 350.0, 5000.0
    )
    main_cutoff_high = max(main_cutoff_high, main_cutoff_low + 150.0)
    main_resonance = _resolve_float(
        m, "filter_resonance", MAIN_RESONANCE_RANGE, 0.0, 0.4
    )
    use_resonant_filter = (
        _resolve_choice(m, "filter_type", ("resonant", "smooth")) == "resonant"
        if m.get("filter_type") and not _is_random(m.get("filter_type"))
        else random.random() < RESONANT_FILTER_PROB
    )

    highpass_hz = 0.0 if not neuro_eq else _resolve_float(
        m, "highpass_hz", REESE_HIGHPASS_RANGE, 25.0, 90.0
    )
    reese_dry_mix = _resolve_float(m, "reese_dry_mix", REESE_DRY_MIX_RANGE, 0.2, 0.75)
    body_mix = 0.0 if not neuro_eq else _resolve_float(m, "body_mix", BODY_MIX_RANGE, 0.0, 0.7)
    sub_lowpass_hz = _resolve_float(
        s, "sub_lowpass_hz", SUB_LOWPASS_RANGE, 70.0, 180.0
    )

    lfo1_rate = 0.0 if "movement" in disabled else _resolve_float(
        m, "lfo1_rate_hz", LFO1_RATE_RANGE, 0.05, 6.0
    )
    lfo1_depth = 0.0 if "movement" in disabled else _resolve_float(
        m, "lfo1_depth", LFO1_DEPTH_RANGE, 0.0, 1.0
    )
    lfo2_rate = 0.0 if "movement" in disabled else _resolve_float(
        m, "lfo2_rate_hz", LFO2_RATE_RANGE, 0.02, 0.6
    )
    lfo2_cents = 0.0 if "movement" in disabled else _resolve_float(
        m, "lfo2_cents", LFO2_CENTS_RANGE, 0.0, 5.5
    )

    drive_soft = 0.0 if "distortion" in disabled else _resolve_float(
        d, "drive_soft", DIST_SOFT_DRIVE_RANGE, 0.0, 6.0
    )
    drive_hard = 0.0 if "distortion" in disabled else _resolve_float(
        d, "drive_hard", DIST_HARD_DRIVE_RANGE, 0.0, 8.0
    )
    hard_mix = _resolve_float(d, "hard_mix", DIST_HARD_MIX_RANGE, 0.0, 0.9)

    post_eq_cut_db = _resolve_float(
        d, "post_eq_cut_db", POST_EQ_CUT_DB_RANGE, -6.0, 2.0
    )
    post_eq_boost_db = _resolve_float(
        d, "post_eq_boost_db", POST_EQ_BOOST_DB_RANGE, 0.0, 5.0
    )

    stereo_width = 0.0 if "fx" in disabled else _resolve_float(
        f, "stereo_width", STEREO_WIDTH_RANGE, 0.0, 1.0
    )
    haas_ms = 0.0 if "fx" in disabled else _resolve_float(
        f, "haas_ms", HAAS_DELAY_MS_RANGE, 0.0, 30.0
    )
    chorus_mix = 0.0 if "fx" in disabled else _resolve_float(
        f, "chorus_mix", CHORUS_MIX_RANGE, 0.0, 1.0
    )
    phaser_mix = 0.0 if "fx" in disabled else _resolve_float(
        f, "phaser_mix", PHASER_MIX_RANGE, 0.0, 1.0
    )

    return {
        "base_freq_hz": C1_HZ,
        "sub_freq_hz": C1_HZ,
        "sub_level": sub_level,
        "reese_level": reese_level,
        "detune_left": detune_left,
        "detune_right": detune_right,
        "wave_a": wave_a,
        "wave_b": wave_b,
        "pulse_width": pulse_width,
        "main_cutoff_low": main_cutoff_low,
        "main_cutoff_high": main_cutoff_high,
        "main_resonance": main_resonance,
        "use_resonant_filter": use_resonant_filter,
        "highpass_hz": highpass_hz,
        "reese_dry_mix": reese_dry_mix,
        "body_mix": body_mix,
        "sub_lowpass_hz": sub_lowpass_hz,
        "lfo1_rate_hz": lfo1_rate,
        "lfo1_depth": lfo1_depth,
        "lfo2_rate_hz": lfo2_rate,
        "lfo2_cents": lfo2_cents,
        "drive_soft": drive_soft,
        "drive_hard": drive_hard,
        "hard_mix": hard_mix,
        "post_eq_cut_db": post_eq_cut_db,
        "post_eq_boost_db": post_eq_boost_db,
        "stereo_width": stereo_width,
        "haas_ms": haas_ms,
        "chorus_mix": chorus_mix,
        "phaser_mix": phaser_mix,
        "neuro_eq": neuro_eq,
    }


# --- Synthesis ----------------------------------------------------------------


def _saw(phase: np.ndarray) -> np.ndarray:
    return 2.0 * (phase / (2 * np.pi) - np.floor(0.5 + phase / (2 * np.pi)))


def _tri(phase: np.ndarray) -> np.ndarray:
    p = np.mod(phase, 2 * np.pi)
    return np.where(p < np.pi, 2.0 * p / np.pi - 1.0, 3.0 - 2.0 * p / np.pi)


def _square(phase: np.ndarray) -> np.ndarray:
    return np.sign(np.sin(phase))


def _pulse(phase: np.ndarray, width: float) -> np.ndarray:
    """PWM: width in 0..1 (duty cycle)."""
    p = np.mod(phase, 2 * np.pi)
    thresh = np.clip(width, 0.05, 0.95) * 2 * np.pi
    return np.where(p < thresh, 1.0, -1.0)


def _osc(phase: np.ndarray, wave: str, pulse_width: float = 0.5) -> np.ndarray:
    if wave == "saw":
        return _saw(phase)
    if wave == "tri":
        return _tri(phase)
    if wave == "square":
        return _square(phase)
    if wave == "pulse":
        return _pulse(phase, pulse_width)
    return _saw(phase)


def _generate_three_osc(
    num_samples: int,
    c1_hz: float,
    detune_left_cents: float,
    detune_right_cents: float,
    lfo2_rate_hz: float,
    lfo2_cents: float,
    phase_offset_a: float,
    phase_offset_b: float,
    wave_a: str,
    wave_b: str,
    pulse_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Osc A: C1, detune_left, wave_a (saw/tri/square/pulse).
    Osc B: C1, detune_right, wave_b.
    LFO2: fine pitch modulation. Returns (reese_mono, sub_mono).
    """
    t = np.arange(num_samples, dtype=np.float64) / SAMPLE_RATE

    lfo2 = np.sin(2 * np.pi * lfo2_rate_hz * t) * lfo2_cents

    freq_a = c1_hz * (2.0 ** (detune_left_cents / 1200.0))
    freq_b = c1_hz * (2.0 ** (detune_right_cents / 1200.0))
    freq_a_t = freq_a * (2.0 ** (lfo2 / 1200.0))
    freq_b_t = freq_b * (2.0 ** (lfo2 / 1200.0))

    phase_a = 2 * np.pi * np.cumsum(freq_a_t) / SAMPLE_RATE + phase_offset_a
    phase_b = 2 * np.pi * np.cumsum(freq_b_t) / SAMPLE_RATE + phase_offset_b

    osc_a = _osc(phase_a, wave_a, pulse_width)
    osc_b = _osc(phase_b, wave_b, pulse_width)

    reese = (osc_a + osc_b) * 0.5
    reese /= np.max(np.abs(reese) + 1e-9)

    sub = np.sin(2 * np.pi * c1_hz * t)
    return reese.astype(np.float64), sub.astype(np.float64)


def generate_reese_sample(
    tempo: int = 170,
    bars: int = 4,
    output: str = "reese.wav",
    config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Generate neuro Reese at C1: 3 oscillators (A/B detuned saws + sub at C1),
    phase randomization, main filter on mids only; sub kept clean and never EQ'd.
    Post-EQ applied only to mids bus so sub and low-end stay present. Normalize at end.
    """
    if config is None:
        config = parse_reese_config()
    cfg = config

    beats_per_bar = 4
    duration_sec = (60.0 / tempo) * bars * beats_per_bar
    num_samples = int(duration_sec * SAMPLE_RATE)

    # Phase: full range so oscillators are more out of phase (wide, full reese)
    phase_offset_a = random.uniform(0, 2 * np.pi)
    phase_offset_b = random.uniform(0, 2 * np.pi)

    reese_raw, sub_raw = _generate_three_osc(
        num_samples,
        c1_hz=C1_HZ,
        detune_left_cents=cfg["detune_left"],
        detune_right_cents=cfg["detune_right"],
        lfo2_rate_hz=cfg["lfo2_rate_hz"],
        lfo2_cents=cfg["lfo2_cents"],
        phase_offset_a=phase_offset_a,
        phase_offset_b=phase_offset_b,
        wave_a=cfg["wave_a"],
        wave_b=cfg["wave_b"],
        pulse_width=cfg["pulse_width"],
    )

    # Sub: only when enabled (sub_level > 0); lowpass and level
    sub_lp = np.zeros(num_samples, dtype=np.float64)
    if cfg["sub_level"] > 0:
        sub_lp = dsp.apply_steep_lowpass(
            (sub_raw * cfg["sub_level"])[np.newaxis, :],
            SAMPLE_RATE,
            cfg["sub_lowpass_hz"],
        )[0]

    # Reese layer: highpass only when neuro_eq (otherwise keep full range for raw reese)
    if cfg.get("neuro_eq", False) and cfg["highpass_hz"] > 0:
        reese_hp = dsp.apply_steep_highpass(
            reese_raw[np.newaxis, :], SAMPLE_RATE, cfg["highpass_hz"]
        )[0]
    else:
        reese_hp = reese_raw.copy()

    # Body band and main filter only when neuro_eq (raw mode skips both)
    body_band = np.zeros(num_samples, dtype=np.float64)
    if cfg.get("neuro_eq", False):
        body_band = dsp.apply_steep_lowpass(
            reese_raw[np.newaxis, :], SAMPLE_RATE, 280.0
        )[0]
        body_band = dsp.apply_steep_highpass(
            body_band[np.newaxis, :], SAMPLE_RATE, 70.0
        )[0]

    # Main filter: LFO-modulated cutoff when neuro_eq; otherwise pass reese through
    if cfg.get("neuro_eq", False):
        t = np.arange(num_samples, dtype=np.float64) / SAMPLE_RATE
        lfo1 = 0.5 * (1.0 + np.sin(2 * np.pi * cfg["lfo1_rate_hz"] * t))
        cutoff_curve = cfg["main_cutoff_low"] + lfo1 * cfg["lfo1_depth"] * (
            cfg["main_cutoff_high"] - cfg["main_cutoff_low"]
        )

        zi = None
        reese_filtered = np.zeros(num_samples, dtype=np.float64)
        use_resonant = cfg["use_resonant_filter"] and cfg["main_resonance"] > 0.02

        for start in range(0, num_samples, BLOCK_SIZE):
            end = min(start + BLOCK_SIZE, num_samples)
            block = reese_hp[start:end].copy()
            mid = start + (end - start) // 2
            cutoff = float(np.clip(cutoff_curve[mid], 60.0, SAMPLE_RATE / 2 - 100))

            if use_resonant:
                q = 0.5 + cfg["main_resonance"] * 3.0
                b, a = dsp.resonant_lowpass_biquad_coeffs(cutoff, q, SAMPLE_RATE)
            else:
                nyq = 0.5 * SAMPLE_RATE
                fc = np.clip(cutoff / nyq, 0.01, 0.99)
                b, a = butter(4, fc, btype="low")
            zi_use = lfilter_zi(b, a) * block[0] if zi is None else zi
            block_f, zi = lfilter(b, a, block, zi=zi_use)
            reese_filtered[start:end] = block_f

        if cfg["body_mix"] > 0:
            body_band = body_band / (np.max(np.abs(body_band)) + 1e-9)
            reese_filtered = reese_filtered + cfg["body_mix"] * body_band
            reese_filtered = reese_filtered / (1.0 + cfg["body_mix"])
    else:
        reese_filtered = reese_hp.copy()

    # Strong saturation on the filtered reese path (so it’s present, not telephone)
    proc = reese_filtered.copy()
    if cfg["drive_soft"] > 0:
        proc = np.tanh(proc * cfg["drive_soft"])
    if cfg["drive_hard"] > 0 and cfg["hard_mix"] > 0:
        hard = np.clip(proc * cfg["drive_hard"], -1.0, 1.0)
        hard = np.tanh(hard * 2.0)
        proc = proc * (1.0 - cfg["hard_mix"]) + hard * cfg["hard_mix"]

    # Parallel dry path: full-range reese with only soft saturation, no main filter
    reese_dry = reese_hp.copy()
    if cfg["drive_soft"] > 0:
        reese_dry = np.tanh(reese_dry * cfg["drive_soft"] * 0.7)
    reese_dry = reese_dry / (np.max(np.abs(reese_dry)) + 1e-9)
    dry_mix = cfg["reese_dry_mix"]
    proc = proc * (1.0 - dry_mix) + reese_dry * dry_mix

    # Apply reese level
    proc = proc * cfg["reese_level"]

    # Stereo: mids layer with chorus/phaser, optional Haas
    mids_2ch = np.stack([proc, proc], axis=0).astype(np.float32)
    if cfg["haas_ms"] > 0:
        delay_samps = int(SAMPLE_RATE * cfg["haas_ms"] * 0.001)
        delay_samps = min(delay_samps, proc.size - 1)
        mids_2ch[1] = np.roll(mids_2ch[1], delay_samps)
        mids_2ch[1, :delay_samps] = 0.0

    fx_chain = []
    if cfg["phaser_mix"] > 0:
        fx_chain.append(
            Phaser(
                rate_hz=random.uniform(0.08, 0.35),
                depth=0.6,
                centre_frequency_hz=600.0,
                feedback=0.5,
                mix=cfg["phaser_mix"],
            )
        )
    if cfg["chorus_mix"] > 0:
        fx_chain.append(
            Chorus(
                rate_hz=random.uniform(0.12, 0.4),
                depth=0.25,
                centre_delay_ms=7.0,
                feedback=0.15,
                mix=cfg["chorus_mix"],
            )
        )
    if fx_chain:
        board = Pedalboard(fx_chain)
        mids_2ch = board(mids_2ch, SAMPLE_RATE)
    if cfg["stereo_width"] > 0:
        mid = 0.5 * (mids_2ch[0] + mids_2ch[1])
        side = 0.5 * (mids_2ch[0] - mids_2ch[1]) * cfg["stereo_width"]
        mids_2ch = np.stack([mid + side, mid - side], axis=0)

    # Post-EQ on mids only when neuro_eq (raw mode leaves reese untouched for user to shape)
    if cfg.get("neuro_eq", False):
        if cfg["post_eq_cut_db"] < -0.5:
            mids_2ch = dsp.apply_peaking_eq(
                mids_2ch, SAMPLE_RATE, 300.0, cfg["post_eq_cut_db"], q=0.8
            )
        if cfg["post_eq_boost_db"] > 0.3:
            mids_2ch = dsp.apply_peaking_eq(
                mids_2ch, SAMPLE_RATE, 1100.0, cfg["post_eq_boost_db"], q=0.7
            )

    # Mix: sub (when enabled) + mids stereo
    sub_stereo = np.stack([sub_lp, sub_lp], axis=0)
    mix = sub_stereo + mids_2ch

    # Normalize the whole sound to target peak so level is consistent
    out = _normalize_audio(mix.T, target_peak=0.9)
    sf.write(output, out, SAMPLE_RATE, subtype="PCM_16")

    params_used = {
        "base_freq_hz": C1_HZ,
        "sub_freq_hz": C1_HZ,
        "sub_level": cfg["sub_level"],
        "reese_level": cfg["reese_level"],
        "detune_left": cfg["detune_left"],
        "detune_right": cfg["detune_right"],
        "wave_a": cfg["wave_a"],
        "wave_b": cfg["wave_b"],
        "pulse_width": cfg["pulse_width"],
        "main_cutoff_low": cfg["main_cutoff_low"],
        "main_cutoff_high": cfg["main_cutoff_high"],
        "main_resonance": cfg["main_resonance"],
        "use_resonant_filter": cfg["use_resonant_filter"],
        "highpass_hz": cfg["highpass_hz"],
        "reese_dry_mix": cfg["reese_dry_mix"],
        "body_mix": cfg["body_mix"],
        "sub_lowpass_hz": cfg["sub_lowpass_hz"],
        "lfo1_rate_hz": cfg["lfo1_rate_hz"],
        "lfo1_depth": cfg["lfo1_depth"],
        "lfo2_rate_hz": cfg["lfo2_rate_hz"],
        "lfo2_cents": cfg["lfo2_cents"],
        "drive_soft": cfg["drive_soft"],
        "drive_hard": cfg["drive_hard"],
        "hard_mix": cfg["hard_mix"],
        "post_eq_cut_db": cfg["post_eq_cut_db"],
        "post_eq_boost_db": cfg["post_eq_boost_db"],
        "stereo_width": cfg["stereo_width"],
        "haas_ms": cfg["haas_ms"],
        "chorus_mix": cfg["chorus_mix"],
        "phaser_mix": cfg["phaser_mix"],
        "neuro_eq": cfg.get("neuro_eq", False),
    }
    return output, params_used
