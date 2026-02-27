"""Bass generator for dronmakr. Reese (C1, 3-osc, optional sub/neuro). Donk (percussive pitch-drop bass)."""

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

# Osc A/B detune: tighter spread to avoid phase issues while keeping width
DETUNE_LEFT_RANGE = (-24.0, -8.0)   # cents
DETUNE_RIGHT_RANGE = (8.0, 24.0)

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

# --- Donk (percussive pitch-drop bass) ---------------------------------------
DONK_BASE_FREQ_RANGE = (40.0, 80.0)
DONK_WAVE_TYPES = ("sine", "tri")
DONK_PITCH_START_SEMITONES_RANGE = (12.0, 24.0)
DONK_PITCH_DECAY_MS_RANGE = (5.0, 30.0)
DONK_AMP_ATTACK_MS_RANGE = (0.0, 5.0)
DONK_AMP_DECAY_MS_RANGE = (80.0, 200.0)
DONK_AMP_SUSTAIN_RANGE = (0.0, 0.2)
DONK_AMP_RELEASE_MS_RANGE = (30.0, 100.0)
DONK_CLICK_LEVEL_RANGE = (0.05, 0.1)
DONK_CLICK_DURATION_MS_RANGE = (1.0, 5.0)
DONK_SAT_DRIVE_RANGE = (1.2, 2.5)
DONK_SAT_MIX_RANGE = (0.2, 0.4)
DONK_LPF_CUTOFF_RANGE = (800.0, 3000.0)
DONK_LPF_RESONANCE_RANGE = (0.0, 0.15)


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


def _sound_flag(parsed: dict[str, str], key: str) -> bool:
    """True if key is present and value is not 0/false/off (for sound string flags like sub, neuro)."""
    if key not in parsed:
        return False
    return parsed[key].strip().lower() not in ("0", "false", "off")


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
) -> dict[str, Any]:
    """Parse param strings. base_freq is always C1; other params randomized if _ or omitted.
    In --sound: 'sub' = sub bass, 'neuro' = neuro EQ/filter. wave_a, wave_b = saw|tri|square|pulse (random if omitted).
    E.g. --sound "sub;neuro", --sound "wave_a:saw;wave_b:tri". Use --disable sub,fx,movement,distortion to force off.
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

    sub_enabled = _sound_flag(s, "sub") and "sub" not in disabled
    neuro_eq = _sound_flag(s, "neuro")

    # Sub: only when sub_enabled
    sub_level = 0.0
    if sub_enabled:
        sub_level = _resolve_float(
            s, "sub_level", SUB_LEVEL_RANGE, 0.0, 1.0
        ) * SUB_LEVEL_SCALE
    reese_level = _resolve_float(s, "reese_level", REESE_LEVEL_RANGE, 0.8, 3.0)
    detune_left = _resolve_float(s, "detune_left", DETUNE_LEFT_RANGE, -30.0, 0.0)
    detune_right = _resolve_float(s, "detune_right", DETUNE_RIGHT_RANGE, 0.0, 30.0)
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
        "sub_freq_hz": C1_HZ / 2.0,
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


def parse_donk_config(sound: str | None = None) -> dict[str, Any]:
    """Parse --sound for donk: base_freq, wave, pitch/amp envelope, click, sat, lpf. Use _ for random."""
    s = _parse_param_string(sound)
    base_freq = _resolve_float(
        s, "base_freq", DONK_BASE_FREQ_RANGE, 30.0, 100.0
    )
    wave = _resolve_choice(s, "wave", DONK_WAVE_TYPES)
    pitch_start_semitones = _resolve_float(
        s, "pitch_start_semitones", DONK_PITCH_START_SEMITONES_RANGE, 6.0, 30.0
    )
    pitch_decay_ms = _resolve_float(
        s, "pitch_decay_ms", DONK_PITCH_DECAY_MS_RANGE, 3.0, 50.0
    )
    amp_attack_ms = _resolve_float(
        s, "amp_attack_ms", DONK_AMP_ATTACK_MS_RANGE, 0.0, 10.0
    )
    amp_decay_ms = _resolve_float(
        s, "amp_decay_ms", DONK_AMP_DECAY_MS_RANGE, 50.0, 250.0
    )
    amp_sustain = _resolve_float(
        s, "amp_sustain", DONK_AMP_SUSTAIN_RANGE, 0.0, 0.4
    )
    amp_release_ms = _resolve_float(
        s, "amp_release_ms", DONK_AMP_RELEASE_MS_RANGE, 20.0, 120.0
    )
    click_enabled = _sound_flag(s, "click")
    click_level = _resolve_float(
        s, "click_level", DONK_CLICK_LEVEL_RANGE, 0.0, 0.2
    ) if click_enabled else 0.0
    click_duration_ms = _resolve_float(
        s, "click_duration_ms", DONK_CLICK_DURATION_MS_RANGE, 0.5, 8.0
    ) if click_enabled else 3.0
    sat_drive = _resolve_float(
        s, "sat_drive", DONK_SAT_DRIVE_RANGE, 1.0, 4.0
    )
    sat_mix = _resolve_float(
        s, "sat_mix", DONK_SAT_MIX_RANGE, 0.0, 0.6
    )
    lpf_cutoff = _resolve_float(
        s, "lpf_cutoff", DONK_LPF_CUTOFF_RANGE, 400.0, 5000.0
    )
    lpf_resonance = _resolve_float(
        s, "lpf_resonance", DONK_LPF_RESONANCE_RANGE, 0.0, 0.3
    )
    return {
        "base_freq": base_freq,
        "wave": wave,
        "pitch_start_semitones": pitch_start_semitones,
        "pitch_decay_ms": pitch_decay_ms,
        "amp_attack_ms": amp_attack_ms,
        "amp_decay_ms": amp_decay_ms,
        "amp_sustain": amp_sustain,
        "amp_release_ms": amp_release_ms,
        "click_enabled": click_enabled,
        "click_level": click_level,
        "click_duration_ms": click_duration_ms,
        "sat_drive": sat_drive,
        "sat_mix": sat_mix,
        "lpf_cutoff": lpf_cutoff,
        "lpf_resonance": lpf_resonance,
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
    Sub: one octave below C1 (C0). LFO2: fine pitch on reese. Returns (reese_mono, sub_mono).
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

    # Sub one octave below root (reese oscillators at c1_hz)
    sub_freq_hz = c1_hz / 2.0
    sub = np.sin(2 * np.pi * sub_freq_hz * t)
    return reese.astype(np.float64), sub.astype(np.float64)


def generate_reese_sample(
    tempo: int = 170,
    bars: int = 4,
    output: str = "reese.wav",
    config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Generate Reese at C1: 3 oscillators (A/B detuned + sub one octave down),
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
        "sub_freq_hz": C1_HZ / 2.0,
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


# --- Donk synthesis -----------------------------------------------------------


def _donk_osc(phase: np.ndarray, wave: str) -> np.ndarray:
    if wave == "tri":
        p = np.mod(phase, 2 * np.pi)
        return np.where(p < np.pi, 2.0 * p / np.pi - 1.0, 3.0 - 2.0 * p / np.pi)
    return np.sin(phase)


def _generate_donk_hit(cfg: dict[str, Any], num_samples: int) -> np.ndarray:
    """Single donk hit: pitch envelope (fast drop), amp envelope, optional click, saturation, LPF. Mono."""
    t = np.arange(num_samples, dtype=np.float64) / SAMPLE_RATE

    # Pitch envelope: start at +N semitones, exponential decay to 0 (base pitch)
    pitch_decay_sec = cfg["pitch_decay_ms"] / 1000.0
    tau = pitch_decay_sec / 4.6  # so exp(-decay_sec/tau) ≈ 0.01
    pitch_env = (cfg["pitch_start_semitones"] / 12.0) * np.exp(-t / tau)
    freq_hz = cfg["base_freq"] * (2.0 ** pitch_env)

    # Phase (reset at 0 each hit)
    phase_inc = 2.0 * np.pi * freq_hz / SAMPLE_RATE
    phase = np.cumsum(phase_inc)
    osc = _donk_osc(phase, cfg["wave"])

    # Amplitude envelope: attack -> decay -> sustain -> release
    attack_ms = cfg["amp_attack_ms"]
    decay_ms = cfg["amp_decay_ms"]
    sustain = cfg["amp_sustain"]
    release_ms = cfg["amp_release_ms"]
    attack_n = int(SAMPLE_RATE * attack_ms / 1000.0)
    decay_n = int(SAMPLE_RATE * decay_ms / 1000.0)
    release_n = int(SAMPLE_RATE * release_ms / 1000.0)

    amp_env = np.zeros(num_samples, dtype=np.float64)
    n = 0
    if attack_n > 0 and n < num_samples:
        end = min(n + attack_n, num_samples)
        amp_env[n:end] = np.linspace(0, 1, end - n)
        n = end
    if decay_n > 0 and n < num_samples:
        end = min(n + decay_n, num_samples)
        decay_len = end - n
        amp_env[n:end] = sustain + (1.0 - sustain) * np.exp(-np.arange(decay_len) * 5.0 / max(decay_len, 1))
        n = end
    if n < num_samples:
        release_len = min(release_n, num_samples - n)
        start_val = amp_env[n - 1] if n > 0 else sustain
        amp_env[n : n + release_len] = start_val * np.exp(-np.arange(release_len) * 5.0 / max(release_len, 1))
        if n + release_len < num_samples:
            amp_env[n + release_len :] = 0.0

    out = osc * amp_env

    # Optional click: short noise burst, highpass > 1 kHz, 5–10% level
    if cfg["click_enabled"] and cfg["click_level"] > 0:
        click_n = int(SAMPLE_RATE * cfg["click_duration_ms"] / 1000.0)
        click_n = min(click_n, num_samples)
        rng = np.random.default_rng()
        click = rng.standard_normal(click_n).astype(np.float64)
        click *= np.exp(-np.linspace(0, 5, click_n))  # short decay
        click = dsp.apply_steep_highpass(
            click[np.newaxis, :], SAMPLE_RATE, 1000.0
        )[0]
        peak = np.max(np.abs(click)) + 1e-9
        click = click / peak * cfg["click_level"]
        out[:click_n] = out[:click_n] + click

    # Mild saturation (tanh), mix 20–40%
    if cfg["sat_mix"] > 0 and cfg["sat_drive"] > 0:
        saturated = np.tanh(out * cfg["sat_drive"])
        out = out * (1.0 - cfg["sat_mix"]) + saturated * cfg["sat_mix"]

    # Lowpass 800 Hz – 3 kHz, low resonance
    if cfg["lpf_cutoff"] > 0:
        q = 0.5 + cfg["lpf_resonance"] * 2.0
        b, a = dsp.resonant_lowpass_biquad_coeffs(
            cfg["lpf_cutoff"], q, SAMPLE_RATE
        )
        out = lfilter(b, a, out)

    return out.astype(np.float64)


def generate_donk_sample(
    tempo: int = 120,
    bars: int = 4,
    output: str = "donk.wav",
    config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate a donk bass loop: percussive hits on the beat, pitch-drop, mono. No detune/LFO/sterero."""
    if config is None:
        config = parse_donk_config()
    cfg = config

    beats_per_bar = 4
    duration_sec = (60.0 / tempo) * bars * beats_per_bar
    num_samples_total = int(duration_sec * SAMPLE_RATE)
    beat_sec = 60.0 / tempo
    beat_samples = int(beat_sec * SAMPLE_RATE)

    # Hit length: attack + decay + release (enough for envelope to finish)
    hit_duration_sec = (
        (cfg["amp_attack_ms"] + cfg["amp_decay_ms"] + cfg["amp_release_ms"]) / 1000.0
    )
    hit_samples = min(int(hit_duration_sec * SAMPLE_RATE), beat_samples - 1)
    hit_samples = max(hit_samples, 1)

    out = np.zeros(num_samples_total, dtype=np.float64)
    num_hits = bars * beats_per_bar
    for i in range(num_hits):
        start = i * beat_samples
        if start + hit_samples > num_samples_total:
            break
        hit = _generate_donk_hit(cfg, hit_samples)
        out[start : start + hit_samples] += hit

    # Mono, normalize
    peak = np.max(np.abs(out)) + 1e-12
    out = (out / peak * 0.9).astype(np.float32)
    sf.write(output, out, SAMPLE_RATE, subtype="PCM_16")

    params_used = {**cfg}
    params_used["tempo"] = tempo
    params_used["bars"] = bars
    return output, params_used
