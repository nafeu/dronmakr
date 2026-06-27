import json
import os
import tempfile
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import fftconvolve

import dronmakr.audio.dsp as dsp
from dronmakr.audio.paulstretch import paulstretch
from dronmakr.core.utils import resolve_presets_index_path, with_process_drone_sample_prompt


def log_sample_processing_line(message: str) -> None:
    """Indented blue detail line aligned with CLI ■ processing sample blocks."""
    print(with_process_drone_sample_prompt(message))


def _read_channels_first(input_path: str) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(input_path, dtype="float32", always_2d=True)
    if data.ndim == 1:
        audio = np.stack([data, data], axis=0)
    else:
        audio = data.T
    return audio, sample_rate


def _write_channels_first(input_path: str, audio: np.ndarray, sample_rate: int) -> None:
    out = np.asarray(audio, dtype=np.float32)
    if out.ndim == 2 and out.shape[0] <= 8 and out.shape[0] < out.shape[1]:
        out = out.T
    parent = os.path.dirname(input_path) or "."
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp.wav", dir=parent)
    os.close(fd)
    try:
        sf.write(tmp_path, out, sample_rate, subtype="PCM_16")
        os.replace(tmp_path, input_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def apply_normalization(input_path):
    audio, sample_rate = _read_channels_first(input_path)
    processed = dsp.apply_master_normalization_chain(audio, sample_rate)
    _write_channels_first(input_path, processed, sample_rate)


def _finalize_fx_processed_audio(
    processed: np.ndarray,
    *,
    target_samples: int | None = None,
    peak_limit: float = 0.99,
) -> np.ndarray:
    """Prepare plug-in FX output for sample replacement without export mastering."""
    out = np.asarray(processed, dtype=np.float32)
    if out.ndim == 1:
        out = np.column_stack([out, out])
    if target_samples is not None and out.shape[0] > target_samples:
        out = out[:target_samples]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > float(peak_limit):
        out = out * (float(peak_limit) / peak)
    return out


def apply_plugin_patch_to_sample(input_path, patch_name, presets_path=None):
    """Run the selected saved FX patch or FX chain through a WAV file in place."""
    from dronmakr.audio.audio_host import render_wav_through_fx_paths

    patch_name = (patch_name or "").strip()
    if not patch_name:
        raise ValueError("Plugin patch name is required.")

    presets_path = presets_path or resolve_presets_index_path()
    if not presets_path:
        raise FileNotFoundError(
            "config/presets.json does not exist — save an FX patch from Generate Samples first."
        )

    from dronmakr.audio.audio_worker import delegate_apply_plugin_patch_if_needed

    if delegate_apply_plugin_patch_if_needed(
        os.path.abspath(input_path),
        patch_name,
        os.path.abspath(presets_path),
    ):
        return

    input_data, input_sample_rate = sf.read(input_path, dtype="float32", always_2d=True)
    target_samples = int(input_data.shape[0])

    with open(presets_path, encoding="utf-8") as handle:
        presets = json.load(handle)
    if not isinstance(presets, list):
        raise ValueError(f"{presets_path} must contain a JSON array of preset objects.")

    effect_preset = next(
        (
            preset
            for preset in presets
            if isinstance(preset, dict)
            and preset.get("type") in ("effect", "effect_chain")
            and (preset.get("name") or "").strip() == patch_name
        ),
        None,
    )
    if effect_preset is None:
        raise ValueError(f"No saved FX patch named '{patch_name}'.")

    from dronmakr.generate.generate_sample import effect_slot_entries

    slot_list = effect_slot_entries(effect_preset)
    if not slot_list:
        raise ValueError(f"FX patch '{patch_name}' has no processors to load.")

    fx_specs = [(step["plugin_path"], step.get("preset_path") or "") for step in slot_list]
    processed, sample_rate = render_wav_through_fx_paths(
        input_path,
        fx_specs,
        tail_sec=3.0,
    )
    if processed.size == 0 or not np.isfinite(processed).all():
        raise RuntimeError(f"Plugin patch '{patch_name}' produced empty or invalid audio.")
    if sample_rate != input_sample_rate:
        raise ValueError(
            f"Sample rate changed during FX render ({input_sample_rate} → {sample_rate})."
        )

    out = _finalize_fx_processed_audio(processed, target_samples=target_samples)
    sf.write(input_path, out, sample_rate, subtype="PCM_16")
    log_sample_processing_line(f"Applied plugin patch '{patch_name}'")


def trim_sample_start(input_path, start_time_s):
    """Trims the start of the audio file at `start_time_s` and overwrites the file."""
    audio, sample_rate = sf.read(input_path)
    start_sample = int(start_time_s * sample_rate)

    if start_sample < 0 or start_sample >= len(audio):
        raise ValueError("Start time is out of bounds.")

    trimmed_audio = audio[start_sample:]
    sf.write(input_path, trimmed_audio, sample_rate)
    log_sample_processing_line(f"Trimmed start at {start_time_s}s")


def trim_sample_end(input_path, end_time_s):
    """Trims the end of the audio file at `end_time_s` and overwrites the file."""
    audio, sample_rate = sf.read(input_path)
    end_sample = int(end_time_s * sample_rate)

    if end_sample <= 0 or end_sample > len(audio):
        raise ValueError("End time is out of bounds.")

    trimmed_audio = audio[:end_sample]
    sf.write(input_path, trimmed_audio, sample_rate)
    log_sample_processing_line(f"Trimmed end at {end_time_s}s")


def fade_sample_start(input_path, fade_in_time_s):
    """Applies an ease-in-out fade-in over `fade_in_time_s` and overwrites the file."""
    audio, sample_rate = sf.read(input_path)
    fade_samples = int(fade_in_time_s * sample_rate)

    if fade_samples > len(audio):
        fade_samples = len(audio)  # Prevent fade being longer than audio

    # Ease-in-out fade curve (mono)
    fade_curve = (1 - np.cos(np.linspace(0, np.pi, fade_samples))) / 2

    # Ensure fade curve applies to all channels
    if len(audio.shape) > 1:  # Stereo or multi-channel
        fade_curve = fade_curve[:, np.newaxis]  # Reshape to (samples, 1)

    # Apply fade
    audio[:fade_samples] *= fade_curve

    sf.write(input_path, audio, sample_rate)
    log_sample_processing_line(f"Applied fade-in ({fade_in_time_s}s)")


def fade_sample_end(input_path, fade_out_time_s):
    """Applies an ease-in-out fade-out over `fade_out_time_s` and overwrites the file."""
    audio, sample_rate = sf.read(input_path)
    fade_samples = int(fade_out_time_s * sample_rate)

    if fade_samples > len(audio):
        fade_samples = len(audio)  # Prevent fade being longer than audio

    # Ease-in-out fade curve (mono)
    fade_curve = (1 - np.cos(np.linspace(0, np.pi, fade_samples))) / 2

    # Ensure fade curve applies to all channels
    if len(audio.shape) > 1:  # Stereo or multi-channel
        fade_curve = fade_curve[:, np.newaxis]  # Reshape to (samples, 1)

    # Apply fade
    audio[-fade_samples:] *= fade_curve[::-1]

    sf.write(input_path, audio, sample_rate)
    log_sample_processing_line(f"Applied fade-out ({fade_out_time_s}s)")


def pad_sample(input_path, amount, side="after"):
    """Append or prepend silence equal to ``amount`` × current sample length (0–2 = 0–200%)."""
    amount = float(amount)
    if amount <= 0:
        return
    side = str(side or "after").strip().lower()
    if side not in ("before", "after"):
        raise ValueError("padding side must be 'before' or 'after'")
    if amount > 2:
        raise ValueError("padding amount must be between 0 and 2")

    audio, sample_rate = sf.read(input_path)
    pad_samples = int(round(len(audio) * amount))
    if pad_samples <= 0:
        return

    if audio.ndim == 1:
        silence = np.zeros(pad_samples, dtype=audio.dtype)
    else:
        silence = np.zeros((pad_samples, audio.shape[1]), dtype=audio.dtype)

    if side == "before":
        padded = np.concatenate([silence, audio], axis=0)
    else:
        padded = np.concatenate([audio, silence], axis=0)

    sf.write(input_path, padded, sample_rate)
    log_sample_processing_line(
        f"Applied padding ({amount * 100:.0f}% {side})"
    )


def increase_sample_gain(input_path, db):
    """Increases the gain of the audio file by `db` decibels and overwrites the file."""
    audio, sample_rate = sf.read(input_path)

    gain_factor = 10 ** (db / 20)  # Convert dB to linear scale
    audio *= gain_factor

    sf.write(input_path, audio, sample_rate)
    log_sample_processing_line(f"Gain +{db} dB")


def decrease_sample_gain(input_path, db):
    """Decreases the gain of the audio file by `db` decibels and overwrites the file."""
    audio, sample_rate = sf.read(input_path)

    gain_factor = 10 ** (-db / 20)  # Convert dB to linear scale
    audio *= gain_factor

    sf.write(input_path, audio, sample_rate)
    log_sample_processing_line(f"Gain -{db} dB")


def normalize_sample(input_path, target_peak_db=-1.0):
    """Normalize sample peak to target dBFS and overwrite file."""
    audio, sample_rate = sf.read(input_path)
    if audio.size == 0:
        raise ValueError("Audio file is empty")
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-12:
        return
    target_linear = 10 ** (float(target_peak_db) / 20.0)
    scale = target_linear / peak
    normalized = np.clip(audio * scale, -1.0, 1.0)
    sf.write(input_path, normalized, sample_rate)
    log_sample_processing_line(f"Normalize to {target_peak_db} dBFS peak")


def apply_noise_gate_to_sample(
    input_path,
    threshold_db=-30.0,
    ratio=8.0,
    attack_ms=2.0,
    release_ms=60.0,
):
    """Apply noise gate and overwrite file."""
    audio, sample_rate = _read_channels_first(input_path)
    processed = dsp.apply_noise_gate(
        audio,
        sample_rate,
        threshold_db=threshold_db,
        ratio=ratio,
        attack_ms=attack_ms,
        release_ms=release_ms,
    )
    _write_channels_first(input_path, processed, sample_rate)
    log_sample_processing_line(
        "Applied noise gate "
        f"(threshold={threshold_db} dB, attack={attack_ms} ms, release={release_ms} ms)"
    )


def reverse_sample(input_path):
    """Reverses the audio file and overwrites the existing file."""
    audio, sample_rate = sf.read(input_path)

    # Reverse the audio along the time axis
    reversed_audio = audio[::-1]

    sf.write(input_path, reversed_audio, sample_rate)
    log_sample_processing_line("Applied reverse")


def double_loop_sample(input_path):
    """
    Concatenate the file with itself (2× duration), sample-accurate, no fades.

    For a one-bar (or any) loop that already wraps cleanly from last sample back
    to the first, the join at the midpoint matches the natural loop point twice.
    """
    audio, sample_rate = sf.read(input_path)
    if audio.size == 0:
        raise ValueError("Audio file is empty")

    doubled = np.concatenate([audio, audio], axis=0)
    sf.write(input_path, doubled, sample_rate)
    log_sample_processing_line("Doubled length (two copies back-to-back)")


def apply_time_stretch_simple(input_path, stretch_factor):
    """
    Simple resampling-based time stretch.
    - stretch_factor < 1.0: shorter/faster, higher pitch
    - stretch_factor > 1.0: longer/slower, lower pitch
    """
    stretch_factor = float(stretch_factor)
    if stretch_factor <= 0:
        raise ValueError("stretch_factor must be greater than 0")

    audio, sample_rate = sf.read(input_path)
    if audio.size == 0:
        raise ValueError("Audio file is empty")

    original_len = audio.shape[0]
    new_len = max(1, int(round(original_len * stretch_factor)))

    src_positions = np.linspace(0, original_len - 1, num=original_len)
    dst_positions = np.linspace(0, original_len - 1, num=new_len)

    if audio.ndim == 1:
        stretched = np.interp(dst_positions, src_positions, audio)
    else:
        channels = [
            np.interp(dst_positions, src_positions, audio[:, ch])
            for ch in range(audio.shape[1])
        ]
        stretched = np.stack(channels, axis=1)

    sf.write(input_path, stretched, sample_rate)
    log_sample_processing_line(f"Applied time stretch (factor={stretch_factor})")


def apply_pitch_shift_preserve_length(input_path, semitones):
    """
    Pitch shift while preserving duration (length) using librosa.
    This keeps tempo locked while still allowing artifacts in a musically useful way.
    """
    semitones = float(semitones)
    if semitones == 0:
        return

    audio, sample_rate = _read_channels_first(input_path)

    if audio.size == 0:
        raise ValueError("Audio file is empty")

    shifted = dsp.apply_pitch_shift_preserve_length(audio, sample_rate, semitones)

    # Keep exact frame length for tight sequencer sync.
    original_frames = audio.shape[1]
    shifted_frames = shifted.shape[1]
    if shifted_frames > original_frames:
        shifted = shifted[:, :original_frames]
    elif shifted_frames < original_frames:
        shifted = np.pad(shifted, ((0, 0), (0, original_frames - shifted_frames)))

    # Deterministic transient-anchor alignment:
    # align first detected transient only (no adaptive cross-correlation),
    # which avoids left/right jitter between repeated operations.
    def _mono_mix(channels_first_audio: np.ndarray) -> np.ndarray:
        if channels_first_audio.shape[0] == 1:
            return channels_first_audio[0]
        return np.mean(channels_first_audio, axis=0)

    def _shift_channels_first(channels_first_audio: np.ndarray, lag_samples: int) -> np.ndarray:
        out = np.zeros_like(channels_first_audio)
        if lag_samples > 0:
            # signal is delayed -> shift left
            if lag_samples < channels_first_audio.shape[1]:
                out[:, :-lag_samples] = channels_first_audio[:, lag_samples:]
        elif lag_samples < 0:
            # signal is early -> shift right
            lead = -lag_samples
            if lead < channels_first_audio.shape[1]:
                out[:, lead:] = channels_first_audio[:, :-lead]
        else:
            out[:] = channels_first_audio
        return out

    def _first_transient_index(mono: np.ndarray, sr: int) -> int:
        if mono.size == 0:
            return 0
        abs_mono = np.abs(mono.astype(np.float32))
        peak = float(np.max(abs_mono))
        if peak <= 1e-8:
            return 0
        # Use an amplitude gate near the beginning to anchor onset consistently.
        gate = max(peak * 0.2, 1e-5)
        # Search first 500ms for the anchor transient.
        search_len = min(abs_mono.shape[0], int(sr * 0.5))
        idxs = np.where(abs_mono[:search_len] >= gate)[0]
        if idxs.size == 0:
            return 0
        return int(idxs[0])

    ref_mono = _mono_mix(audio).astype(np.float32)
    shifted_mono = _mono_mix(shifted).astype(np.float32)
    if ref_mono.size > 0 and shifted_mono.size > 0:
        ref_anchor = _first_transient_index(ref_mono, sample_rate)
        shifted_anchor = _first_transient_index(shifted_mono, sample_rate)
        lag = shifted_anchor - ref_anchor
        # Safety clamp to avoid large accidental jumps.
        max_lag = int(sample_rate * 0.03)  # 30ms max correction
        if lag > max_lag:
            lag = max_lag
        elif lag < -max_lag:
            lag = -max_lag
        shifted = _shift_channels_first(shifted, lag)

    _write_channels_first(input_path, shifted, sample_rate)
    log_sample_processing_line(f"Applied pitch shift ({semitones:+g} semitones)")


def apply_transpose_pitch_by_resampling_inplace(input_path: str, semitones: float) -> None:
    """Resample transpose: shifts pitch by changing nominal sample rate (duration follows)."""
    semitones = float(semitones)
    if semitones == 0:
        return

    audio, sample_rate = _read_channels_first(input_path)
    processed_audio = dsp.apply_resample_transpose(audio, sample_rate, semitones)
    _write_channels_first(input_path, processed_audio, sample_rate)
    log_sample_processing_line(
        f"Applied resampling transpose ({semitones:+g} semitones)"
    )


def apply_paulstretch_to_sample(
    input_path: str, stretch: float = 8.0, window_size: float = 2.5
) -> None:
    stretch = float(stretch)
    window_size = float(window_size)
    out_dir = os.path.dirname(os.path.abspath(input_path))
    if not out_dir:
        out_dir = os.getcwd()

    fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=out_dir)
    os.close(fd)
    try:
        paulstretch(
            input_path,
            output_path=tmp_path,
            stretch=stretch,
            window_size=window_size,
            show_logs=False,
        )
        os.replace(tmp_path, input_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    log_sample_processing_line(
        f"Applied Paulstretch (stretch={stretch}, window={window_size}s)"
    )


def apply_reverb_to_sample(
    input_path,
    wet_level=0.35,
    reverb_length_sec=0.7,
    decay_sec=0.5,
    reverb_highpass_hz=100.0,
    *,
    early_reflections=None,
    tail_diffusion=None,
    early_diffuse=None,
):
    """
    Apply high-quality convolution reverb (offline). Uses a generated IR with
    early reflections + dense decay tail. Optional highpass on the reverb
    only (reverb_highpass_hz > 0) to avoid low-end phase clash on drums;
    set to 0 to leave low-end in the reverb.
    """
    audio, sample_rate = _read_channels_first(input_path)
    n_channels, n_samples = audio.shape

    ir_kwargs = {
        "length_sec": reverb_length_sec,
        "decay_sec": decay_sec,
        "highpass_cutoff_hz": reverb_highpass_hz,
    }
    if early_reflections is not None:
        ir_kwargs["early_reflections"] = early_reflections
    if tail_diffusion is not None:
        ir_kwargs["tail_diffusion"] = tail_diffusion
    if early_diffuse is not None:
        ir_kwargs["early_diffuse"] = early_diffuse

    ir = dsp.make_reverb_ir(sample_rate, **ir_kwargs)
    wet = np.zeros_like(audio)
    for ch in range(n_channels):
        wet[ch] = fftconvolve(audio[ch], ir, mode="full")[:n_samples]
    wet_peak = np.max(np.abs(wet)) + 1e-12
    dry_peak = np.max(np.abs(audio)) + 1e-12
    wet = wet * (dry_peak / wet_peak)
    mix = wet_level * wet + (1.0 - wet_level) * audio
    _write_channels_first(input_path, mix, sample_rate)
    msg = f"Applied convolution reverb (wet={wet_level}, IR {reverb_length_sec}s"
    if reverb_highpass_hz > 0:
        msg += f", HPF {reverb_highpass_hz} Hz"
    msg += ")"
    log_sample_processing_line(msg)


def apply_delay_to_sample(
    input_path,
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
):
    """
    Tempo-synced or manual-ms feedback delay (fractional delay line, filtered feedback).
    See ``dsp.apply_feedback_delay`` for stereo/ping-pong behaviour.
    """
    audio, sample_rate = _read_channels_first(input_path)
    processed = dsp.apply_feedback_delay(
        audio,
        float(sample_rate),
        time_mode=time_mode,
        bpm=float(bpm),
        division=str(division),
        delay_ms=float(delay_ms),
        delay_offset_ms=float(delay_offset_ms),
        stereo_spread_ms=float(stereo_spread_ms),
        feedback=float(feedback),
        mix=float(mix),
        ping_pong=bool(ping_pong),
        stereo_width=float(stereo_width),
        input_crossfeed=float(input_crossfeed),
        feedback_lowpass_hz=float(feedback_lowpass_hz),
        feedback_highpass_hz=float(feedback_highpass_hz),
        max_delay_sec=float(max_delay_sec),
    )
    _write_channels_first(input_path, processed, sample_rate)
    bits = [f"{time_mode}"]
    if str(time_mode).lower() == "sync":
        bits.append(f"{division} @ {float(bpm):g} BPM")
    else:
        bits.append(f"{delay_ms:g} ms")
    bits.append(f"fb={feedback:.2f} mix={mix:.2f}")
    log_sample_processing_line("Applied delay (" + ", ".join(bits) + ")")


def apply_reverb_room_to_sample(input_path):
    """Small room reverb: short length and decay."""
    apply_reverb_to_sample(
        input_path,
        wet_level=0.3,
        reverb_length_sec=0.4,
        decay_sec=0.3,
        reverb_highpass_hz=100.0,
    )


def apply_reverb_bedroom_to_sample(input_path):
    """Very mild short room reverb, like reflections in a small bedroom."""
    apply_reverb_to_sample(
        input_path,
        wet_level=0.18,
        reverb_length_sec=0.25,
        decay_sec=0.18,
        reverb_highpass_hz=120.0,
    )


def apply_reverb_hall_to_sample(input_path):
    """Hall reverb: medium size and decay."""
    apply_reverb_to_sample(
        input_path,
        wet_level=0.35,
        reverb_length_sec=1.0,
        decay_sec=0.8,
        reverb_highpass_hz=100.0,
    )


def apply_reverb_large_to_sample(input_path):
    """Large reverb: long length and decay."""
    apply_reverb_to_sample(
        input_path,
        wet_level=0.4,
        reverb_length_sec=1.8,
        decay_sec=1.5,
        reverb_highpass_hz=100.0,
    )


def apply_reverb_amphitheatre_to_sample(input_path):
    """Huge amphitheatre-style reverb with an expansive tail."""
    apply_reverb_to_sample(
        input_path,
        wet_level=0.48,
        reverb_length_sec=3.5,
        decay_sec=3.0,
        reverb_highpass_hz=80.0,
    )


def apply_reverb_space_to_sample(input_path):
    """Near-endless ambient reverb tail."""
    apply_reverb_to_sample(
        input_path,
        wet_level=0.62,
        reverb_length_sec=8.0,
        decay_sec=7.0,
        reverb_highpass_hz=50.0,
    )


def apply_reverb_void_to_sample(input_path):
    """Denser, longer diffuse tail than Space (ambient wash)."""
    apply_reverb_to_sample(
        input_path,
        wet_level=0.74,
        reverb_length_sec=14.0,
        decay_sec=12.0,
        reverb_highpass_hz=38.0,
        early_reflections=11,
        tail_diffusion=0.93,
        early_diffuse=True,
    )


def apply_reverb_distant_to_sample(input_path):
    """Sparse early field with a smoother late bloom."""
    apply_reverb_to_sample(
        input_path,
        wet_level=0.53,
        reverb_length_sec=11.5,
        decay_sec=10.5,
        reverb_highpass_hz=62.0,
        early_reflections=3,
        tail_diffusion=0.95,
        early_diffuse=False,
    )


def apply_distortion_to_sample(input_path, drive_db=6.0):
    """Apply distortion and overwrite the file."""
    audio, sample_rate = _read_channels_first(input_path)
    processed = dsp.apply_distortion(audio, sample_rate, drive_db=drive_db)
    _write_channels_first(input_path, processed, sample_rate)
    log_sample_processing_line(f"Applied distortion (drive={drive_db} dB)")


def apply_compress_to_sample(
    input_path,
    threshold_db=-20.0,
    ratio=10.0,
    attack_ms=3.0,
    release_ms=80.0,
):
    """Apply compression and overwrite the file."""
    audio, sample_rate = _read_channels_first(input_path)
    processed = dsp.apply_compressor(
        audio,
        sample_rate,
        threshold_db=threshold_db,
        ratio=ratio,
        attack_ms=attack_ms,
        release_ms=release_ms,
    )
    _write_channels_first(input_path, processed, sample_rate)
    log_sample_processing_line("Applied compression")


def apply_compress_mild_to_sample(input_path):
    """Light compression."""
    apply_compress_to_sample(
        input_path,
        threshold_db=-14.0,
        ratio=2.5,
        attack_ms=10.0,
        release_ms=140.0,
    )


def apply_compress_medium_to_sample(input_path):
    """Current/default compression behavior."""
    apply_compress_to_sample(input_path)


def apply_compress_heavy_to_sample(input_path):
    """Heavy compression."""
    apply_compress_to_sample(
        input_path,
        threshold_db=-28.0,
        ratio=16.0,
        attack_ms=1.5,
        release_ms=60.0,
    )


def apply_overdrive_mids_to_sample(
    input_path,
    drive_db=14.0,
    highpass_hz=200.0,
    lowpass_hz=4000.0,
):
    """Overdrive focused on mids: HPF to cut lows, distortion, LPF to cut highs."""
    audio, sample_rate = _read_channels_first(input_path)
    processed = dsp.apply_steep_highpass(audio, sample_rate, highpass_hz, steepness=0.0)
    processed = dsp.apply_distortion(processed, sample_rate, drive_db=drive_db)
    processed = dsp.apply_steep_lowpass(processed, sample_rate, lowpass_hz, steepness=0.0)
    _write_channels_first(input_path, processed, sample_rate)
    log_sample_processing_line("Applied mid-focused overdrive")


def apply_overdrive_mild_to_sample(input_path):
    """Mild mid-focused overdrive."""
    apply_overdrive_mids_to_sample(
        input_path,
        drive_db=8.0,
        highpass_hz=150.0,
        lowpass_hz=5500.0,
    )


def apply_overdrive_medium_to_sample(input_path):
    """Current/default overdrive behavior."""
    apply_overdrive_mids_to_sample(input_path)


def apply_overdrive_heavy_to_sample(input_path):
    """Heavy mid-focused overdrive."""
    apply_overdrive_mids_to_sample(
        input_path,
        drive_db=20.0,
        highpass_hz=300.0,
        lowpass_hz=3000.0,
    )


def apply_chorus_to_sample(
    input_path,
    rate_hz=1.0,
    depth=0.25,
    centre_delay_ms=7.0,
    feedback=0.0,
    mix=0.5,
):
    """Apply chorus and overwrite the file."""
    audio, sample_rate = _read_channels_first(input_path)
    processed = dsp.apply_modulated_delay_effect(
        audio,
        sample_rate,
        rate_hz=rate_hz,
        depth=depth,
        centre_delay_ms=centre_delay_ms,
        feedback=feedback,
        mix=mix,
    )
    _write_channels_first(input_path, processed, sample_rate)
    log_sample_processing_line("Applied chorus")


def apply_chorus_mild_to_sample(input_path):
    apply_chorus_to_sample(
        input_path,
        rate_hz=0.7,
        depth=0.15,
        centre_delay_ms=8.0,
        feedback=0.0,
        mix=0.3,
    )


def apply_chorus_medium_to_sample(input_path):
    apply_chorus_to_sample(input_path)


def apply_chorus_heavy_to_sample(input_path):
    apply_chorus_to_sample(
        input_path,
        rate_hz=1.4,
        depth=0.45,
        centre_delay_ms=6.0,
        feedback=0.2,
        mix=0.75,
    )


def apply_flanger_to_sample(
    input_path,
    rate_hz=0.5,
    depth=0.4,
    centre_delay_ms=2.0,
    feedback=0.3,
    mix=0.5,
):
    """Apply flanger-style modulated delay and overwrite the file."""
    audio, sample_rate = _read_channels_first(input_path)
    processed = dsp.apply_modulated_delay_effect(
        audio,
        sample_rate,
        rate_hz=rate_hz,
        depth=depth,
        centre_delay_ms=centre_delay_ms,
        feedback=feedback,
        mix=mix,
    )
    _write_channels_first(input_path, processed, sample_rate)
    log_sample_processing_line("Applied flanger")


def apply_flanger_mild_to_sample(input_path):
    apply_flanger_to_sample(
        input_path,
        rate_hz=0.35,
        depth=0.2,
        centre_delay_ms=2.5,
        feedback=0.15,
        mix=0.35,
    )


def apply_flanger_medium_to_sample(input_path):
    apply_flanger_to_sample(input_path)


def apply_flanger_heavy_to_sample(input_path):
    apply_flanger_to_sample(
        input_path,
        rate_hz=0.9,
        depth=0.65,
        centre_delay_ms=1.2,
        feedback=0.55,
        mix=0.75,
    )


def apply_phaser_to_sample(
    input_path,
    rate_hz=1.0,
    depth=0.7,
    centre_frequency_hz=1000.0,
    feedback=0.5,
    mix=0.5,
):
    """Apply phaser and overwrite the file."""
    audio, sample_rate = _read_channels_first(input_path)
    processed = dsp.apply_phaser(
        audio,
        sample_rate,
        rate_hz=rate_hz,
        depth=depth,
        centre_frequency_hz=centre_frequency_hz,
        feedback=feedback,
        mix=mix,
    )
    _write_channels_first(input_path, processed, sample_rate)
    log_sample_processing_line("Applied phaser")


def apply_phaser_mild_to_sample(input_path):
    apply_phaser_to_sample(
        input_path,
        rate_hz=0.7,
        depth=0.45,
        centre_frequency_hz=900.0,
        feedback=0.25,
        mix=0.35,
    )


def apply_phaser_medium_to_sample(input_path):
    apply_phaser_to_sample(input_path)


def apply_phaser_heavy_to_sample(input_path):
    apply_phaser_to_sample(
        input_path,
        rate_hz=1.4,
        depth=0.95,
        centre_frequency_hz=1300.0,
        feedback=0.7,
        mix=0.75,
    )


def apply_distortion_mild_to_sample(input_path):
    apply_distortion_to_sample(input_path, drive_db=3.0)


def apply_distortion_medium_to_sample(input_path):
    apply_distortion_to_sample(input_path, drive_db=6.0)


def apply_distortion_heavy_to_sample(input_path):
    apply_distortion_to_sample(input_path, drive_db=11.0)


def apply_lowpass_to_sample(input_path, cutoff_hz=1200, **_kwargs):
    """Apply steep low-pass; optional resonance / steepness in _kwargs."""
    resonance = float(_kwargs.get("resonance", 0.0))
    steepness = float(_kwargs.get("steepness", 0.72))
    audio, sample_rate = _read_channels_first(input_path)
    processed = dsp.apply_steep_lowpass(
        audio,
        sample_rate,
        cutoff_hz,
        resonance=resonance,
        steepness=steepness,
    )
    _write_channels_first(input_path, processed, sample_rate)
    log_sample_processing_line(
        f"Applied low-pass ({cutoff_hz} Hz · steep={steepness:.2f} · res={resonance:.2f})"
    )


def apply_highpass_to_sample(input_path, cutoff_hz=120, **_kwargs):
    """Apply steep high-pass; optional resonance / steepness in _kwargs."""
    resonance = float(_kwargs.get("resonance", 0.0))
    steepness = float(_kwargs.get("steepness", 0.72))
    audio, sample_rate = _read_channels_first(input_path)
    processed = dsp.apply_steep_highpass(
        audio,
        sample_rate,
        cutoff_hz,
        resonance=resonance,
        steepness=steepness,
    )
    _write_channels_first(input_path, processed, sample_rate)
    log_sample_processing_line(
        f"Applied high-pass ({cutoff_hz} Hz · steep={steepness:.2f} · res={resonance:.2f})"
    )


def apply_bandpass_to_sample(input_path, low_hz=300, high_hz=6000, **_kwargs):
    """Steep Chebyshev band-pass (optional steepness=0 mimics mild 4th-order Butterworth)."""
    resonance = float(_kwargs.get("resonance", 0.0))
    steepness = float(_kwargs.get("steepness", 0.0))
    audio, sample_rate = _read_channels_first(input_path)
    processed = dsp.apply_steep_bandpass(
        audio,
        sample_rate,
        low_hz,
        high_hz,
        resonance=resonance,
        steepness=steepness,
    )
    _write_channels_first(input_path, processed, sample_rate)
    log_sample_processing_line(
        f"Applied band-pass ({low_hz:.0f}–{high_hz:.0f} Hz · steep={steepness:.2f} · res={resonance:.2f})"
    )


def apply_eq_lows_to_sample(input_path, gain_db, cutoff_hz=250):
    """Apply low-shelf EQ and overwrite the file. gain_db: e.g. +5 or -5."""
    audio, sample_rate = _read_channels_first(input_path)
    processed = dsp.apply_low_shelf(audio, sample_rate, cutoff_hz, gain_db)
    _write_channels_first(input_path, processed, sample_rate)
    log_sample_processing_line(f"EQ lows {gain_db:+.0f} dB")


def apply_eq_mids_to_sample(input_path, gain_db, centre_hz=1000):
    """Apply mid peak EQ and overwrite the file. gain_db: e.g. +5 or -5."""
    audio, sample_rate = _read_channels_first(input_path)
    processed = dsp.apply_peaking_eq(audio, sample_rate, centre_hz, gain_db, q=0.7)
    _write_channels_first(input_path, processed, sample_rate)
    log_sample_processing_line(f"EQ mids {gain_db:+.0f} dB")


def apply_eq_highs_to_sample(input_path, gain_db, cutoff_hz=4000):
    """Apply high-shelf EQ and overwrite the file. gain_db: e.g. +5 or -5."""
    audio, sample_rate = _read_channels_first(input_path)
    processed = dsp.apply_high_shelf(audio, sample_rate, cutoff_hz, gain_db)
    _write_channels_first(input_path, processed, sample_rate)
    log_sample_processing_line(f"EQ highs {gain_db:+.0f} dB")


def apply_granular_synthesis(
    input_file,
    grain_size=0.05,
    overlap=0.5,
    pitch_shift=0,
    stretch_factor=1.0,
    randomness=0.1,
    window_type="hann",
):
    """
    Apply granular synthesis directly to an audio file (overwrites the input file).

    Parameters:
    - input_file (str): Path to the input WAV file (will be overwritten).
    - grain_size (float): Grain size in seconds.
    - overlap (float): Overlap factor (0 to 1).
    - pitch_shift (float): Pitch shift in semitones.
    - stretch_factor (float): Time stretching factor (1.0 = no change).
    - randomness (float): Random grain position shift (0.0 = no randomness, 1.0 = max randomness).
    - window_type (str): Type of window function ('hann' or 'gaussian') to smooth grain edges.
    """

    # Load audio
    y, sr = librosa.load(input_file, sr=None)

    # Calculate grain length in samples
    grain_samples = int(grain_size * sr)
    hop_samples = int(grain_samples * (1 - overlap))

    # Generate a windowing function to smooth grains
    if window_type == "hann":
        window = np.hanning(grain_samples)
    elif window_type == "gaussian":
        window = np.exp(
            -0.5 * ((np.linspace(-2, 2, grain_samples)) ** 2)
        )  # Gaussian window
    else:
        window = np.ones(grain_samples)  # Default to rectangular (no windowing)

    # Generate grain positions
    grain_positions = np.arange(0, len(y) - grain_samples, hop_samples)

    # Apply randomness to grain positions
    if randomness > 0:
        max_shift = int(grain_samples * randomness)  # Max shift in samples
        random_shifts = np.random.randint(
            -max_shift, max_shift, size=grain_positions.shape
        )
        grain_positions = np.clip(
            grain_positions + random_shifts, 0, len(y) - grain_samples
        )  # Ensure positions stay valid

    # Process grains
    processed_audio = np.zeros(len(y))
    for pos in grain_positions:
        grain = y[pos : pos + grain_samples] * window  # Apply windowing function

        # Apply pitch shifting
        if pitch_shift != 0:
            grain = librosa.effects.pitch_shift(grain, sr, pitch_shift)

        # Apply time-stretching
        if stretch_factor != 1.0:
            grain = librosa.effects.time_stretch(grain, stretch_factor)

        # Overlap-add the grain to the processed audio
        processed_audio[pos : pos + grain_samples] += grain

    # Normalize to prevent clipping
    processed_audio = processed_audio / np.max(np.abs(processed_audio))

    # Overwrite the original file with the processed audio
    sf.write(input_file, processed_audio, sr)