import os
import numpy as np
import soundfile as sf
import librosa
import pedalboard
from scipy.signal import fftconvolve

import dsp
from pedalboard import (
    Chorus,
    Compressor,
    Distortion,
    Gain,
    HighpassFilter,
    HighShelfFilter,
    Limiter,
    LowpassFilter,
    LowShelfFilter,
    Pedalboard,
    PeakFilter,
    Phaser,
    PitchShift,
    Resample,
    Reverb,
)
from pedalboard.io import AudioFile
from paulstretch import paulstretch
from utils import (
    process_drone_sample_header,
    with_process_drone_sample_prompt as with_prompt,
    BLUE,
    RESET,
)


def apply_long_wet_reverb(input_path):
    """
    Apply high-quality offline convolution reverb using our custom IR builder
    in dsp.make_reverb_ir. Designed for long, stadium-style tails for drones.
    """
    from dsp import make_reverb_ir

    print(with_prompt("applying HQ convolution reverb (stadium)"))

    # Construct output filename
    dir_name, base_name = os.path.split(input_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(dir_name, f"{name}_reverbed{ext}")

    # Load audio
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate

    # Build a long, stadium-style IR:
    # - length_sec: long tail for big space
    # - decay_sec: slow decay
    # - early_reflections: richer early field
    # - highpass_cutoff_hz: keep low-end clean
    # - tail_diffusion: high for smooth stadium tail
    ir = make_reverb_ir(
        sample_rate=sample_rate,
        length_sec=3.5,
        decay_sec=3.0,
        early_reflections=8,
        highpass_cutoff_hz=120.0,
        tail_diffusion=0.85,
        early_diffuse=True,
    )

    # Convolve per channel
    n_channels, n_samples = audio.shape
    wet = np.zeros_like(audio)
    for ch in range(n_channels):
        wet[ch] = fftconvolve(audio[ch], ir, mode="full")[:n_samples]

    # Match dry peak and mix heavily wet for drone vibe
    wet_peak = np.max(np.abs(wet)) + 1e-12
    dry_peak = np.max(np.abs(audio)) + 1e-12
    wet = wet * (dry_peak / wet_peak)
    wet_level = 0.9
    mix = wet_level * wet + (1.0 - wet_level) * audio

    # Write out reverbed file
    with AudioFile(
        output_path, "w", samplerate=sample_rate, num_channels=n_channels
    ) as f:
        f.write(mix)

    return output_path


def apply_normalization(input_path):
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate

    board = Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=40),  # Remove sub-bass rumble
            LowpassFilter(cutoff_frequency_hz=18000),  # Remove harsh highs if needed
            Compressor(
                threshold_db=-24, ratio=1.5, attack_ms=30, release_ms=200
            ),  # Gentle compression for control
            Limiter(
                threshold_db=-4, release_ms=250
            ),  # Ensure headroom without crushing dynamics
        ]
    )

    processed_audio = board(audio, sample_rate)

    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=processed_audio.shape[0]
    ) as f:
        f.write(processed_audio)


def apply_transposition(input_path, semitones):
    print(with_prompt(f"applying pedalboard transposition"))
    output_path = input_path.replace(".wav", f"_-_{semitones}st.wav")

    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate

    # Calculate new sample rate based on pitch shift
    pitch_factor = 2 ** (semitones / 12.0)
    new_sample_rate = int(sample_rate * pitch_factor)

    # Resampling to new pitch (changes speed naturally)
    board = Pedalboard([Resample(new_sample_rate)])

    # Apply pitch shift
    processed_audio = board(audio, sample_rate)

    # Save the new transposed file
    with AudioFile(
        output_path,
        "w",
        samplerate=new_sample_rate,
        num_channels=processed_audio.shape[0],
    ) as f:
        f.write(processed_audio)

    return output_path


def trim_sample_start(input_path, start_time_s):
    """Trims the start of the audio file at `start_time_s` and overwrites the file."""
    audio, sample_rate = sf.read(input_path)
    start_sample = int(start_time_s * sample_rate)

    if start_sample < 0 or start_sample >= len(audio):
        raise ValueError("Start time is out of bounds.")

    trimmed_audio = audio[start_sample:]
    sf.write(input_path, trimmed_audio, sample_rate)
    print(f"Trimmed start at {start_time_s}s: {input_path}")


def trim_sample_end(input_path, end_time_s):
    """Trims the end of the audio file at `end_time_s` and overwrites the file."""
    audio, sample_rate = sf.read(input_path)
    end_sample = int(end_time_s * sample_rate)

    if end_sample <= 0 or end_sample > len(audio):
        raise ValueError("End time is out of bounds.")

    trimmed_audio = audio[:end_sample]
    sf.write(input_path, trimmed_audio, sample_rate)
    print(f"Trimmed end at {end_time_s}s: {input_path}")


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
    print(f"Applied fade-in ({fade_in_time_s}s) to: {input_path}")


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
    print(f"Applied fade-out ({fade_out_time_s}s) to: {input_path}")


def increase_sample_gain(input_path, db):
    """Increases the gain of the audio file by `db` decibels and overwrites the file."""
    audio, sample_rate = sf.read(input_path)

    gain_factor = 10 ** (db / 20)  # Convert dB to linear scale
    audio *= gain_factor

    sf.write(input_path, audio, sample_rate)
    print(f"Increased gain by {db} dB: {input_path}")


def decrease_sample_gain(input_path, db):
    """Decreases the gain of the audio file by `db` decibels and overwrites the file."""
    audio, sample_rate = sf.read(input_path)

    gain_factor = 10 ** (-db / 20)  # Convert dB to linear scale
    audio *= gain_factor

    sf.write(input_path, audio, sample_rate)
    print(f"Decreased gain by {db} dB: {input_path}")


def reverse_sample(input_path):
    """Reverses the audio file and overwrites the existing file."""
    audio, sample_rate = sf.read(input_path)

    # Reverse the audio along the time axis
    reversed_audio = audio[::-1]

    sf.write(input_path, reversed_audio, sample_rate)
    print(f"Reversed sample saved to: {input_path}")


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
    print(f"Applied simple time stretch ({stretch_factor}x) to: {input_path}")


def apply_pitch_shift_preserve_length(input_path, semitones):
    """
    Pitch shift while preserving duration (length) using pedalboard PitchShift.
    This keeps tempo locked while still allowing artifacts in a musically useful way.
    """
    semitones = float(semitones)
    if semitones == 0:
        return

    with AudioFile(input_path) as f:
        audio = f.read(f.frames)  # shape: (channels, samples)
        sample_rate = f.samplerate

    if audio.size == 0:
        raise ValueError("Audio file is empty")

    board = Pedalboard([PitchShift(semitones=semitones)])
    shifted = board(audio, sample_rate)

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

    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=shifted.shape[0]
    ) as f:
        f.write(shifted)
    print(f"Applied pitch shift ({semitones:+g} st) to: {input_path}")


def apply_reverb_to_sample(
    input_path,
    wet_level=0.35,
    reverb_length_sec=0.7,
    decay_sec=0.5,
    reverb_highpass_hz=100.0,
):
    """
    Apply high-quality convolution reverb (offline). Uses a generated IR with
    early reflections + dense decay tail. Optional highpass on the reverb
    only (reverb_highpass_hz > 0) to avoid low-end phase clash on drums;
    set to 0 to leave low-end in the reverb.
    """
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
    n_channels, n_samples = audio.shape
    ir = dsp.make_reverb_ir(
        sample_rate,
        length_sec=reverb_length_sec,
        decay_sec=decay_sec,
        highpass_cutoff_hz=reverb_highpass_hz,
    )
    wet = np.zeros_like(audio)
    for ch in range(n_channels):
        wet[ch] = fftconvolve(audio[ch], ir, mode="full")[:n_samples]
    wet_peak = np.max(np.abs(wet)) + 1e-12
    dry_peak = np.max(np.abs(audio)) + 1e-12
    wet = wet * (dry_peak / wet_peak)
    mix = wet_level * wet + (1.0 - wet_level) * audio
    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=n_channels
    ) as out:
        out.write(mix)
    msg = f"Applied convolution reverb (wet={wet_level}, {reverb_length_sec}s IR"
    if reverb_highpass_hz > 0:
        msg += f", HPF {reverb_highpass_hz} Hz"
    msg += f") to: {input_path}"
    print(msg)


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


def apply_distortion_to_sample(input_path, drive_db=6.0):
    """Apply pedalboard Distortion and overwrite the file."""
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
    board = Pedalboard([Distortion(drive_db=drive_db)])
    processed = board(audio, sample_rate)
    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=processed.shape[0]
    ) as out:
        out.write(processed)
    print(f"Applied distortion (drive={drive_db} dB) to: {input_path}")


def apply_compress_to_sample(
    input_path,
    threshold_db=-20.0,
    ratio=10.0,
    attack_ms=3.0,
    release_ms=80.0,
):
    """Apply aggressive pedalboard compression and overwrite the file."""
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
    board = Pedalboard([
        Compressor(
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
        ),
    ])
    processed = board(audio, sample_rate)
    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=processed.shape[0]
    ) as out:
        out.write(processed)
    print(f"Applied aggressive compression to: {input_path}")


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
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=highpass_hz),
        Distortion(drive_db=drive_db),
        LowpassFilter(cutoff_frequency_hz=lowpass_hz),
    ])
    processed = board(audio, sample_rate)
    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=processed.shape[0]
    ) as out:
        out.write(processed)
    print(f"Applied mid-focused overdrive to: {input_path}")


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
    """Apply pedalboard Chorus and overwrite the file."""
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
    board = Pedalboard([
        Chorus(
            rate_hz=rate_hz,
            depth=depth,
            centre_delay_ms=centre_delay_ms,
            feedback=feedback,
            mix=mix,
        )
    ])
    processed = board(audio, sample_rate)
    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=processed.shape[0]
    ) as out:
        out.write(processed)
    print(f"Applied chorus to: {input_path}")


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
    """Apply flanger-style effect (Chorus with short delay and feedback) and overwrite the file."""
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
    board = Pedalboard([
        Chorus(
            rate_hz=rate_hz,
            depth=depth,
            centre_delay_ms=centre_delay_ms,
            feedback=feedback,
            mix=mix,
        ),
    ])
    processed = board(audio, sample_rate)
    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=processed.shape[0]
    ) as out:
        out.write(processed)
    print(f"Applied flanger to: {input_path}")


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
    """Apply pedalboard Phaser and overwrite the file."""
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
    board = Pedalboard([
        Phaser(
            rate_hz=rate_hz,
            depth=depth,
            centre_frequency_hz=centre_frequency_hz,
            feedback=feedback,
            mix=mix,
        )
    ])
    processed = board(audio, sample_rate)
    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=processed.shape[0]
    ) as out:
        out.write(processed)
    print(f"Applied phaser to: {input_path}")


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


def apply_lowpass_to_sample(input_path, cutoff_hz=6000, order=6):
    """Apply aggressive low-pass with steep cutoff. Truly cuts high frequencies."""
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
    processed = dsp.apply_steep_lowpass(audio, sample_rate, cutoff_hz)
    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=processed.shape[0]
    ) as out:
        out.write(processed)
    print(f"Applied LPF ({cutoff_hz} Hz) to: {input_path}")


def apply_highpass_to_sample(input_path, cutoff_hz=100, order=6):
    """Apply aggressive high-pass with steep cutoff. Truly cuts low frequencies."""
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
    processed = dsp.apply_steep_highpass(audio, sample_rate, cutoff_hz)
    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=processed.shape[0]
    ) as out:
        out.write(processed)
    print(f"Applied HPF ({cutoff_hz} Hz) to: {input_path}")


def apply_bandpass_to_sample(
    input_path, low_hz=300, high_hz=6000, order=4
):
    """Apply Butterworth bandpass filter and overwrite the file."""
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
    processed = dsp.apply_butter_filter(
        audio, sample_rate, "band", (low_hz, high_hz), order
    )
    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=processed.shape[0]
    ) as out:
        out.write(processed)
    print(f"Applied BPF ({low_hz}–{high_hz} Hz, order {order}) to: {input_path}")


def apply_eq_lows_to_sample(input_path, gain_db, cutoff_hz=250):
    """Apply low-shelf EQ and overwrite the file. gain_db: e.g. +5 or -5."""
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
    board = Pedalboard([
        LowShelfFilter(cutoff_frequency_hz=cutoff_hz, gain_db=gain_db, q=0.7),
    ])
    processed = board(audio, sample_rate)
    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=processed.shape[0]
    ) as out:
        out.write(processed)
    print(f"Applied Lows {gain_db:+.0f} dB to: {input_path}")


def apply_eq_mids_to_sample(input_path, gain_db, centre_hz=1000):
    """Apply mid peak EQ and overwrite the file. gain_db: e.g. +5 or -5."""
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
    board = Pedalboard([
        PeakFilter(cutoff_frequency_hz=centre_hz, gain_db=gain_db, q=0.7),
    ])
    processed = board(audio, sample_rate)
    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=processed.shape[0]
    ) as out:
        out.write(processed)
    print(f"Applied Mids {gain_db:+.0f} dB to: {input_path}")


def apply_eq_highs_to_sample(input_path, gain_db, cutoff_hz=4000):
    """Apply high-shelf EQ and overwrite the file. gain_db: e.g. +5 or -5."""
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
    board = Pedalboard([
        HighShelfFilter(cutoff_frequency_hz=cutoff_hz, gain_db=gain_db, q=0.7),
    ])
    processed = board(audio, sample_rate)
    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=processed.shape[0]
    ) as out:
        out.write(processed)
    print(f"Applied Highs {gain_db:+.0f} dB to: {input_path}")


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


def process_drone_sample(
    input_path,
    stretch=8.0,  # Default stretch amount
    window_size=0.25,  # Default window size (seconds)
    start_frame=0,
    end_frame=None,
):
    print(process_drone_sample_header())
    print(
        with_prompt(
            f"applying paulstretch (stretch={stretch}, window_size={window_size})"
        )
    )
    stretch_output = paulstretch(
        input_path,
        stretch=stretch,
        window_size=window_size,
        start_frame=start_frame,
        end_frame=end_frame,
        show_logs=False,
    )
    stretch_reverb_output = apply_long_wet_reverb(stretch_output)
    stretch_reverb_transpose_output = apply_transposition(stretch_reverb_output, -12)
    print(with_prompt("normalizing audio"))
    apply_normalization(input_path)
    apply_normalization(stretch_output)
    apply_normalization(stretch_reverb_output)
    print(f"{BLUE}│{RESET}")

    return stretch_output, stretch_reverb_output, stretch_reverb_transpose_output
