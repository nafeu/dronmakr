import os
import numpy as np
import soundfile as sf
import librosa
import pedalboard
from scipy.signal import butter, fftconvolve, filtfilt
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
    print(with_prompt(f"applying pedalboard reverb"))

    # Construct output filename
    dir_name, base_name = os.path.split(input_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(dir_name, f"{name}_reverbed{ext}")

    # Load audio file correctly
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)  # Read the entire file
        sample_rate = f.samplerate  # Get the sample rate

    # Create a very long, wet, and deep reverb effect for drone music
    board = pedalboard.Pedalboard(
        [
            Gain(gain_db=-4),
            Reverb(
                room_size=0.95,  # Almost max size for a massive, cathedral-like space
                damping=0.1,  # Less damping for a lush, long sustain
                wet_level=0.9,  # Very wet, almost entirely reverberated
                dry_level=0.1,  # Low dry signal to make it almost fully wet
                width=1.0,  # Full stereo width for immersive sound
                freeze_mode=0.0,  # Set to 1.0 if you want infinite sustain
            ),
        ]
    )

    # Process the audio through the reverb
    processed_audio = board(audio, sample_rate)

    # Save the processed audio correctly
    with AudioFile(
        output_path, "w", samplerate=sample_rate, num_channels=processed_audio.shape[0]
    ) as f:
        f.write(processed_audio)

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


def _highpass_ir(ir: np.ndarray, sample_rate: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """Apply highpass to an IR so reverb has no energy below cutoff_hz (avoids low-end phase clash)."""
    if cutoff_hz <= 0 or cutoff_hz >= sample_rate / 2.1:
        return ir
    nyq = 0.5 * sample_rate
    low = cutoff_hz / nyq
    b, a = butter(order, low, btype="high")
    return filtfilt(b, a, ir).astype(ir.dtype)


def _make_reverb_ir(
    sample_rate,
    length_sec=0.7,
    decay_sec=0.5,
    early_reflections=5,
    highpass_cutoff_hz=0.0,
):
    """
    Build a high-quality reverb impulse response for offline convolution.
    Early reflections (discrete echoes) + dense tail (exponential decay of
    filtered noise). Optional highpass on the IR so reverb doesn't reflect
    low-end (set highpass_cutoff_hz > 0, e.g. 80–120 for drums).
    """
    n = int(sample_rate * length_sec)
    ir = np.zeros(n, dtype=np.float64)
    # Early reflections: sparse delays, smaller time window for small room
    rng = np.random.default_rng(42)
    for _ in range(early_reflections):
        idx = int(rng.uniform(0.002 * sample_rate, 0.04 * sample_rate))
        if idx < n:
            ir[idx] += rng.uniform(0.12, 0.35)
    # Dense tail: exponential decay of lowpassed noise
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
        ir = _highpass_ir(ir, sample_rate, highpass_cutoff_hz)
    ir = ir / (np.max(np.abs(ir)) + 1e-12)
    return ir.astype(np.float32)


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
    ir = _make_reverb_ir(
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


def apply_distortion_to_sample(input_path):
    """Apply pedalboard Distortion and overwrite the file."""
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
    board = Pedalboard([Distortion(drive_db=6)])
    processed = board(audio, sample_rate)
    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=processed.shape[0]
    ) as out:
        out.write(processed)
    print(f"Applied distortion to: {input_path}")


def apply_chorus_to_sample(input_path):
    """Apply pedalboard Chorus and overwrite the file."""
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
    board = Pedalboard([Chorus(rate_hz=1.0, depth=0.25, centre_delay_ms=7.0, feedback=0.0, mix=0.5)])
    processed = board(audio, sample_rate)
    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=processed.shape[0]
    ) as out:
        out.write(processed)
    print(f"Applied chorus to: {input_path}")


def apply_flanger_to_sample(input_path):
    """Apply flanger-style effect (Chorus with short delay and feedback) and overwrite the file."""
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
    board = Pedalboard([
        Chorus(
            rate_hz=0.5,
            depth=0.4,
            centre_delay_ms=2.0,
            feedback=0.3,
            mix=0.5,
        ),
    ])
    processed = board(audio, sample_rate)
    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=processed.shape[0]
    ) as out:
        out.write(processed)
    print(f"Applied flanger to: {input_path}")


def apply_phaser_to_sample(input_path):
    """Apply pedalboard Phaser and overwrite the file."""
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
    board = Pedalboard([Phaser(rate_hz=1.0, depth=0.7, centre_frequency_hz=1000, feedback=0.5, mix=0.5)])
    processed = board(audio, sample_rate)
    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=processed.shape[0]
    ) as out:
        out.write(processed)
    print(f"Applied phaser to: {input_path}")


def apply_lowpass_to_sample(input_path, cutoff_hz=8000):
    """Apply low-pass filter and overwrite the file. Cuts highs above cutoff."""
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
    board = Pedalboard([LowpassFilter(cutoff_frequency_hz=cutoff_hz)])
    processed = board(audio, sample_rate)
    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=processed.shape[0]
    ) as out:
        out.write(processed)
    print(f"Applied LPF ({cutoff_hz} Hz) to: {input_path}")


def apply_highpass_to_sample(input_path, cutoff_hz=80):
    """Apply high-pass filter and overwrite the file. Cuts lows below cutoff."""
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate
    board = Pedalboard([HighpassFilter(cutoff_frequency_hz=cutoff_hz)])
    processed = board(audio, sample_rate)
    with AudioFile(
        input_path, "w", samplerate=sample_rate, num_channels=processed.shape[0]
    ) as out:
        out.write(processed)
    print(f"Applied HPF ({cutoff_hz} Hz) to: {input_path}")


def apply_eq_lows_to_sample(input_path, gain_db, cutoff_hz=250):
    """Apply low-shelf EQ and overwrite the file. gain_db: +2 or -2."""
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
    """Apply mid peak EQ and overwrite the file. gain_db: +2 or -2."""
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
    """Apply high-shelf EQ and overwrite the file. gain_db: +2 or -2."""
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
