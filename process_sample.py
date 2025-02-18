import os
import numpy as np
import soundfile as sf
import pedalboard
from pedalboard import (
    Compressor,
    Reverb,
    Gain,
    HighpassFilter,
    LowpassFilter,
    Limiter,
    Pedalboard,
    Resample,
)
from pedalboard.io import AudioFile
from paulstretch import paulstretch
from utils import (
    process_sample_header,
    with_process_sample_prompt as with_prompt,
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


def process_sample(
    input_path,
    stretch=8.0,  # Default stretch amount
    window_size=0.25,  # Default window size (seconds)
    start_frame=0,
    end_frame=None,
):
    print(process_sample_header())
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
