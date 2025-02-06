import os
import pedalboard
from pedalboard import Reverb
from pedalboard.io import AudioFile
from paulstretch import paulstretch
from utils import stretch_sample_header, with_stretch_sample_prompt as with_prompt, BLUE, RESET

def apply_long_wet_reverb(input_path):
    # Construct output filename
    dir_name, base_name = os.path.split(input_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(dir_name, f"{name}_reverbed{ext}")

    # Load audio file correctly
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)  # Read the entire file
        sample_rate = f.samplerate  # Get the sample rate

    # Create a very long, wet, and deep reverb effect for drone music
    board = pedalboard.Pedalboard([
        Reverb(
            room_size=0.95,   # Almost max size for a massive, cathedral-like space
            damping=0.2,      # Less damping for a lush, long sustain
            wet_level=0.9,    # Very wet, almost entirely reverberated
            dry_level=0.1,    # Low dry signal to make it almost fully wet
            width=1.0,        # Full stereo width for immersive sound
            freeze_mode=0.0   # Set to 1.0 if you want infinite sustain
        )
    ])

    # Process the audio through the reverb
    processed_audio = board(audio, sample_rate)

    # Save the processed audio correctly
    with AudioFile(output_path, 'w', samplerate=sample_rate, num_channels=processed_audio.shape[0]) as f:
        f.write(processed_audio)

    print(with_prompt(f"applying pedalboard reverb"))

    return output_path

def stretch_sample(
    input_path,
    output_path,
    stretch=8.0,  # Default stretch amount
    window_size=0.25,  # Default window size (seconds)
    start_frame=0,
    end_frame=None
):
    print(stretch_sample_header())
    print(with_prompt(f"applying paulstretch (stretch={stretch}, window_size={window_size})"))
    paulstretch(input_path, output_path, stretch=stretch, window_size=window_size, start_frame=start_frame, end_frame=end_frame, show_logs=False)
    reverberated_output = apply_long_wet_reverb(output_path)
    print(f"{BLUE}│{RESET}")

    return output_path, reverberated_output
