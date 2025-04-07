import sys
import os
import json
import random
import pedalboard
import os
import json
import pedalboard
from pedalboard import (
    Compressor,
    Gain,
    HighpassFilter,
    LowpassFilter,
    Limiter,
    Pedalboard,
)
from pedalboard.io import AudioFile
from mido import MidiFile, Message
from utils import (
    with_generate_sample_prompt as with_prompt,
    extract_plugin,
    GREEN,
    RESET,
    generate_sample_header,
)

PRESETS_PATH = "presets/presets.json"


# Convert MIDI file to MIDI messages
def midi_to_messages(midi_file_path):
    """Reads a MIDI file and converts it to MIDI messages for processing."""
    if not midi_file_path.lower().endswith(".mid"):
        raise ValueError(
            f"Invalid file type: {midi_file_path}. The script requires a MIDI (.mid) file."
        )

    mid = MidiFile(midi_file_path)
    midi_messages = []
    current_time = 0

    for track in mid.tracks:
        for msg in track:
            if msg.type in ["note_on", "note_off"]:  # Only process note messages
                current_time += (
                    msg.time / mid.ticks_per_beat
                )  # Convert ticks to seconds
                midi_messages.append(
                    Message(
                        msg.type,
                        note=msg.note,
                        velocity=msg.velocity,
                        time=current_time,
                    )
                )

    return midi_messages, mid.length  # Return the list of messages and the duration


SAMPLE_RATE = 44100


def generate_sample(
    input_path="input.mid",
    output_path="generated_sample.wav",
    presets_path=PRESETS_PATH,
    instrument=None,
    effect=None,
):
    print(generate_sample_header())

    loaded_effects = []

    # Load presets from JSON
    with open(presets_path, "r") as f:
        presets = json.load(f)

    # Separate instruments and effects
    instruments = [p for p in presets if p["type"] == "instrument"]
    effects = [p for p in presets if p["type"] == "effect_chain"]

    if not instruments or not effects:
        raise ValueError(f"No valid instruments or effects found in {presets_path}")

    if not instrument:
        instrument_preset = random.choice(instruments)
    else:
        # Try to find a matching instrument by name
        for preset in instruments:
            if preset["name"] == instrument:
                instrument_preset = preset
                break
        else:
            # If no match found, raise an error
            raise ValueError(f"No instrument found with the name '{instrument}'")

    if not effect:
        effect_preset = random.choice(effects)
    else:
        # Try to find a matching instrument by name
        for preset in effects:
            if preset["name"] == effect:
                effect_preset = preset
                break
        else:
            # If no match found, raise an error
            raise ValueError(f"No effect chain found with the name '{effect}'")

    print(
        with_prompt(
            f"selected {GREEN}{instrument_preset['name']}{RESET} sound processed with {GREEN}{effect_preset['name']}{RESET}"
        )
    )

    # Load the instrument plugin with `plugin_name` if available
    if instrument_preset["plugin_name"]:
        print(
            with_prompt(
                f"loading instrument {GREEN}{extract_plugin(instrument_preset['plugin_path'])}{RESET} as {GREEN}{instrument_preset['plugin_name']}{RESET}"
            )
        )
        instrument_plugin = pedalboard.load_plugin(
            instrument_preset["plugin_path"],
            plugin_name=instrument_preset["plugin_name"],
        )
    else:
        print(
            with_prompt(
                f"loading instrument {GREEN}{extract_plugin(instrument_preset['plugin_path'])}{RESET}"
            )
        )
        instrument_plugin = pedalboard.load_plugin(instrument_preset["plugin_path"])

    # Load the instrument's preset data
    with open(instrument_preset["preset_path"], "rb") as f:
        print(
            with_prompt(
                f"using preset {GREEN}{instrument_preset['name']}{RESET} ({instrument_preset['desc']})"
            )
        )
        instrument_plugin.preset_data = f.read()

    # Load the effect plugin with `plugin_name` if available
    print(
        with_prompt(
            f"loading fx chain {GREEN}{effect_preset['name']}{RESET} ({effect_preset['desc']})"
        )
    )

    for effect in effect_preset["effects"]:
        if effect["plugin_name"]:
            print(
                with_prompt(
                    f"inserting {GREEN}{extract_plugin(effect['plugin_path'])}{RESET} as {effect['plugin_name']}"
                )
            )
            effect_plugin = pedalboard.load_plugin(
                effect["plugin_path"], plugin_name=effect["plugin_name"]
            )
        else:
            print(
                with_prompt(
                    f"inserting {GREEN}{extract_plugin(effect['plugin_path'])}{RESET}"
                )
            )
            effect_plugin = pedalboard.load_plugin(effect["plugin_path"])

        # Load the effect's preset data
        with open(effect["preset_path"], "rb") as f:
            print(
                with_prompt(
                    f"using preset {GREEN}{effect['name']}{RESET} ({effect['desc']})"
                )
            )
            if hasattr(effect_plugin, "preset_data"):
                effect_plugin.preset_data = f.read()
            else:
                effect_plugin.raw_state = f.read()

        loaded_effects.append(effect_plugin)

    # Load MIDI file and get messages
    midi_messages, audio_length_s = midi_to_messages(input_path)

    # Process MIDI through the instrument plugin
    pre_fx_signal = instrument_plugin(
        midi_messages,
        duration=audio_length_s,
        sample_rate=SAMPLE_RATE,
        num_channels=2,  # Stereo
        buffer_size=8192,
        reset=False,
    )

    # Create a gain reduction effect (-6dB = 0.5 multiplier)
    gain_reduction = Pedalboard([Gain(gain_db=-6)])  # -6dB reduction

    pre_fx_signal = gain_reduction(pre_fx_signal, SAMPLE_RATE)

    print(with_prompt(f"sending midi and rendering audio ({audio_length_s:.2f}s)"))

    # Apply the selected effect plugin
    fx_chain = Pedalboard(loaded_effects)
    post_fx_signal = fx_chain(pre_fx_signal, SAMPLE_RATE)

    output_path = output_path.replace("#", "sharp")

    # Export processed audio
    with AudioFile(output_path, "w", SAMPLE_RATE, 2) as f:
        f.write(post_fx_signal)

    print(f"{GREEN}│{RESET}")

    return output_path


def apply_effect(input_path, effect_chain, presets_path=PRESETS_PATH):
    """Applies an effect chain from presets to a WAV file and overwrites it after backing up."""

    # Load presets
    with open(presets_path, "r") as f:
        presets = json.load(f)

    effects = [p for p in presets if p["type"] == "effect_chain"]

    if effect_chain == "":
        effect_preset = random.choice(effects)
    else:
        # Find the requested effect chain
        effect_preset = next(
            (
                p
                for p in presets
                if p["type"] == "effect_chain" and p["name"] == effect_chain
            ),
            None,
        )

    if not effect_preset:
        raise ValueError(f"Effect chain '{effect_chain}' not found in presets.")

    output_path = input_path.replace(".wav", f"_{effect_preset['name']}.wav")

    print(f"Applying effect chain: {effect_preset['name']}")

    # Load effect plugins
    loaded_effects = []

    for effect in effect_preset["effects"]:
        if effect["plugin_name"]:
            print(
                f"inserting {extract_plugin(effect['plugin_path'])} as {effect['plugin_name']}"
            )
            effect_plugin = pedalboard.load_plugin(
                effect["plugin_path"], plugin_name=effect["plugin_name"]
            )
        else:
            print(f"inserting {extract_plugin(effect['plugin_path'])}")
            effect_plugin = pedalboard.load_plugin(effect["plugin_path"])

        # Load the effect's preset data
        with open(effect["preset_path"], "rb") as f:
            print(f"using preset {effect['name']} ({effect['desc']})")
            if hasattr(effect_plugin, "preset_data"):
                effect_plugin.preset_data = f.read()
            else:
                effect_plugin.raw_state = f.read()

        loaded_effects.append(effect_plugin)

    # Load audio file
    with AudioFile(input_path) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate

    # Apply effects
    normalization_chain = [
        HighpassFilter(cutoff_frequency_hz=40),  # Remove sub-bass rumble
        LowpassFilter(cutoff_frequency_hz=18000),  # Remove harsh highs if needed
        Compressor(
            threshold_db=-24, ratio=1.5, attack_ms=30, release_ms=200
        ),  # Gentle compression for control
        Limiter(
            threshold_db=-4, release_ms=250
        ),  # Ensure headroom without crushing dynamics
    ]
    fx_chain = Pedalboard(loaded_effects + normalization_chain)
    processed_audio = fx_chain(audio, sample_rate)

    with AudioFile(
        output_path, "w", samplerate=sample_rate, num_channels=processed_audio.shape[0]
    ) as f:
        f.write(processed_audio)

    print(f"Effect chain '{effect_chain}' applied and saved: {input_path}")


def main():
    args = sys.argv[1:]

    if not args:
        print("Error: input_path is required (.mid)")
        return

    input_path = args[0]
    output_path = args[1] if len(args) > 1 else "generated_sample.wav"

    generate_sample(input_path, output_path)


if __name__ == "__main__":
    main()
