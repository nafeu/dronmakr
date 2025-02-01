import glob
import json
import select
import os
import pedalboard
import subprocess
import sys
import uuid
import termios
import tty
from dotenv import load_dotenv
from mido import Message
from pedalboard import Pedalboard
from pedalboard.io import AudioFile
from threading import Event, Thread

RED = "\033[31m"
RESET = "\033[0m"

PROMPT = f"{RED}┌ dronmakr ■ preset builder {RED}┐{RESET}"

PRESET_DIR = "presets"
PREVIEW_NUM_BARS = 1
PREVIEW_SAMPLE_RATE = 44100
PREVIEW_TEMP_PATH = "temp_preset_preview.wav"
PREVIEW_TEMPO_BPM = 120
PREVIEW_TIME_SIGNATURE = "4/4"

listener_running = True
close_window_event = Event()

def calculate_audio_length(tempo_bpm, time_signature, num_bars):
    beats_per_bar, beat_duration_s = calculate_beat_info(tempo_bpm, time_signature)
    total_beats = beats_per_bar * num_bars
    total_duration_s = total_beats * beat_duration_s
    return total_duration_s


def calculate_beat_info(tempo_bpm, time_signature):
    beats_per_bar, beat_type = map(int, time_signature.split("/"))
    beat_duration_s = 60 / tempo_bpm
    return beats_per_bar, beat_duration_s


def list_vst_plugins(vst_dirs):
    vst_plugins = []
    for vst_dir in vst_dirs:
        vst_dir = vst_dir.strip()
        if os.path.exists(vst_dir):
            vst_plugins.extend(glob.glob(os.path.join(vst_dir, "*.vst3")))
            vst_plugins.extend(glob.glob(os.path.join(vst_dir, "*.dll")))
            vst_plugins.extend(glob.glob(os.path.join(vst_dir, "*.so")))
    return vst_plugins


def format_plugin_name(plugin_path):
    return os.path.basename(plugin_path).replace(".vst3", "")


def generate_midi():
    """Generate a simple MIDI sequence"""
    audio_length_s = calculate_audio_length(PREVIEW_TEMPO_BPM, PREVIEW_TIME_SIGNATURE, PREVIEW_NUM_BARS)
    scale_notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C D E F G A B C (MIDI notes)
    midi_messages = []
    for index, note in enumerate(scale_notes):
        note_length_s = audio_length_s / len(scale_notes)
        midi_messages.append(
            Message("note_on", note=note, time=(index * note_length_s))
        )
        midi_messages.append(
            Message("note_off", note=note, time=((index + 1) * note_length_s))
        )

    empty_bar_length = calculate_audio_length(PREVIEW_TEMPO_BPM, PREVIEW_TIME_SIGNATURE, 1)
    audio_length_s += empty_bar_length

    return midi_messages, audio_length_s


def main():
    os.makedirs(PRESET_DIR, exist_ok=True)

    load_dotenv()

    selected_plugin = ""
    selected_plugin_name = ""

    if len(sys.argv) > 1:
        selected_plugin = sys.argv[1]
        print(f"{PROMPT} loading {selected_plugin}...")
    else:

        vst_paths = os.getenv("VST_PATHS", "").split(",")

        if not vst_paths or vst_paths == [""]:
            print(
                f"{PROMPT} error: No VST paths found in the .env file. Make sure to set VST_PATHS."
            )
            exit(1)

        available_plugins = list_vst_plugins(vst_paths)

        if not available_plugins:
            print(f"{PROMPT} error: No VST plugins found in the specified paths.")
            exit(1)

        plugin_map = {format_plugin_name(path): path for path in available_plugins}

        print(f"{PROMPT} available plugins:\n")
        columns = 3
        for i, plugin in enumerate(plugin_map.keys(), start=1):
            print(f"{i}. {plugin}".ljust(35), end="")

            if i % columns == 0:
                print()

        sys.stdout.flush()
        print("\n")

        while True:
            try:
                selection = (
                    int(input(f"{PROMPT} enter the number of the VST you want to open: ")) - 1
                )
                if 0 <= selection < len(plugin_map.keys()):
                    selected_plugin = plugin_map[list(plugin_map.keys())[selection]]
                    break
                else:
                    print(f"{PROMPT} invalid selection, try again.")
            except ValueError:
                print(f"{PROMPT} please enter a valid number.")

    try:
        plugin = pedalboard.load_plugin(selected_plugin)
    except ValueError as e:
        error_message = str(e)
        if "contains" in error_message and "To open a specific plugin" in error_message:

            plugin_names = [
                line.strip().strip('"')
                for line in error_message.split("\n")
                if line.startswith('\t"')
            ]

            print(f"{PROMPT} this VST3 file contains multiple plugins:")
            for i, name in enumerate(plugin_names):
                print(f"{i + 1}. {name}")

            while True:
                try:
                    sub_selection = (
                        int(
                            input(f"{PROMPT} enter the number of the plugin you want to load: ")
                        )
                        - 1
                    )
                    if 0 <= sub_selection < len(plugin_names):
                        selected_plugin_name = plugin_names[sub_selection]
                        plugin = pedalboard.load_plugin(
                            selected_plugin, plugin_name=selected_plugin_name
                        )
                        break
                    else:
                        print(f"{PROMPT} invalid selection, please try again.")
                except ValueError:
                    print(f"{PROMPT} please enter a valid number.")
        else:
            raise

    preset_file = "preset.vstpreset"

    if os.path.exists(preset_file):
        print(f"{PROMPT} loading existing preset...")
        with open(preset_file, "rb") as f:
            existing_preset_data = f.read()
            plugin.preset_data = existing_preset_data

    def preview_preset():
        """Renders the preset through the VST and writes an audio file."""
        num_channels = 2

        if plugin.is_instrument:
            # For VST Instruments: Generate MIDI
            midi_messages, audio_length_s = generate_midi()
            print(f"{PROMPT} Using MIDI input for instrument plugin...")

            # Process MIDI through the VST instrument
            pre_fx_signal = plugin(
                midi_messages,
                duration=audio_length_s,
                sample_rate=PREVIEW_SAMPLE_RATE,
                num_channels=num_channels,
                buffer_size=8192,  # Default buffer size
                reset=False,  # Avoid resetting the plugin state
            )

        elif plugin.is_effect:
            # For VST Effects: Use pre-recorded audio
            print(f"{PROMPT} Using audio file for effect plugin...")

            # Load the audio file into a NumPy array
            with AudioFile("cmaj-piano.wav", "r") as f:
                audio_length_s = f.frames / f.samplerate
                pre_fx_signal = f.read(f.frames)  # Read entire file

            # Process audio through the VST effect
            pre_fx_signal = plugin(
                pre_fx_signal,  # Audio input
                sample_rate=PREVIEW_SAMPLE_RATE,
                buffer_size=8192,  # Default buffer size
                reset=False,  # Avoid resetting the plugin state
            )

        else:
            print(f"{PROMPT} unknown plugin type.")
            return

        # Apply additional effects if needed
        fx_chain = Pedalboard([])  # Add effects here if needed
        post_fx_signal = fx_chain(pre_fx_signal, PREVIEW_SAMPLE_RATE)

        # Write the output audio file
        with AudioFile(PREVIEW_TEMP_PATH, "w", PREVIEW_SAMPLE_RATE, num_channels) as f:
            f.write(post_fx_signal)

        print(f"{PROMPT} previewing for {audio_length_s} seconds...")

        # Play the rendered audio
        subprocess.call(["afplay", PREVIEW_TEMP_PATH])

        # Delete the temporary preview file
        if os.path.exists(PREVIEW_TEMP_PATH):
            os.remove(PREVIEW_TEMP_PATH)
            print(f"{PROMPT} deleted temporary file: {PREVIEW_TEMP_PATH}")


    def listen_for_key():
        """Waits for Spacebar input from the user to trigger audio rendering immediately"""
        global listener_running
        print(
            f"{PROMPT} [SPACEBAR] preview sound, [ENTER] continue.\n"
        )

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)

            while listener_running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)

                    if key == " ":
                        preview_preset()
                    elif key == "\n":
                        print(f"{PROMPT} closing editor...")
                        close_window_event.set()
                        break

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    key_listener_thread = Thread(target=listen_for_key, daemon=True)
    key_listener_thread.start()

    print(f"{PROMPT} opening VST...")
    plugin.show_editor(close_window_event)

    listener_running = False
    key_listener_thread.join()

    preset_name = input(f"{PROMPT} enter a name for this preset: ").strip()

    preset_uid = str(uuid.uuid4())[:8]

    preset_filename = f"{preset_name}_{preset_uid}.vstpreset"
    preset_path = os.path.join(PRESET_DIR, preset_filename)

    with open(preset_path, "wb") as f:
        f.write(plugin.preset_data)

    print(f"{PROMPT} preset saved to {preset_path}")

    preset_index_file = os.path.join(PRESET_DIR, "presets.json")

    if os.path.exists(preset_index_file):
        with open(preset_index_file, "r") as f:
            try:
                presets_data = json.load(f)
            except json.JSONDecodeError:
                presets_data = []
    else:
        presets_data = []

    presets_data.append({
        "id": preset_uid,
        "name": preset_name,
        "plugin_path": selected_plugin,
        "plugin_name": selected_plugin_name,
        "preset_path": preset_path,
        "type": "instrument" if plugin.is_instrument else "effect"
    })

    with open(preset_index_file, "w") as f:
        json.dump(presets_data, f, indent=4)

if __name__ == "__main__":
    try:
        main()  # Your main script logic
    except KeyboardInterrupt:
        print(f"\n{PROMPT} exiting...")  # Graceful exit on CTRL+C
        sys.exit(0)
    except EOFError:
        print(f"\n{PROMPT} exiting...")  # Graceful exit on CTRL+D
        sys.exit(0)
