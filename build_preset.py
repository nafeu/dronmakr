import glob
import json
import select
import os
import pedalboard
import subprocess
import sys
import termios
import tty
from mido import Message
from settings import get_setting
from pedalboard import Pedalboard
from pedalboard.io import AudioFile
from threading import Event, Thread
from utils import (
    with_build_preset_prompt as with_prompt,
    generate_id,
    MAGENTA,
    RESET,
    TEMP_DIR,
    PRESETS_DIR,
)
from generate_midi import SUPPORTED_PATTERNS_INFO

PRESET_JSON = "presets.json"
PREVIEW_NUM_BARS = 1
PREVIEW_SAMPLE = "resources/CDEFGABC.wav"
PREVIEW_SAMPLE_RATE = 44100
PREVIEW_TEMP_PATH = "temp/preset_preview.wav"
PREVIEW_TEMPO_BPM = 120
PREVIEW_TIME_SIGNATURE = "4/4"

listener_running = True
close_window_event = Event()


def build_preset():
    os.makedirs(PRESETS_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    plugin_paths = get_setting("PLUGIN_PATHS", "").split(",")
    assert_instrument = get_setting("ASSERT_INSTRUMENT", "").split(",")
    ignore_plugins = get_setting("IGNORE_PLUGINS", "").split(",")
    custom_plugins = [
        plugin for plugin in get_setting("CUSTOM_PLUGINS", "").split(",") if plugin
    ]

    if not plugin_paths or plugin_paths == [""]:
        print(with_prompt("error: No VST paths found in settings (PLUGIN_PATHS)."))
        exit(1)

    available_plugins = list_installed_plugins(plugin_paths, custom_plugins)
    if not available_plugins:
        print(with_prompt("error: No VST plugins found."))
        exit(1)

    plugin_map = {
        format_plugin_name(path): path
        for path in available_plugins
        if not any(ignore in format_plugin_name(path) for ignore in ignore_plugins)
    }
    print(with_prompt("available plugins:\n"))

    # Display plugins in columns
    columns = 3
    for i, plugin in enumerate(plugin_map.keys(), start=1):
        print(f"{i}. {plugin}".ljust(30), end="")
        if i % columns == 0:
            print()

    sys.stdout.flush()
    print("\n")

    # Select the first plugin
    selected_plugin = select_plugin(plugin_map)

    plugin, selected_plugin_name = load_plugin(selected_plugin)

    # Check if it's an effect
    effect_chain = []
    preset_name = ""
    preset_uid = ""
    preset_path = ""
    if plugin.is_effect and selected_plugin_name not in assert_instrument:
        effect_chain.append((selected_plugin, selected_plugin_name, plugin))

        (
            chain_preset_name,
            chain_preset_desc,
            chain_preset_uid,
            selected_plugin,
            selected_plugin_name,
            chain_preset_path,
            effect_chain,
        ) = edit_preset_with_ui(
            plugin, effect_chain, selected_plugin, selected_plugin_name
        )

        effect_chain[-1] = (
            *effect_chain[-1],
            (chain_preset_name, chain_preset_uid, chain_preset_path, chain_preset_desc),
        )

        while True:
            add_more = input(with_prompt("Add another effect? (y/n): ")).strip().lower()
            if add_more != "y":
                preset_name = input(
                    with_prompt("enter a name for this chain preset: ")
                ).strip()
                preset_desc = input(
                    with_prompt("enter a description for this chain preset: ")
                ).strip()
                preset_uid = generate_id()
                break

            # Select another effect
            selected_plugin = select_plugin(plugin_map)
            plugin, selected_plugin_name = load_plugin(selected_plugin)

            if plugin.is_effect:
                effect_chain.append((selected_plugin, selected_plugin_name, plugin))

                (
                    chain_preset_name,
                    chain_preset_desc,
                    chain_preset_uid,
                    selected_plugin,
                    selected_plugin_name,
                    chain_preset_path,
                    effect_chain,
                ) = edit_preset_with_ui(
                    plugin, effect_chain, selected_plugin, selected_plugin_name
                )

                effect_chain[-1] = (
                    *effect_chain[-1],
                    (
                        chain_preset_name,
                        chain_preset_uid,
                        chain_preset_path,
                        chain_preset_desc,
                    ),
                )
            else:
                print(with_prompt("That is not an effect! Try again."))
    else:
        (
            preset_name,
            preset_desc,
            preset_uid,
            selected_plugin,
            selected_plugin_name,
            preset_path,
            effect_chain,
        ) = edit_preset_with_ui(
            plugin, effect_chain, selected_plugin, selected_plugin_name
        )

    save_preset(
        preset_name,
        preset_desc,
        preset_uid,
        selected_plugin,
        selected_plugin_name,
        preset_path,
        effect_chain,
    )

    if not os.listdir(TEMP_DIR):
        os.rmdir(TEMP_DIR)


def calculate_audio_length(tempo_bpm, time_signature, num_bars):
    beats_per_bar, beat_duration_s = calculate_beat_info(tempo_bpm, time_signature)
    total_beats = beats_per_bar * num_bars
    total_duration_s = total_beats * beat_duration_s
    return total_duration_s


def calculate_beat_info(tempo_bpm, time_signature):
    beats_per_bar, beat_type = map(int, time_signature.split("/"))
    beat_duration_s = 60 / tempo_bpm
    return beats_per_bar, beat_duration_s


def list_installed_plugins(plugin_dirs, custom_plugins):
    plugin_paths = [plugin for plugin in custom_plugins]
    for plugin_dir in plugin_dirs:
        plugin_dir = plugin_dir.strip()
        if os.path.exists(plugin_dir):
            plugin_paths.extend(glob.glob(os.path.join(plugin_dir, "*.vst3")))
            plugin_paths.extend(glob.glob(os.path.join(plugin_dir, "*.vst")))
            plugin_paths.extend(glob.glob(os.path.join(plugin_dir, "*.dll")))
            plugin_paths.extend(glob.glob(os.path.join(plugin_dir, "*.so")))
            plugin_paths.extend(glob.glob(os.path.join(plugin_dir, "*.component")))
    return plugin_paths


def format_plugin_name(plugin_path):
    return (
        os.path.basename(plugin_path)
        .replace(".vst3", "")
        .replace(".vst", "")
        .replace(".component", " (AU)")
    )


def generate_midi():
    """Generate a simple MIDI sequence"""
    audio_length_s = calculate_audio_length(
        PREVIEW_TEMPO_BPM, PREVIEW_TIME_SIGNATURE, PREVIEW_NUM_BARS
    )
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

    empty_bar_length = calculate_audio_length(
        PREVIEW_TEMPO_BPM, PREVIEW_TIME_SIGNATURE, 1
    )
    audio_length_s += empty_bar_length

    return midi_messages, audio_length_s


def select_plugin(plugin_map):
    """Handles user input for selecting a plugin"""
    while True:
        try:
            selection = int(input(with_prompt("select a VST: "))) - 1
            if 0 <= selection < len(plugin_map.keys()):
                plugin_path = plugin_map[list(plugin_map.keys())[selection]]
                return plugin_path
            else:
                print(with_prompt("Invalid selection."))
        except ValueError:
            print(with_prompt("Enter a number."))


def load_plugin(plugin_path):
    """Loads a VST plugin, handling multiple embedded plugins if necessary."""
    try:
        return pedalboard.load_plugin(plugin_path), ""
    except ValueError as e:
        error_message = str(e)
        if "contains" in error_message and "To open a specific plugin" in error_message:

            # Extract available sub-plugins from the error message
            plugin_names = [
                line.strip().strip('"')
                for line in error_message.split("\n")
                if line.startswith('\t"')
            ]

            print(with_prompt("this VST3 file contains multiple plugins:"))
            for i, name in enumerate(plugin_names):
                print(f"{i + 1}. {name}")

            # Prompt user to select a sub-plugin
            while True:
                try:
                    sub_selection = (
                        int(
                            input(
                                with_prompt(
                                    "enter the number of the plugin you want to load: "
                                )
                            )
                        )
                        - 1
                    )
                    if 0 <= sub_selection < len(plugin_names):
                        selected_plugin_name = plugin_names[sub_selection]
                        return (
                            pedalboard.load_plugin(
                                plugin_path, plugin_name=selected_plugin_name
                            ),
                            selected_plugin_name,
                        )
                    else:
                        print(with_prompt("invalid selection, please try again."))
                except ValueError:
                    print(with_prompt("please enter a valid number."))

        else:
            raise


def preview_preset(plugin, effect_chain):
    """Processes the preset through an instrument or effect chain"""
    num_channels = 2

    if plugin.is_instrument:
        midi_messages, audio_length_s = generate_midi()
        pre_fx_signal = plugin(
            midi_messages,
            duration=audio_length_s,
            sample_rate=PREVIEW_SAMPLE_RATE,
            num_channels=num_channels,
            buffer_size=8192,
            reset=False,
        )

    elif plugin.is_effect:
        with AudioFile(PREVIEW_SAMPLE, "r") as f:
            audio_length_s = f.frames / f.samplerate
            pre_fx_signal = f.read(f.frames)

        # Apply all effects in the chain
        fx_chain = Pedalboard([fx[2] for fx in effect_chain])
        pre_fx_signal = fx_chain(pre_fx_signal, PREVIEW_SAMPLE_RATE)

    else:
        print(with_prompt("Unknown plugin type."))
        return

    # Write output
    with AudioFile(PREVIEW_TEMP_PATH, "w", PREVIEW_SAMPLE_RATE, num_channels) as f:
        f.write(pre_fx_signal)

    subprocess.call(["afplay", PREVIEW_TEMP_PATH])
    os.remove(PREVIEW_TEMP_PATH)


def save_preset(name, desc, uid, plugin_path, plugin_name, preset_path, effect_chain):
    """Saves the preset to `presets.json`"""
    preset_index_file = os.path.join(PRESETS_DIR, PRESET_JSON)

    if os.path.exists(preset_index_file):
        with open(preset_index_file, "r") as f:
            try:
                presets_data = json.load(f)
            except json.JSONDecodeError:
                presets_data = []
    else:
        presets_data = []

    if effect_chain:
        preset_data = {
            "id": uid,
            "name": name,
            "desc": desc,
            "type": "effect_chain",
            "effects": [
                {
                    "id": fx[3][1],
                    "name": fx[3][0],
                    "desc": fx[3][3],
                    "plugin_path": fx[0],
                    "plugin_name": fx[1],
                    "preset_path": fx[3][2],
                }
                for fx in effect_chain  # (chain_preset_name, chain_preset_id, chain_preset_path)
            ],
        }
    else:
        preset_data = {
            "id": uid,
            "name": name,
            "desc": desc,
            "plugin_path": plugin_path,
            "plugin_name": plugin_name,
            "preset_path": preset_path,
            "type": "instrument",
        }

    presets_data.append(preset_data)

    with open(preset_index_file, "w") as f:
        json.dump(presets_data, f, indent=4)


def listen_for_key(plugin, effect_chain):
    """Waits for Spacebar input from the user to trigger audio rendering immediately"""
    global listener_running
    print(with_prompt("[SPACEBAR] preview sound, [ENTER] continue.\n"))

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)

        while listener_running:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)

                if key == " ":
                    preview_preset(plugin, effect_chain)  # Preview the sound
                elif key == "\n":
                    print(with_prompt("closing editor..."))
                    close_window_event.set()
                    break

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def edit_preset_with_ui(plugin, effect_chain, selected_plugin, selected_plugin_name):
    """Handles opening the VST editor and listening for spacebar preview"""
    global listener_running

    close_window_event.clear()  # Ensure the event is reset before opening
    listener_running = True

    key_listener_thread = Thread(
        target=listen_for_key, args=(plugin, effect_chain), daemon=True
    )
    key_listener_thread.start()

    print(with_prompt("opening VST..."))
    plugin.show_editor(close_window_event)  # Open editor & wait

    listener_running = False
    key_listener_thread.join()

    while True:
        preset_name = input(with_prompt("enter a name for this preset: ")).strip()
        if not name_exists(preset_name):
            break
        else:
            print(with_prompt("name already in use, please enter a different name:"))
    preset_desc = input(with_prompt("enter a description for this preset: ")).strip()
    preset_uid = generate_id()

    preset_path = os.path.join(PRESETS_DIR, f"{preset_name}_{preset_uid}.vstpreset")
    with open(preset_path, "wb") as f:
        if hasattr(plugin, "preset_data"):
            f.write(plugin.preset_data)
        else:
            f.write(plugin.raw_state)

    print(with_prompt(f"preset saved to {preset_path}"))

    return (
        preset_name,
        preset_desc,
        preset_uid,
        selected_plugin,
        selected_plugin_name,
        preset_path,
        effect_chain,
    )


def list_presets(show_chain_plugins=False, show_patterns=False):
    preset_index_file = os.path.join(PRESETS_DIR, PRESET_JSON)

    if not os.path.exists(preset_index_file):
        print("No presets found.")
        return

    try:
        with open(preset_index_file, "r") as f:
            presets_data = json.load(f)
    except json.JSONDecodeError:
        print(with_prompt("Error reading preset index file."))
        return

    patterns = []
    instruments = []
    effect_chains = []

    for idx, preset in enumerate(presets_data, start=1):
        if preset.get("type") == "instrument":
            instruments.append((idx, preset["name"], preset["desc"]))
        elif preset.get("type") == "effect_chain":
            effect_chains.append((idx, preset))

    for idx, pattern in enumerate(SUPPORTED_PATTERNS_INFO, start=1):
        patterns.append((idx, pattern[0], pattern[1]))

    if len(instruments) < 1:
        print(with_prompt("No instruments added, use 'dronmakr preset'"))
        return

    if len(effect_chains) < 1:
        print(with_prompt("No effect chains added, use 'dronmakr preset'"))
        return

    longest_pattern_name_length = len(max(patterns, key=lambda x: len(x[1]))[1])
    longest_instrument_name_length = len(max(instruments, key=lambda x: len(x[1]))[1])
    longest_effect_chain_name_length = max(
        [len(item[1]["name"]) for item in effect_chains]
    )

    if show_chain_plugins:
        for idx, effect_chain in effect_chains:
            for plugin in effect_chain["effects"]:
                if (len(plugin["name"]) + 2) > longest_effect_chain_name_length:
                    longest_effect_chain_name_length = len(plugin["name"]) + 2

    if show_patterns:
        print(f"{MAGENTA}■ patterns{RESET}")
        print(f"{MAGENTA}│{RESET}")
        for idx, name, desc in patterns:
            desc_spacing = " " * (longest_pattern_name_length - len(name))
            print(f"{MAGENTA}│  {name} {RESET}{desc_spacing}{desc}")
        print(f"{MAGENTA}│{RESET}")

    print(f"{MAGENTA}■ instruments{RESET}")
    print(f"{MAGENTA}│{RESET}")
    for idx, name, desc in instruments:
        desc_spacing = " " * (longest_instrument_name_length - len(name))
        print(f"{MAGENTA}│  {name} {RESET}{desc_spacing}{desc}")

    print(f"{MAGENTA}│{RESET}")
    if effect_chains:
        print(f"{MAGENTA}■ effects{RESET}")
        print(f"{MAGENTA}│{RESET}")
        for idx, chain in effect_chains:
            desc_spacing = " " * (longest_effect_chain_name_length - len(chain["name"]))
            print(f"{MAGENTA}│  {chain['name']} {RESET}{desc_spacing}{chain['desc']}")
            if show_chain_plugins:
                for effect in chain["effects"]:
                    desc_spacing = " " * (
                        longest_effect_chain_name_length - len(effect["name"]) - 2
                    )
                    print(
                        f"{MAGENTA}│    {effect['name']} {RESET}{desc_spacing}{effect['desc']}"
                    )


def name_exists(name):
    """Checks the collection at `presets.json` and sees if any entry has that name already"""
    preset_index_file = os.path.join(PRESETS_DIR, PRESET_JSON)

    if not os.path.exists(preset_index_file):
        return False

    with open(preset_index_file, "r") as f:
        presets_data = json.load(f)

    for preset in presets_data:
        if preset["name"].lower() == name.lower():
            return True

    return False


if __name__ == "__main__":
    try:
        build_preset()
    except (KeyboardInterrupt, EOFError):
        print(with_prompt("exiting..."))
        sys.exit(0)
