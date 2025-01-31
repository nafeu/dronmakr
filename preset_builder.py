import os
import sys
import glob
import pedalboard
import uuid
import json
from dotenv import load_dotenv

# Set sample rate and buffer size
SAMPLE_RATE = 44100
BUFFER_SIZE = 512  # Size of each processing chunk

def list_vst_plugins(vst_dirs):
    vst_plugins = []
    for vst_dir in vst_dirs:
        vst_dir = vst_dir.strip()
        if os.path.exists(vst_dir):
            vst_plugins.extend(glob.glob(os.path.join(vst_dir, "*.vst3")))  # macOS VST3
            vst_plugins.extend(glob.glob(os.path.join(vst_dir, "*.dll")))   # Windows VST2
            vst_plugins.extend(glob.glob(os.path.join(vst_dir, "*.so")))    # Linux VST2
    return vst_plugins


def format_plugin_name(plugin_path):
    return os.path.basename(plugin_path).replace(".vst3", "")


def main():
    # Ensure the `presets/` folder exists
    PRESET_DIR = "presets"
    os.makedirs(PRESET_DIR, exist_ok=True)

    # Load environment variables from .env file
    load_dotenv()

    selected_plugin = ""

    if len(sys.argv) > 1:
        selected_plugin = sys.argv[1]
        print(f"Loading: {selected_plugin}")
    else:
        # Read VST paths from .env file
        vst_paths = os.getenv("VST_PATHS", "").split(",")

        # Validate VST paths
        if not vst_paths or vst_paths == [""]:
            print("Error: No VST paths found in the .env file. Make sure to set VST_PATHS.")
            exit(1)

        # Get the list of available VST plugins
        available_plugins = list_vst_plugins(vst_paths)

        if not available_plugins:
            print("No VST plugins found in the specified paths.")
            exit(1)

        plugin_map = { format_plugin_name(path): path for path in available_plugins }

        # Display available plugins
        print("Available VST Plugins:")
        for i, plugin in enumerate(plugin_map.keys()):
            print(f"{i + 1}. {plugin}")

        # Ask user to select a plugin
        while True:
            try:
                selection = int(input("\nEnter the number of the VST you want to open: ")) - 1
                if 0 <= selection < len(plugin_map.keys()):
                    selected_plugin = plugin_map[list(plugin_map.keys())[selection]]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    try:
        plugin = pedalboard.load_plugin(selected_plugin)
    except ValueError as e:
        error_message = str(e)
        if "contains" in error_message and "To open a specific plugin" in error_message:
            # Extract available plugin names
            plugin_names = [line.strip().strip('"') for line in error_message.split("\n") if line.startswith('\t"')]

            # Ask user to select a specific plugin within the file
            print("\nThis VST3 file contains multiple plugins:")
            for i, name in enumerate(plugin_names):
                print(f"{i + 1}. {name}")

            while True:
                try:
                    sub_selection = int(input("\nEnter the number of the plugin you want to load: ")) - 1
                    if 0 <= sub_selection < len(plugin_names):
                        selected_plugin_name = plugin_names[sub_selection]
                        plugin = pedalboard.load_plugin(selected_plugin, plugin_name=selected_plugin_name)
                        break
                    else:
                        print("Invalid selection. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
        else:
            raise  # Re-raise unexpected errors

    preset_file = "preset.vstpreset"

    # Check if a preset already exists
    if os.path.exists(preset_file):
        print("\nLoading existing preset...")
        with open(preset_file, "rb") as f:
            existing_preset_data = f.read()
            plugin.preset_data = existing_preset_data  # Load preset into plugin

    # Open the VST GUI and let the user tweak parameters
    print("\nOpening VST... Adjust parameters and close the window when done.")
    plugin.show_editor()

    # Prompt the user for a preset name
    preset_name = input("\nEnter a name for this preset: ").strip()

    # Generate a unique ID
    preset_uid = str(uuid.uuid4())[:8]  # Shorter UID for cleaner filenames

    # Construct the preset filename
    preset_filename = f"{preset_name}_{preset_uid}.vstpreset"
    preset_path = os.path.join(PRESET_DIR, preset_filename)

    # Save the VST preset data to the generated file
    with open(preset_path, "wb") as f:
        f.write(plugin.preset_data)

    print(f"\nPreset saved to {preset_path}")

    # Path to `presets.json`
    preset_index_file = os.path.join(PRESET_DIR, "presets.json")

    # Load existing presets.json (if it exists)
    if os.path.exists(preset_index_file):
        with open(preset_index_file, "r") as f:
            try:
                presets_data = json.load(f)
            except json.JSONDecodeError:
                presets_data = []  # If JSON is corrupted, reset it
    else:
        presets_data = []

    # Add the new preset entry
    presets_data.append({"plugin": selected_plugin, "preset_path": preset_path})

    # Save updated presets.json
    with open(preset_index_file, "w") as f:
        json.dump(presets_data, f, indent=4)

    print("\nUpdated presets.json successfully!")

if __name__ == "__main__":
    main()