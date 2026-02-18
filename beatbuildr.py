"""
Beatbuildr: drum kit and pattern builder logic. Registers routes and socket
handlers when used from the unified webui.
"""

import json
import os
import random
import shutil
from pathlib import Path

from dotenv import load_dotenv
from flask import send_from_directory

from utils import (
    TEMP_DIR,
    delete_all_files,
)

# Injected by register_beatbuildr(app, socketio); used by socket handlers for emits.
_socketio = None

DRUM_ROW_ORDER = [
    "kick",
    "snar",
    "ghos",
    "clap",
    "hhat",
    "halt",
    "shkr",
    "prca",
    "prcb",
    "prcc",
    "tomm",
    "cymb",
]


ENV_TO_ROW_MAPPING = {
    "kick": "DRUM_KICK_PATHS",
    "snar": "DRUM_SNARE_PATHS",
    "ghos": "DRUM_SNARE_PATHS",
    "clap": "DRUM_CLAP_PATHS",
    "hhat": "DRUM_HIHAT_PATHS",
    "halt": "DRUM_HIHAT_PATHS",
    "shkr": "DRUM_SHAKER_PATHS",
    "prca": "DRUM_PERC_PATHS",
    "prcb": "DRUM_PERC_PATHS",
    "prcc": "DRUM_PERC_PATHS",
    "tomm": "DRUM_TOM_PATHS",
    "cymb": "DRUM_CYMBAL_PATHS",
}


def _choose_random_file(folder: Path) -> Path | None:
    """Return a random .wav file path from a folder, or None if unavailable."""
    if not folder.exists() or not folder.is_dir():
        return None
    candidates = [
        f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".wav"
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _get_random_sample_for_env(env_key: str) -> Path | None:
    """
    Pick a random WAV file from one of the comma-separated paths in the given env var.
    Returns a small descriptor dict with name and path, or None if not resolvable.
    """
    paths = os.getenv(env_key, "")
    roots = [p.strip() for p in paths.split(",") if p.strip()]
    if not roots:
        return None

    root = Path(random.choice(roots))
    return _choose_random_file(root)


def generate_random_drum_kit() -> dict:
    """
    Build a random drum kit using the same env-configured sample folders
    that power the `generate-beat` CLI command.
    """
    load_dotenv()

    # Prepare temp directory to host the current kit's samples so they can be
    # served over HTTP to the browser.
    kit_temp_root = Path(TEMP_DIR) / "beatbuildr"
    kit_temp_root.mkdir(parents=True, exist_ok=True)
    delete_all_files(str(kit_temp_root))

    kit: dict[str, dict | None] = {}
    for row in DRUM_ROW_ORDER:
        env_key = ENV_TO_ROW_MAPPING.get(row)
        if not env_key:
            continue
        src_path = _get_random_sample_for_env(env_key)
        if not src_path:
            kit[row] = None
            continue

        filename = f"{row}_{src_path.name}"
        dest_path = kit_temp_root / filename
        try:
            shutil.copy2(src_path, dest_path)
        except Exception:
            kit[row] = None
            continue

        kit[row] = {
            "name": src_path.stem,
            "path": str(src_path),
            "url": f"/kit-samples/{filename}",
        }

    return {"rows": DRUM_ROW_ORDER, "kit": kit, "base_url": "/kit-samples"}


def replace_sample_for_row(row: str) -> dict | None:
    """
    Replace a single row's sample with a new random one from the same env.
    Overwrites the file in the existing kit temp dir. Returns descriptor or None.
    """
    load_dotenv()
    env_key = ENV_TO_ROW_MAPPING.get(row)
    if not env_key:
        return None
    src_path = _get_random_sample_for_env(env_key)
    if not src_path:
        return None
    kit_temp_root = Path(TEMP_DIR) / "beatbuildr"
    kit_temp_root.mkdir(parents=True, exist_ok=True)
    filename = f"{row}_{src_path.name}"
    dest_path = kit_temp_root / filename
    try:
        shutil.copy2(src_path, dest_path)
    except Exception:
        return None
    return {
        "name": src_path.stem,
        "path": str(src_path),
        "url": f"/kit-samples/{filename}",
    }


def _handle_request_new_kit():
    """Client requested a full new random drum kit."""
    drum_kit = generate_random_drum_kit()
    _socketio.emit("kit", drum_kit)


def _handle_replace_sample(payload):
    """Replace a single row's sample. Payload: { "row": "kick" }."""
    row = (payload or {}).get("row")
    if row not in DRUM_ROW_ORDER:
        return
    descriptor = replace_sample_for_row(row)
    if descriptor:
        _socketio.emit("sampleReplaced", {"row": row, **descriptor})


def _handle_save_pattern(payload):
    """
    Save a beat pattern to config/beat-patterns.json.
    Payload: { "name": str, "pattern": dict, "overwrite": bool }
    """
    name = (payload or {}).get("name", "").strip()
    pattern = (payload or {}).get("pattern")
    overwrite = (payload or {}).get("overwrite", False)

    if not name:
        _socketio.emit("patternSaveResult", {"error": "Pattern name is required"})
        return

    if not pattern or not isinstance(pattern, dict):
        _socketio.emit("patternSaveResult", {"error": "Invalid pattern data"})
        return

    expected_rows = set(DRUM_ROW_ORDER)
    if set(pattern.keys()) != expected_rows:
        _socketio.emit("patternSaveResult", {"error": "Invalid pattern structure"})
        return

    beat_patterns_file = "config/beat-patterns.json"

    try:
        if os.path.exists(beat_patterns_file):
            with open(beat_patterns_file, "r") as f:
                patterns = json.load(f)
        else:
            patterns = {}

        if name in patterns and not overwrite:
            _socketio.emit(
                "patternSaveResult", {"needsConfirmation": True, "name": name}
            )
            return

        patterns[name] = pattern

        os.makedirs(os.path.dirname(beat_patterns_file), exist_ok=True)
        with open(beat_patterns_file, "w") as f:
            f.write("{\n")
            pattern_items = list(patterns.items())
            for i, (pattern_name, pattern_data) in enumerate(pattern_items):
                is_last_pattern = i == len(pattern_items) - 1
                f.write(f'  "{pattern_name}": {{\n')

                row_items = list(pattern_data.items())
                for j, (row_key, row_values) in enumerate(row_items):
                    is_last_row = j == len(row_items) - 1
                    values_str = json.dumps(row_values)
                    comma = "" if is_last_row else ","
                    f.write(f'    "{row_key}": {values_str}{comma}\n')

                comma = "" if is_last_pattern else ","
                f.write(f"  }}{comma}\n")
            f.write("}\n")

        _socketio.emit("patternSaveResult", {"success": True, "name": name})

    except Exception as e:
        _socketio.emit(
            "patternSaveResult", {"error": f"Failed to save pattern: {str(e)}"}
        )


def serve_kit_sample(filename: str):
    """Serve the currently selected drum kit samples to the browser."""
    kit_temp_root = Path(TEMP_DIR) / "beatbuildr"
    return send_from_directory(kit_temp_root, filename)


def ensure_beat_patterns():
    """Ensure config/beat-patterns.json exists, copy from sample if needed."""
    beat_patterns_file = "config/beat-patterns.json"
    beat_patterns_sample = "resources/beat-patterns-sample.json"

    if not os.path.exists(beat_patterns_file):
        os.makedirs(os.path.dirname(beat_patterns_file), exist_ok=True)
        if os.path.exists(beat_patterns_sample):
            shutil.copy2(beat_patterns_sample, beat_patterns_file)
            print(f"Created {beat_patterns_file} from sample template")


def register_beatbuildr(app, socketio):
    """Register beatbuildr routes and socket handlers on the given app and socketio."""
    global _socketio
    _socketio = socketio

    app.add_url_rule(
        "/kit-samples/<path:filename>",
        "beatbuildr_serve_kit_sample",
        serve_kit_sample,
    )

    socketio.on_event("requestNewKit", _handle_request_new_kit)
    socketio.on_event("replaceSample", _handle_replace_sample)
    socketio.on_event("savePattern", _handle_save_pattern)


if __name__ == "__main__":
    from webui import run

    run()
