"""
Beatbuildr: drum kit and pattern builder logic. Registers routes and socket
handlers when used from the unified webui.
"""

import base64
import json
import os
import random
import shutil
from pathlib import Path

from flask import request, send_file
from flask import send_from_directory
from settings import get_setting

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
    Pick a random WAV file from one of the comma-separated paths in the given setting.
    Returns a small descriptor dict with name and path, or None if not resolvable.
    """
    paths = get_setting(env_key, "")
    roots = [p.strip() for p in paths.split(",") if p.strip()]
    if not roots:
        return None

    root = Path(random.choice(roots))
    return _choose_random_file(root)


def generate_random_drum_kit() -> dict:
    """
    Build a random drum kit using the same settings-configured sample folders
    that power the `generate-beat` CLI command.
    """
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
    Replace a single row's sample with a new random one from the same setting.
    Overwrites the file in the existing kit temp dir. Returns descriptor or None.
    """
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


# --- Kick sample browser: cached list, recursive scan ---
KICK_CACHE_FILE = os.path.join(TEMP_DIR, "sample-cache-kicks.json")
_kick_samples_cache: list[dict] = []  # [{"path": str, "name": str}, ...]
_kick_samples_paths_set: set[str] = set()  # For path validation when serving


def _collect_kick_samples_recursive(root: Path, results: list[dict], seen: set[str]) -> None:
    """Recursively collect .wav and .aiff files under root. Modifies results and seen in place."""
    if not root.exists() or not root.is_dir():
        return
    try:
        for entry in root.iterdir():
            if entry.is_file() and entry.suffix.lower() in (".wav", ".aiff"):
                path_str = str(entry.resolve())
                if path_str not in seen:
                    seen.add(path_str)
                    results.append({"path": path_str, "name": entry.stem})
            elif entry.is_dir():
                _collect_kick_samples_recursive(entry, results, seen)
    except (PermissionError, OSError):
        pass


def _collect_kick_samples() -> list[dict]:
    """Scan DRUM_KICK_PATHS recursively for all .wav and .aiff files. Returns list of {path, name}."""
    paths_str = get_setting("DRUM_KICK_PATHS", "")
    roots = [p.strip() for p in paths_str.split(",") if p.strip()]
    results: list[dict] = []
    seen: set[str] = set()
    for root_str in roots:
        root = Path(root_str).expanduser().resolve()
        _collect_kick_samples_recursive(root, results, seen)
    return sorted(results, key=lambda x: (x["name"].lower(), x["path"]))


def _load_kick_cache_from_file() -> bool:
    """Load kick samples from temp cache file. Returns True if loaded."""
    global _kick_samples_cache, _kick_samples_paths_set
    if not os.path.exists(KICK_CACHE_FILE):
        return False
    try:
        with open(KICK_CACHE_FILE, "r") as f:
            data = json.load(f)
        items = data.get("samples") if isinstance(data, dict) else []
        if isinstance(items, list):
            _kick_samples_cache = [x for x in items if isinstance(x, dict) and "path" in x and "name" in x]
            _kick_samples_paths_set = {x["path"] for x in _kick_samples_cache}
            return True
    except Exception:
        pass
    return False


def _save_kick_cache_to_file() -> None:
    """Write kick samples cache to temp file."""
    os.makedirs(os.path.dirname(KICK_CACHE_FILE), exist_ok=True)
    try:
        with open(KICK_CACHE_FILE, "w") as f:
            json.dump({"samples": _kick_samples_cache}, f)
    except Exception:
        pass


def get_kick_samples(force_refresh: bool = False) -> list[dict]:
    """Return cached kick samples. Lazy-loads from file or scans on first call. Use force_refresh to rescan."""
    global _kick_samples_cache, _kick_samples_paths_set
    if force_refresh:
        _kick_samples_cache = _collect_kick_samples()
        _kick_samples_paths_set = {x["path"] for x in _kick_samples_cache}
        _save_kick_cache_to_file()
        return _kick_samples_cache
    if _kick_samples_cache:
        return _kick_samples_cache
    if _load_kick_cache_from_file():
        return _kick_samples_cache
    _kick_samples_cache = _collect_kick_samples()
    _kick_samples_paths_set = {x["path"] for x in _kick_samples_cache}
    _save_kick_cache_to_file()
    return _kick_samples_cache


def serve_sample_preview():
    """Serve a sample file for preview. Query param 'p' = base64-encoded path. Path must be in allowed set."""
    global _kick_samples_paths_set
    if not _kick_samples_paths_set:
        get_kick_samples(force_refresh=False)
    enc = request.args.get("p", "")
    if not enc:
        return {"error": "Missing path"}, 400
    try:
        path_bytes = base64.urlsafe_b64decode(enc)
        path_str = path_bytes.decode("utf-8")
    except Exception:
        return {"error": "Invalid path"}, 400
    path_str = os.path.normpath(path_str)
    if path_str not in _kick_samples_paths_set:
        return {"error": "Path not allowed"}, 403
    p = Path(path_str)
    if not p.exists() or not p.is_file():
        return {"error": "File not found"}, 404
    mimetype = "audio/aiff" if p.suffix.lower() in (".aiff", ".aif") else "audio/wav"
    return send_file(p, mimetype=mimetype, as_attachment=False)


def replace_sample_with_path(row: str, path_str: str) -> dict | None:
    """Replace a row's sample with a file at the given path. Copies to kit temp, returns descriptor."""
    if row not in DRUM_ROW_ORDER:
        return None
    path_str = path_str.strip()
    if not path_str or path_str not in _kick_samples_paths_set:
        return None
    src_path = Path(path_str)
    if not src_path.exists() or not src_path.is_file():
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


DRUM_KITS_FILE = "config/drum-kits.json"
DRUM_KITS_SAMPLE = "resources/drum-kits-sample.json"


def ensure_drum_kits():
    """Ensure config/drum-kits.json exists, copy from sample if needed."""
    if not os.path.exists(DRUM_KITS_FILE):
        os.makedirs(os.path.dirname(DRUM_KITS_FILE), exist_ok=True)
        if os.path.exists(DRUM_KITS_SAMPLE):
            shutil.copy2(DRUM_KITS_SAMPLE, DRUM_KITS_FILE)
            print(f"Created {DRUM_KITS_FILE} from sample template")


def _load_drum_kits_config() -> dict:
    """Load drum kits from config/drum-kits.json."""
    if not os.path.exists(DRUM_KITS_FILE):
        return {}
    try:
        with open(DRUM_KITS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def load_kit_by_name(kit_name: str) -> dict | None:
    """
    Load a drum kit from config by name. Copies sample files to temp and returns
    the same structure as generate_random_drum_kit(). Returns None if kit not found
    or invalid.
    """
    kits = _load_drum_kits_config()
    kit_data = kits.get(kit_name)
    if not kit_data or not isinstance(kit_data, dict):
        return None

    kit_temp_root = Path(TEMP_DIR) / "beatbuildr"
    kit_temp_root.mkdir(parents=True, exist_ok=True)
    delete_all_files(str(kit_temp_root))

    kit: dict[str, dict | None] = {}
    for row in DRUM_ROW_ORDER:
        path_str = (kit_data or {}).get(row)
        if not path_str or not isinstance(path_str, str):
            kit[row] = None
            continue

        src_path = Path(path_str)
        if not src_path.exists() or not src_path.is_file():
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


def _handle_get_drum_kits():
    """Return list of saved drum kit names to the client."""
    kits = _load_drum_kits_config()
    _socketio.emit("drumKits", {"kits": list(kits.keys())})


def _handle_load_kit(payload):
    """Load a drum kit by name. Payload: { "name": str }."""
    name = (payload or {}).get("name", "").strip()
    if not name:
        _socketio.emit("kitLoadResult", {"error": "Kit name is required"})
        return

    drum_kit = load_kit_by_name(name)
    if drum_kit:
        _socketio.emit("kit", drum_kit)
        _socketio.emit("kitLoadResult", {"success": True, "name": name})
    else:
        _socketio.emit("kitLoadResult", {"error": f"Kit '{name}' not found or invalid"})


def _handle_save_kit(payload):
    """
    Save current drum kit to config/drum-kits.json.
    Payload: { "name": str, "kit": dict, "overwrite": bool }
    kit: { rowKey: { "path": str }, ... }
    """
    name = (payload or {}).get("name", "").strip()
    kit = (payload or {}).get("kit")
    overwrite = (payload or {}).get("overwrite", False)

    if not name:
        _socketio.emit("kitSaveResult", {"error": "Kit name is required"})
        return

    if not kit or not isinstance(kit, dict):
        _socketio.emit("kitSaveResult", {"error": "Invalid kit data"})
        return

    # Build paths dict from kit: only include rows with valid path
    paths: dict[str, str] = {}
    for row in DRUM_ROW_ORDER:
        desc = kit.get(row)
        if desc and isinstance(desc, dict):
            p = desc.get("path")
            if p and isinstance(p, str) and p.strip():
                paths[row] = p.strip()

    if not paths:
        _socketio.emit("kitSaveResult", {"error": "No valid samples to save"})
        return

    try:
        kits = _load_drum_kits_config()

        if name in kits and not overwrite:
            _socketio.emit(
                "kitSaveResult", {"needsConfirmation": True, "name": name}
            )
            return

        kits[name] = paths

        os.makedirs(os.path.dirname(DRUM_KITS_FILE), exist_ok=True)
        with open(DRUM_KITS_FILE, "w") as f:
            f.write("{\n")
            items = list(kits.items())
            for i, (kit_name, kit_data) in enumerate(items):
                comma = "" if i == len(items) - 1 else ","
                paths_json = json.dumps(kit_data)
                f.write(f'  "{kit_name}": {paths_json}{comma}\n')
            f.write("}\n")

        _socketio.emit("kitSaveResult", {"success": True, "name": name})
        _socketio.emit("drumKits", {"kits": list(kits.keys())})

    except Exception as e:
        _socketio.emit(
            "kitSaveResult", {"error": f"Failed to save kit: {str(e)}"}
        )


def _handle_get_kick_samples():
    """Return cached list of kick samples to the client."""
    samples = get_kick_samples(force_refresh=False)
    _socketio.emit("kickSamples", {"samples": samples})


def _handle_refresh_sample_paths():
    """Rescan kick paths and emit updated list."""
    samples = get_kick_samples(force_refresh=True)
    _socketio.emit("kickSamples", {"samples": samples})


def _handle_replace_sample_with_path(payload):
    """Replace a row's sample with a specific file. Payload: { "row": str, "path": str }."""
    row = (payload or {}).get("row")
    path_str = (payload or {}).get("path")
    if not row or row not in DRUM_ROW_ORDER:
        return
    if not path_str or not isinstance(path_str, str):
        return
    descriptor = replace_sample_with_path(row, path_str)
    if descriptor:
        _socketio.emit("sampleReplaced", {"row": row, **descriptor})


def register_beatbuildr(app, socketio):
    """Register beatbuildr routes and socket handlers on the given app and socketio."""
    global _socketio
    _socketio = socketio

    app.add_url_rule(
        "/kit-samples/<path:filename>",
        "beatbuildr_serve_kit_sample",
        serve_kit_sample,
    )
    app.add_url_rule(
        "/api/sample-preview",
        "beatbuildr_serve_sample_preview",
        serve_sample_preview,
        methods=["GET"],
    )

    socketio.on_event("requestNewKit", _handle_request_new_kit)
    socketio.on_event("replaceSample", _handle_replace_sample)
    socketio.on_event("savePattern", _handle_save_pattern)
    socketio.on_event("getDrumKits", _handle_get_drum_kits)
    socketio.on_event("loadKit", _handle_load_kit)
    socketio.on_event("saveKit", _handle_save_kit)
    socketio.on_event("getKickSamples", _handle_get_kick_samples)
    socketio.on_event("refreshSamplePaths", _handle_refresh_sample_paths)
    socketio.on_event("replaceSampleWithPath", _handle_replace_sample_with_path)


if __name__ == "__main__":
    from webui import run

    run()
