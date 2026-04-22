"""
Auditionr: sample audition and processing logic. Registers routes and uses
shared app/socketio when used from the unified webui. Includes generate
sidebar (drone, bass, transition) — same backend as former generatr view.
"""

import os
import shutil
import subprocess
import sys
import hashlib
import json
import random
import fnmatch

from flask import request, jsonify, send_from_directory
from settings import (
    ensure_settings,
    get_active_drum_path_preset_name,
    set_active_drum_path_preset,
)
from utils import (
    get_latest_exports,
    get_auditionr_folder_counts,
    get_presets,
    delete_all_files,
    EXPORTS_DIR,
    ARCHIVE_DIR,
    SAVED_DIR,
    TRASH_DIR,
    TEMP_DIR,
    format_name,
    generate_beat_name,
    generate_drone_name,
    generate_id,
    with_main_prompt as with_prompt,
    RED,
    RESET,
)
from generate_midi import get_patterns, generate_drone_midi, get_pattern_config
from processing_actions import (
    get_processing_actions_payload,
    parse_post_processing_spec,
    apply_post_processing_actions,
)
from generate_sample import apply_effect, generate_drone_sample, generate_beat_sample
from generate_transition import (
    generate_closh_sample,
    generate_kickboom_sample,
    generate_sweep_sample,
    generate_longcrash_sample,
    generate_riser_sample,
    generate_drop_sample,
    parse_closh_config,
    parse_sweep_config,
)
from generate_bass import (
    generate_donk_sample,
    generate_reese_sample,
    parse_donk_config,
    parse_reese_config,
)
from beatbuildr import generate_random_drum_kit
from process_sample import (
    process_drone_sample,
    trim_sample_start,
    trim_sample_end,
    fade_sample_start,
    fade_sample_end,
    increase_sample_gain,
    decrease_sample_gain,
    reverse_sample,
    apply_time_stretch_simple,
    apply_pitch_shift_preserve_length,
    apply_granular_synthesis,
    apply_reverb_to_sample,
    apply_reverb_bedroom_to_sample,
    apply_reverb_room_to_sample,
    apply_reverb_hall_to_sample,
    apply_reverb_large_to_sample,
    apply_reverb_amphitheatre_to_sample,
    apply_reverb_space_to_sample,
    apply_distortion_to_sample,
    apply_distortion_mild_to_sample,
    apply_distortion_medium_to_sample,
    apply_distortion_heavy_to_sample,
    apply_compress_to_sample,
    apply_compress_mild_to_sample,
    apply_compress_medium_to_sample,
    apply_compress_heavy_to_sample,
    apply_overdrive_mids_to_sample,
    apply_overdrive_mild_to_sample,
    apply_overdrive_medium_to_sample,
    apply_overdrive_heavy_to_sample,
    apply_chorus_to_sample,
    apply_chorus_mild_to_sample,
    apply_chorus_medium_to_sample,
    apply_chorus_heavy_to_sample,
    apply_flanger_to_sample,
    apply_flanger_mild_to_sample,
    apply_flanger_medium_to_sample,
    apply_flanger_heavy_to_sample,
    apply_phaser_to_sample,
    apply_phaser_mild_to_sample,
    apply_phaser_medium_to_sample,
    apply_phaser_heavy_to_sample,
    apply_lowpass_to_sample,
    apply_highpass_to_sample,
    apply_bandpass_to_sample,
    apply_eq_lows_to_sample,
    apply_eq_mids_to_sample,
    apply_eq_highs_to_sample,
)

# Injected by register_auditionr(app, socketio); used by view functions for emits.
_socketio = None
UNDO_DIR = os.path.join(TEMP_DIR, "auditionr_undo")
PITCH_DIR = os.path.join(TEMP_DIR, "auditionr_pitch")
PITCH_STATE_FILE = os.path.join(PITCH_DIR, "state.json")


def _ensure_undo_dir():
    os.makedirs(UNDO_DIR, exist_ok=True)


def _ensure_pitch_dir():
    os.makedirs(PITCH_DIR, exist_ok=True)


def _load_pitch_state() -> dict:
    _ensure_pitch_dir()
    if not os.path.exists(PITCH_STATE_FILE):
        return {}
    try:
        with open(PITCH_STATE_FILE, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_pitch_state(state: dict) -> None:
    _ensure_pitch_dir()
    with open(PITCH_STATE_FILE, "w") as f:
        json.dump(state, f)


def _pitch_snapshot_path(file_path: str) -> str:
    _ensure_pitch_dir()
    key = hashlib.sha1(os.path.abspath(file_path).encode("utf-8")).hexdigest()
    return os.path.join(PITCH_DIR, f"{key}_base.wav")


def _clear_pitch_state_for_file(file_path: str) -> None:
    state = _load_pitch_state()
    abs_path = os.path.abspath(file_path)
    if abs_path in state:
        del state[abs_path]
        _save_pitch_state(state)
    snap = _pitch_snapshot_path(file_path)
    if os.path.exists(snap):
        os.remove(snap)


def _apply_pitch_with_fixed_base(file_path: str, semitones_delta: float) -> None:
    """
    Apply pitch shift cumulatively from a fixed base snapshot (not from the
    most recently processed output), preventing transient drift across repeated
    pitch operations.
    """
    abs_path = os.path.abspath(file_path)
    state = _load_pitch_state()
    entry = state.get(abs_path)
    base_snapshot = _pitch_snapshot_path(file_path)

    if not entry or not os.path.exists(base_snapshot):
        shutil.copy2(file_path, base_snapshot)
        cumulative = float(semitones_delta)
    else:
        cumulative = float(entry.get("semitones", 0.0)) + float(semitones_delta)

    # Rebuild from the fixed base every time.
    shutil.copy2(base_snapshot, file_path)
    if abs(cumulative) > 1e-9:
        apply_pitch_shift_preserve_length(file_path, cumulative)

    state[abs_path] = {"semitones": cumulative}
    _save_pitch_state(state)


def _undo_snapshot_path(file_path: str) -> str:
    _ensure_undo_dir()
    key = hashlib.sha1(os.path.abspath(file_path).encode("utf-8")).hexdigest()
    return os.path.join(UNDO_DIR, f"{key}.wav")


def _save_undo_snapshot(file_path: str):
    if not os.path.exists(file_path):
        return
    snapshot = _undo_snapshot_path(file_path)
    shutil.copy2(file_path, snapshot)


def _clear_undo_snapshot(file_path: str):
    snapshot = _undo_snapshot_path(file_path)
    if os.path.exists(snapshot):
        os.remove(snapshot)


def _has_undo_snapshot(file_path: str) -> bool:
    return os.path.exists(_undo_snapshot_path(file_path))


def _undo_availability_for_files(files):
    availability = {}
    for file in files:
        normalized = file.lstrip("./")
        availability[file] = _has_undo_snapshot(f"./{normalized}")
    return availability


def _emit_folder_counts():
    if _socketio:
        _socketio.emit("folder_counts", get_auditionr_folder_counts())


def _open_files_with_default_player(file_paths):
    if not file_paths:
        return
    try:
        if sys.platform.startswith("darwin"):
            subprocess.run(["open"] + file_paths, check=False)
        elif sys.platform.startswith("win"):
            for file_path in file_paths:
                os.startfile(file_path)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open"] + file_paths, check=False)
    except Exception as e:
        print(with_prompt(f"Failed to open generated files: {e}"))


def serve_exported_file(filename):
    """Allows direct access to exported .wav files"""
    return send_from_directory(EXPORTS_DIR, filename)


def skip_file():
    params = request.get_json() or {}
    if not params["path"]:
        return jsonify({"error": "File path is required."}), 400

    file_path = f".{params['path']}"

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist."}), 404

    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)

    file_name = file_path.split(f"{EXPORTS_DIR}/")[1]
    _clear_undo_snapshot(file_path)
    _clear_pitch_state_for_file(file_path)

    shutil.move(file_path, os.path.join(ARCHIVE_DIR, file_name))

    _socketio.emit("exports", {"files": get_latest_exports()})
    _emit_folder_counts()
    return jsonify({"success": "File moved to archive."}), 200


def unarchive_files():
    if not os.path.exists(ARCHIVE_DIR):
        return jsonify({"error": "Archive does not exist"}), 404

    if not os.path.exists(EXPORTS_DIR):
        return jsonify({"error": "Exports do not exist"}), 404

    move_all_files(ARCHIVE_DIR, EXPORTS_DIR)

    _socketio.emit("exports", {"files": get_latest_exports()})
    _emit_folder_counts()
    return jsonify({"success": "Files moved back from archive"}), 200


def empty_trash():
    if not os.path.exists(TRASH_DIR):
        return jsonify({"error": "Trash does not exist"}), 404

    delete_all_files(TRASH_DIR)

    _emit_folder_counts()
    return jsonify({"success": "Trash has been emptied"}), 200


def reprocess():
    params = request.get_json() or {}
    if not params["path"]:
        return jsonify({"error": "File path is required."}), 400

    file_path = f".{params['path']}"

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist."}), 404

    apply_effect(file_path, params["effect"])

    _socketio.emit("exports", {"files": get_latest_exports()})
    _emit_folder_counts()
    _socketio.emit("status", {"done": True})
    return jsonify({"success": "File moved to archive."}), 200


def move_all_files(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files = os.listdir(source_dir)

    for file in files:
        shutil.move(os.path.join(source_dir, file), target_dir)


def delete_file():
    params = request.get_json() or {}
    if not params["path"]:
        return jsonify({"error": "File path is required."}), 400

    file_path = f".{params['path']}"

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist."}), 404

    if not os.path.exists(TRASH_DIR):
        os.makedirs(TRASH_DIR)

    file_name = file_path.split(f"{EXPORTS_DIR}/")[1]
    _clear_undo_snapshot(file_path)
    _clear_pitch_state_for_file(file_path)

    shutil.move(file_path, os.path.join(TRASH_DIR, file_name))

    _socketio.emit("exports", {"files": get_latest_exports()})
    _emit_folder_counts()
    return jsonify({"success": "File moved to trash."}), 200


def duplicate_file():
    params = request.get_json() or {}
    if not params.get("path"):
        return jsonify({"error": "File path is required."}), 400

    file_path = f".{params['path']}"
    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist."}), 404

    source_dir = os.path.dirname(file_path)
    source_name = os.path.basename(file_path)
    source_stem, source_ext = os.path.splitext(source_name)
    duplicate_name = f"{source_stem}_copy{source_ext}"
    duplicate_path = os.path.join(source_dir, duplicate_name)

    counter = 2
    while os.path.exists(duplicate_path):
        duplicate_name = f"{source_stem}_copy{counter}{source_ext}"
        duplicate_path = os.path.join(source_dir, duplicate_name)
        counter += 1

    shutil.copy2(file_path, duplicate_path)

    _socketio.emit("exports", {"files": get_latest_exports()})
    _emit_folder_counts()
    return jsonify({"success": f"File duplicated as {duplicate_name}"}), 200


def refresh_configs():
    _socketio.emit(
        "configs",
        {
            "presets": get_presets(),
            "patterns": get_patterns(),
            "processingActions": get_processing_actions_payload(),
        },
    )
    return jsonify({"success": "Refreshed configurations"}), 200


def save_file():
    params = request.get_json() or {}
    if not params["path"]:
        return jsonify({"error": "File path is required."}), 400

    file_path = f".{params['path']}"

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist."}), 404

    if not os.path.exists(SAVED_DIR):
        os.makedirs(SAVED_DIR)

    file_name = file_path.split(f"{EXPORTS_DIR}/")[1]
    _clear_undo_snapshot(file_path)
    _clear_pitch_state_for_file(file_path)

    shutil.move(file_path, os.path.join(SAVED_DIR, file_name))

    _socketio.emit("exports", {"files": get_latest_exports()})
    _emit_folder_counts()
    return jsonify({"success": "File moved to saved."}), 200


def process_file():
    params = request.get_json() or {}
    if not params["path"]:
        return jsonify({"error": "File path is required."}), 400

    file_path = f".{params['path']}"

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist."}), 404

    if not params["command"]:
        return jsonify({"error": "File command is required."}), 400

    command = params["command"]
    if command != "undo_last_edit":
        _save_undo_snapshot(file_path)
    if command != "pitch_shift_sample":
        _clear_pitch_state_for_file(file_path)

    match command:
        case "trim_sample_start":
            trim_sample_start(file_path, params["seconds"])
        case "trim_sample_end":
            trim_sample_end(file_path, params["seconds"])
        case "fade_sample_start":
            fade_sample_start(file_path, params["seconds"])
        case "fade_sample_end":
            fade_sample_end(file_path, params["seconds"])
        case "increase_sample_gain":
            increase_sample_gain(file_path, params["db"])
        case "decrease_sample_gain":
            decrease_sample_gain(file_path, params["db"])
        case "reverse_sample":
            reverse_sample(file_path)
        case "stretch_sample":
            apply_time_stretch_simple(file_path, params.get("stretch_factor", 1.0))
        case "pitch_shift_sample":
            _apply_pitch_with_fixed_base(file_path, params.get("semitones", 0))
        case "granularize_sample":
            apply_granular_synthesis(file_path)
        case "reverb_sample":
            apply_reverb_to_sample(file_path)
        case "reverb_bedroom_sample":
            apply_reverb_bedroom_to_sample(file_path)
        case "reverb_room_sample":
            apply_reverb_room_to_sample(file_path)
        case "reverb_hall_sample":
            apply_reverb_hall_to_sample(file_path)
        case "reverb_large_sample":
            apply_reverb_large_to_sample(file_path)
        case "reverb_amphitheatre_sample":
            apply_reverb_amphitheatre_to_sample(file_path)
        case "reverb_space_sample":
            apply_reverb_space_to_sample(file_path)
        case "compress_sample":
            apply_compress_to_sample(file_path)
        case "compress_mild_sample":
            apply_compress_mild_to_sample(file_path)
        case "compress_medium_sample":
            apply_compress_medium_to_sample(file_path)
        case "compress_heavy_sample":
            apply_compress_heavy_to_sample(file_path)
        case "overdrive_mids_sample":
            apply_overdrive_mids_to_sample(file_path)
        case "overdrive_mild_sample":
            apply_overdrive_mild_to_sample(file_path)
        case "overdrive_medium_sample":
            apply_overdrive_medium_to_sample(file_path)
        case "overdrive_heavy_sample":
            apply_overdrive_heavy_to_sample(file_path)
        case "distort_sample":
            apply_distortion_to_sample(file_path)
        case "distort_mild_sample":
            apply_distortion_mild_to_sample(file_path)
        case "distort_medium_sample":
            apply_distortion_medium_to_sample(file_path)
        case "distort_heavy_sample":
            apply_distortion_heavy_to_sample(file_path)
        case "chorus_sample":
            apply_chorus_to_sample(file_path)
        case "chorus_mild_sample":
            apply_chorus_mild_to_sample(file_path)
        case "chorus_medium_sample":
            apply_chorus_medium_to_sample(file_path)
        case "chorus_heavy_sample":
            apply_chorus_heavy_to_sample(file_path)
        case "flanger_sample":
            apply_flanger_to_sample(file_path)
        case "flanger_mild_sample":
            apply_flanger_mild_to_sample(file_path)
        case "flanger_medium_sample":
            apply_flanger_medium_to_sample(file_path)
        case "flanger_heavy_sample":
            apply_flanger_heavy_to_sample(file_path)
        case "phaser_sample":
            apply_phaser_to_sample(file_path)
        case "phaser_mild_sample":
            apply_phaser_mild_to_sample(file_path)
        case "phaser_medium_sample":
            apply_phaser_medium_to_sample(file_path)
        case "phaser_heavy_sample":
            apply_phaser_heavy_to_sample(file_path)
        case "lpf_sample":
            apply_lowpass_to_sample(file_path, cutoff_hz=params.get("cutoff_hz", 6000))
        case "hpf_sample":
            apply_highpass_to_sample(file_path, cutoff_hz=params.get("cutoff_hz", 100))
        case "bpf_sample":
            low = params.get("low_hz", 300)
            high = params.get("high_hz", 6000)
            apply_bandpass_to_sample(file_path, low_hz=low, high_hz=high)
        case "eq_lows_sample":
            apply_eq_lows_to_sample(file_path, params.get("db", 0))
        case "eq_mids_sample":
            apply_eq_mids_to_sample(file_path, params.get("db", 0))
        case "eq_highs_sample":
            apply_eq_highs_to_sample(file_path, params.get("db", 0))
        case "undo_last_edit":
            snapshot = _undo_snapshot_path(file_path)
            if not os.path.exists(snapshot):
                return jsonify({"error": "No undo snapshot available"}), 400
            shutil.copy2(snapshot, file_path)
            _clear_undo_snapshot(file_path)
            _clear_pitch_state_for_file(file_path)
        case _:
            return jsonify({"error": "Command not recognized"}), 400

    files = get_latest_exports(sort_override=params["files"])
    undo_available = _undo_availability_for_files(files)
    _socketio.emit(
        "exports",
        {
            "files": files,
            "updated_path": params["path"].lstrip("./"),
            "undo_available": undo_available,
        },
    )
    _emit_folder_counts()
    return jsonify({"success": f"File processed with {command}"}), 200


def undo_status():
    params = request.get_json() or {}
    if not params.get("path"):
        return jsonify({"error": "File path is required."}), 400

    file_path = f".{params['path']}"
    return jsonify({"can_undo": _has_undo_snapshot(file_path)}), 200


def reveal_in_explorer():
    """Reveal the .wav file in the system file manager (Finder, Explorer, or Linux)."""
    params = request.get_json() or {}
    if not params.get("path"):
        return jsonify({"error": "File path is required."}), 400

    file_path = os.path.abspath(f".{params['path']}")

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist."}), 404

    try:
        if sys.platform == "darwin":
            subprocess.run(["open", "-R", file_path], check=True)
        elif sys.platform == "win32":
            subprocess.run(
                ["explorer", "/select," + file_path],
                check=True,
                shell=False,
            )
        else:
            # Linux: open parent directory (xdg-open selects folder)
            parent = os.path.dirname(file_path)
            subprocess.run(["xdg-open", parent], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return jsonify({"error": f"Could not reveal file: {e}"}), 500

    return jsonify({"success": "Revealed in file manager."}), 200


# ---------------------------------------------------------------------------
# Generate sidebar (drone, bass, transition) — same logic as former generatr
# ---------------------------------------------------------------------------


def _run_generate_drone():
    """One iteration of drone generation. Returns list of paths. Logs to server stdout."""
    if not os.path.exists("presets/presets.json"):
        raise FileNotFoundError(
            "presets/presets.json does not exist, please run build_preset.py"
        )
    filters = {}
    midi_file, selected_chart = generate_drone_midi(
        pattern=None,
        shift_octave_down=None,
        shift_root_note=None,
        filters=filters,
        notes=None,
    )
    base_sample_name = (
        f"{generate_drone_name()}_-_{selected_chart}_-_{generate_id()}"
    )
    sample_name = format_name(f"drone___{base_sample_name}")
    output_path = f"{EXPORTS_DIR}/{sample_name}"
    generated_sample = generate_drone_sample(
        input_path=midi_file,
        output_path=f"{output_path}.wav",
        instrument=None,
        effect=None,
    )
    (
        generated_sample_stretched,
        generated_sample_stretched_reverberated,
        generated_sample_stretched_reverberated_transposed,
    ) = process_drone_sample(input_path=generated_sample)
    return [
        midi_file,
        generated_sample,
        generated_sample_stretched,
        generated_sample_stretched_reverberated,
        generated_sample_stretched_reverberated_transposed,
    ]


def _run_generate_bass(subcommand: str):
    """One iteration of bass (reese | donk). Returns list of one path."""
    if subcommand == "reese":
        config = parse_reese_config(
            sound=None, movement=None, distortion=None, fx=None, disable=None
        )
        beat_name = generate_beat_name()
        name_parts = ["reese", beat_name, "170bpm", "4bars", generate_id()]
        sample_name = format_name("___".join(name_parts))
        output_path = f"{EXPORTS_DIR}/{sample_name}.wav"
        output_path, _ = generate_reese_sample(
            tempo=170, bars=4, output=output_path, config=config
        )
        print(with_prompt(f"generated: {output_path}"))
        return [output_path]
    elif subcommand == "donk":
        config = parse_donk_config(sound=None)
        beat_name = generate_beat_name()
        name_parts = ["donk", beat_name, "120bpm", "1bars", generate_id()]
        sample_name = format_name("___".join(name_parts))
        output_path = f"{EXPORTS_DIR}/{sample_name}.wav"
        output_path, _ = generate_donk_sample(
            tempo=120, bars=1, output=output_path, config=config
        )
        print(with_prompt(f"generated: {output_path}"))
        return [output_path]
    else:
        raise ValueError(f"Unknown bass subcommand: {subcommand}")


def _run_generate_transition(subcommand: str):
    """One iteration of transition (sweep | closh | kickboom | longcrash | riser | drop). Returns list of one path."""
    if subcommand == "sweep":
        config = parse_sweep_config(
            sound=None, curve=None, filter_str=None, tremolo=None,
            phaser=None, chorus=None, flanger=None, disable=None,
        )
        beat_name = generate_beat_name()
        name_parts = ["transition_sweep", beat_name, "120bpm", "8bars", generate_id()]
        sample_name = format_name("___".join(name_parts))
        output_path = f"{EXPORTS_DIR}/{sample_name}.wav"
        output_path, _ = generate_sweep_sample(
            tempo=120, bars=8, output=output_path, config=config
        )
        print(with_prompt(f"generated: {output_path}"))
        return [output_path]
    elif subcommand == "closh":
        config = parse_closh_config(reverb=None, delay=None)
        beat_name = generate_beat_name()
        name_parts = ["transition_closh", beat_name, "120bpm", "4bars", generate_id()]
        sample_name = format_name("___".join(name_parts))
        output_path = f"{EXPORTS_DIR}/{sample_name}.wav"
        output_path, _ = generate_closh_sample(
            tempo=120, bars=4, output=output_path, config=config
        )
        print(with_prompt(f"generated: {output_path}"))
        return [output_path]
    elif subcommand == "kickboom":
        config = parse_closh_config(reverb=None, delay=None)
        beat_name = generate_beat_name()
        name_parts = ["transition_kickboom", beat_name, "120bpm", "4bars", generate_id()]
        sample_name = format_name("___".join(name_parts))
        output_path = f"{EXPORTS_DIR}/{sample_name}.wav"
        output_path, _ = generate_kickboom_sample(
            tempo=120, bars=4, output=output_path, config=config
        )
        print(with_prompt(f"generated: {output_path}"))
        return [output_path]
    elif subcommand == "longcrash":
        config = parse_closh_config(reverb=None, delay=None)
        beat_name = generate_beat_name()
        name_parts = ["transition_longcrash", beat_name, "120bpm", "8bars", generate_id()]
        sample_name = format_name("___".join(name_parts))
        output_path = f"{EXPORTS_DIR}/{sample_name}.wav"
        output_path, _ = generate_longcrash_sample(
            tempo=120, bars=8, output=output_path, config=config
        )
        print(with_prompt(f"generated: {output_path}"))
        return [output_path]
    elif subcommand == "riser":
        longcrash_config = parse_closh_config(reverb=None, delay=None)
        sweep_config = parse_sweep_config()
        beat_name = generate_beat_name()
        name_parts = ["transition_riser", beat_name, "120bpm", "4bars", generate_id()]
        sample_name = format_name("___".join(name_parts))
        output_path = f"{EXPORTS_DIR}/{sample_name}.wav"
        output_path, _ = generate_riser_sample(
            tempo=120, bars=4, output=output_path,
            longcrash_config=longcrash_config, sweep_config=sweep_config,
        )
        print(with_prompt(f"generated: {output_path}"))
        return [output_path]
    elif subcommand == "drop":
        longcrash_config = parse_closh_config(reverb=None, delay=None)
        sweep_config = parse_sweep_config()
        beat_name = generate_beat_name()
        name_parts = ["transition_drop", beat_name, "120bpm", "4bars", generate_id()]
        sample_name = format_name("___".join(name_parts))
        output_path = f"{EXPORTS_DIR}/{sample_name}.wav"
        output_path, _ = generate_drop_sample(
            tempo=120, bars=4, output=output_path,
            longcrash_config=longcrash_config, sweep_config=sweep_config,
        )
        print(with_prompt(f"generated: {output_path}"))
        return [output_path]
    else:
        raise ValueError(f"Unknown transition subcommand: {subcommand}")


def _load_beat_patterns_for_generate() -> dict:
    path = "config/beat-patterns.json"
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _load_drum_kits_for_generate() -> dict:
    path = "config/drum-kits.json"
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


API_SUPPORTED_TEMPO_VARIANTS: dict[str, float] = {
    "1/2": 0.5,
    "1/4": 0.25,
    "3/4": 0.75,
}


def _parse_api_variants(raw) -> list[str]:
    """Normalize API variants payload (list or comma-separated string)."""
    out: list[str] = []
    if isinstance(raw, str):
        items = [x.strip() for x in raw.split(",")]
    elif isinstance(raw, list):
        items = [str(x).strip() for x in raw]
    else:
        items = []
    for item in items:
        if item in API_SUPPORTED_TEMPO_VARIANTS and item not in out:
            out.append(item)
    return out


def _run_generate_beat(payload: dict):
    beat_patterns = _load_beat_patterns_for_generate()
    available_patterns = list(beat_patterns.keys())
    if not available_patterns:
        raise ValueError("No patterns found in config/beat-patterns.json")

    drum_kits = _load_drum_kits_for_generate()

    tempo_raw = payload.get("tempo")
    tempo = int(tempo_raw) if tempo_raw not in (None, "", False) else None
    loops = max(1, int(payload.get("loops", 2)))
    pattern = (payload.get("pattern") or "").strip()
    kit = (payload.get("kit") or "").strip()
    randomization_config = (payload.get("sampleLibrary") or "").strip()
    variants = _parse_api_variants(payload.get("variants", []))
    # Backward compatibility for old UI payload.
    if bool(payload.get("halfTempoVariant", False)) and "1/2" not in variants:
        variants.append("1/2")
    swing = float(payload.get("swing", 0.0))
    iterations = max(1, int(payload.get("iterations", 1)))

    post_processing = payload.get("postProcessing")
    if isinstance(post_processing, list):
        post_processing_spec = ",".join(str(p).strip() for p in post_processing if str(p).strip())
    else:
        post_processing_spec = post_processing
    post_actions = parse_post_processing_spec(post_processing_spec)

    if pattern:
        if "*" in pattern:
            matching_patterns = [p for p in available_patterns if fnmatch.fnmatch(p, pattern)]
            if not matching_patterns:
                raise ValueError(f"No patterns match '{pattern}'")
        else:
            if pattern not in available_patterns:
                raise ValueError(f"Pattern '{pattern}' not found in config/beat-patterns.json")
            matching_patterns = [pattern]
    else:
        matching_patterns = None

    available_kits = list(drum_kits.keys())
    if kit:
        if "*" in kit:
            matching_kits = [k for k in available_kits if fnmatch.fnmatch(k, kit)]
            if not matching_kits:
                raise ValueError(f"No kits match '{kit}'")
        else:
            if kit not in drum_kits:
                raise ValueError(f"Kit '{kit}' not found in config/drum-kits.json")
            matching_kits = [kit]
    else:
        matching_kits = None

    original_preset_name = None
    if not matching_kits and randomization_config:
        original_preset_name = get_active_drum_path_preset_name()
        ok, result = set_active_drum_path_preset(randomization_config)
        if not ok:
            raise ValueError(result)
        print(with_prompt(f"Using drum randomization preset: {result}"))

    paths: list[str] = []
    try:
        for _ in range(iterations):
            current_pattern = (
                random.choice(matching_patterns)
                if matching_patterns is not None
                else random.choice(available_patterns)
            )
            current_kit_name = ""
            current_kit_paths = None
            if matching_kits is not None:
                current_kit_name = random.choice(matching_kits)
                kp = drum_kits.get(current_kit_name, {})
                current_kit_paths = kp if isinstance(kp, dict) else {}
            else:
                # Freeze one concrete random kit per iteration so tempo-variant exports
                # remain true tempo variants of the same underlying sample choices.
                random_kit_payload = generate_random_drum_kit()
                random_kit_rows = random_kit_payload.get("kit", {})
                current_kit_paths = {}
                if isinstance(random_kit_rows, dict):
                    for row_key, descriptor in random_kit_rows.items():
                        if not isinstance(descriptor, dict):
                            continue
                        path = descriptor.get("path")
                        if isinstance(path, str) and path.strip():
                            current_kit_paths[row_key] = path.strip()

            raw = beat_patterns.get(current_pattern, {})
            meta_tempo = None
            meta_swing = None
            pattern_config = None
            if isinstance(raw, dict) and raw:
                gs, ts, ln, meta_tempo, meta_swing = get_pattern_config(raw)
                if gs is not None:
                    pattern_config = {"gridSize": gs, "timeSignature": ts, "length": ln}

            current_tempo = (
                tempo
                if tempo is not None
                else int(meta_tempo)
                if meta_tempo is not None
                else random.randint(80, 180)
            )
            current_swing = meta_swing if meta_swing is not None else swing
            beat_name = generate_beat_name()
            tempo_variants = [current_tempo]
            for token in variants:
                factor = API_SUPPORTED_TEMPO_VARIANTS.get(token)
                if factor is None:
                    continue
                vtempo = max(1, int(round(current_tempo * factor)))
                if vtempo not in tempo_variants:
                    tempo_variants.append(vtempo)

            for export_tempo in tempo_variants:
                name_parts = ["drumpattern", beat_name, current_pattern]
                if current_kit_name:
                    name_parts.append(format_name(current_kit_name))
                name_parts.append(f"{export_tempo}bpm")
                name_parts.append(generate_id())
                sample_name = format_name("___".join(name_parts))
                output_path = f"{EXPORTS_DIR}/{sample_name}.wav"

                generate_beat_sample(
                    bpm=export_tempo,
                    bars=loops,
                    output=output_path,
                    style=current_pattern,
                    swing=current_swing,
                    play=False,
                    pattern_config=pattern_config,
                    kit_paths=current_kit_paths,
                    pattern_data=raw if isinstance(raw, dict) else None,
                )
                if post_actions:
                    apply_post_processing_actions(output_path, post_actions)
                paths.append(output_path)
    finally:
        if original_preset_name:
            set_active_drum_path_preset(original_preset_name)

    return paths


def _handle_api_generate_options():
    ensure_settings()
    return jsonify(
        {
            "patterns": sorted(_load_beat_patterns_for_generate().keys()),
            "kits": sorted(_load_drum_kits_for_generate().keys()),
            "processingActions": get_processing_actions_payload(),
        }
    )


def _handle_api_generate():
    """
    POST /api/generatr/generate
    Body: { "type": "drone" | "bass" | "transition" | "beat", "subcommand": optional }
    Returns: { "paths": [...], "error": null } or { "paths": [], "error": "..." }
    """
    ensure_settings()
    data = request.get_json() or {}
    gen_type = (data.get("type") or "").strip().lower()
    subcommand = (data.get("subcommand") or "").strip().lower()

    if not gen_type:
        return jsonify({"paths": [], "error": "Missing type (drone, bass, transition, or beat)"}), 400
    if gen_type not in ("drone", "bass", "transition", "beat"):
        return jsonify({"paths": [], "error": f"Unknown type: {gen_type}"}), 400
    if gen_type == "bass" and subcommand not in ("reese", "donk"):
        return jsonify({"paths": [], "error": "Bass requires subcommand: reese or donk"}), 400
    if gen_type == "transition" and subcommand not in ("sweep", "closh", "kickboom", "longcrash", "riser", "drop"):
        return jsonify(
            {"paths": [], "error": "Transition requires subcommand: sweep, closh, kickboom, longcrash, riser, or drop"}
        ), 400

    try:
        print(f"{RED}│{RESET} generate: {gen_type}" + (f" {subcommand}" if subcommand else ""))
        if gen_type == "drone":
            paths = _run_generate_drone()
        elif gen_type == "bass":
            paths = _run_generate_bass(subcommand)
        elif gen_type == "transition":
            paths = _run_generate_transition(subcommand)
        else:
            paths = _run_generate_beat(data)
        print(f"{RED}■ generate completed{RESET}")
        _socketio.emit("exports", {"files": get_latest_exports()})
        _emit_folder_counts()
        return jsonify({"paths": paths, "error": None})
    except Exception as e:
        print(with_prompt(f"generate error: {e}"))
        return jsonify({"paths": [], "error": str(e)}), 500


def _handle_request_exports():
    """Socket handler: client requests latest exports and folder counts (e.g. after generate)."""
    _socketio.emit("exports", {"files": get_latest_exports()})
    _emit_folder_counts()


def register_auditionr(app, socketio):
    """Register auditionr routes on the given Flask app. Socketio is used for emits."""
    global _socketio
    _socketio = socketio

    socketio.on_event("requestExports", _handle_request_exports)

    app.add_url_rule(
        "/exports/<path:filename>",
        "auditionr_serve_exported_file",
        serve_exported_file,
    )
    app.add_url_rule("/skip", "auditionr_skip_file", skip_file, methods=["POST"])
    app.add_url_rule(
        "/unarchive", "auditionr_unarchive_files", unarchive_files, methods=["GET"]
    )
    app.add_url_rule(
        "/emptytrash", "auditionr_empty_trash", empty_trash, methods=["GET"]
    )
    app.add_url_rule("/reprocess", "auditionr_reprocess", reprocess, methods=["POST"])
    app.add_url_rule("/delete", "auditionr_delete_file", delete_file, methods=["POST"])
    app.add_url_rule(
        "/duplicate", "auditionr_duplicate_file", duplicate_file, methods=["POST"]
    )
    app.add_url_rule(
        "/refresh", "auditionr_refresh_configs", refresh_configs, methods=["GET"]
    )
    app.add_url_rule("/save", "auditionr_save_file", save_file, methods=["POST"])
    app.add_url_rule(
        "/process", "auditionr_process_file", process_file, methods=["POST"]
    )
    app.add_url_rule(
        "/reveal", "auditionr_reveal", reveal_in_explorer, methods=["POST"]
    )
    app.add_url_rule(
        "/undo-status", "auditionr_undo_status", undo_status, methods=["POST"]
    )
    app.add_url_rule(
        "/api/generatr/generate",
        "auditionr_api_generate",
        _handle_api_generate,
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/generatr/options",
        "auditionr_api_generate_options",
        _handle_api_generate_options,
        methods=["GET"],
    )


if __name__ == "__main__":
    from webui import run

    run()
