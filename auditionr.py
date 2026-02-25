"""
Auditionr: sample audition and processing logic. Registers routes and uses
shared app/socketio when used from the unified webui.
"""

import os
import shutil
import subprocess
import sys

from flask import request, jsonify, send_from_directory
from utils import (
    get_latest_exports,
    get_auditionr_folder_counts,
    get_presets,
    delete_all_files,
    EXPORTS_DIR,
    ARCHIVE_DIR,
    SAVED_DIR,
    TRASH_DIR,
)
from generate_midi import get_patterns
from generate_sample import apply_effect
from process_sample import (
    trim_sample_start,
    trim_sample_end,
    fade_sample_start,
    fade_sample_end,
    increase_sample_gain,
    decrease_sample_gain,
    reverse_sample,
    apply_granular_synthesis,
    apply_reverb_to_sample,
    apply_reverb_room_to_sample,
    apply_reverb_hall_to_sample,
    apply_reverb_large_to_sample,
    apply_distortion_to_sample,
    apply_compress_to_sample,
    apply_overdrive_mids_to_sample,
    apply_chorus_to_sample,
    apply_flanger_to_sample,
    apply_phaser_to_sample,
    apply_lowpass_to_sample,
    apply_highpass_to_sample,
    apply_bandpass_to_sample,
    apply_eq_lows_to_sample,
    apply_eq_mids_to_sample,
    apply_eq_highs_to_sample,
)

# Injected by register_auditionr(app, socketio); used by view functions for emits.
_socketio = None


def _emit_folder_counts():
    if _socketio:
        _socketio.emit("folder_counts", get_auditionr_folder_counts())


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

    shutil.move(file_path, os.path.join(TRASH_DIR, file_name))

    _socketio.emit("exports", {"files": get_latest_exports()})
    _emit_folder_counts()
    return jsonify({"success": "File moved to trash."}), 200


def refresh_configs():
    _socketio.emit("configs", {"presets": get_presets(), "patterns": get_patterns()})
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
        case "granularize_sample":
            apply_granular_synthesis(file_path)
        case "reverb_sample":
            apply_reverb_to_sample(file_path)
        case "reverb_room_sample":
            apply_reverb_room_to_sample(file_path)
        case "reverb_hall_sample":
            apply_reverb_hall_to_sample(file_path)
        case "reverb_large_sample":
            apply_reverb_large_to_sample(file_path)
        case "compress_sample":
            apply_compress_to_sample(file_path)
        case "overdrive_mids_sample":
            apply_overdrive_mids_to_sample(file_path)
        case "distort_sample":
            apply_distortion_to_sample(file_path)
        case "chorus_sample":
            apply_chorus_to_sample(file_path)
        case "flanger_sample":
            apply_flanger_to_sample(file_path)
        case "phaser_sample":
            apply_phaser_to_sample(file_path)
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
        case _:
            return jsonify({"error": "Command not recognized"}), 400

    _socketio.emit(
        "exports", {"files": get_latest_exports(sort_override=params["files"])}
    )
    _emit_folder_counts()
    return jsonify({"success": f"File processed with {command}"}), 200


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


def register_auditionr(app, socketio):
    """Register auditionr routes on the given Flask app. Socketio is used for emits."""
    global _socketio
    _socketio = socketio

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
        "/refresh", "auditionr_refresh_configs", refresh_configs, methods=["GET"]
    )
    app.add_url_rule("/save", "auditionr_save_file", save_file, methods=["POST"])
    app.add_url_rule(
        "/process", "auditionr_process_file", process_file, methods=["POST"]
    )
    app.add_url_rule(
        "/reveal", "auditionr_reveal", reveal_in_explorer, methods=["POST"]
    )


if __name__ == "__main__":
    from webui import run

    run()
