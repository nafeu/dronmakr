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
import time
import tempfile

from flask import request, jsonify, send_from_directory, send_file
from dronmakr.core.settings import (
    ensure_settings,
    get_active_drum_path_preset_name,
    has_configured_drum_paths,
    has_configured_plugin_paths,
    set_active_drum_path_preset,
)
from dronmakr.core.utils import (
    BLUE,
    allocate_dragged_saved_filename,
    get_latest_exports,
    get_auditionr_folder_counts,
    get_presets,
    delete_all_files,
    format_name,
    generate_beat_name,
    generate_drone_name,
    generate_id,
    with_main_prompt as with_prompt,
    process_drone_sample_header,
    with_process_drone_sample_prompt,
    RED,
    RESET,
)
import dronmakr.core.utils as managed_paths
from dronmakr.core.chord_scale_catalog import get_chord_scale_catalog, get_chord_scale_picklists, warm_chord_scale_picklists
from dronmakr.generate.generate_midi import (
    coerce_drone_midi_length_bars,
    coerce_drone_midi_padding_bars,
    coerce_drone_midi_tempo_bpm,
    generate_drone_midi,
    get_pattern_config,
    get_patterns,
    get_patterns_catalog,
    build_midi_preview_payload,
    format_pattern_display_name,
    write_drone_midi_temp,
    extract_drone_piano_notes_from_midi_bytes,
)
from dronmakr.processing.processing_actions import (
    append_post_processing_shortcut,
    get_processing_actions_payload,
    parse_post_processing_spec,
    parse_single_processing_spec,
    apply_post_processing_actions,
    apply_processing_command,
    read_post_processing_shortcuts_document,
    remove_post_processing_shortcut,
)
from dronmakr.generate.generate_sample import apply_effect, generate_drone_sample, generate_beat_sample, BEAT_EXPORT_PEAK_DB, effect_slot_entries
from dronmakr.core.paths import get_managed_file, normalize_path_basename
from dronmakr.generate.generate_transition import (
    generate_sweep_sample,
    generate_wash_sample,
    parse_sweep_config,
    parse_wash_config,
)
from dronmakr.generate.generate_bass import (
    generate_donk_sample,
    generate_reese_sample,
    parse_donk_config,
    parse_reese_config,
)
from dronmakr.apps.beatbuildr import generate_random_drum_kit
from dronmakr.audio.process_sample import (
    apply_pitch_shift_preserve_length,
    reverse_sample,
    double_loop_sample,
    apply_granular_synthesis,
)

# Injected by register_auditionr(app, socketio); used by view functions for emits.
_socketio = None
UNDO_DIR = ""
PITCH_DIR = ""
PITCH_STATE_FILE = ""
DRONE_AUDIO_PREVIEW_PATTERN = "quantized_straight_eighth"
DRONE_AUDIO_PREVIEW_BARS = 2
_MANAGED_AUDIO_ROOTS: list[str] = []


def refresh_auditionr_paths() -> None:
    """Rebind auditionr managed paths after FILES_ROOT changes."""
    global UNDO_DIR, PITCH_DIR, PITCH_STATE_FILE, _MANAGED_AUDIO_ROOTS

    if managed_paths.TEMP_DIR:
        UNDO_DIR = os.path.join(managed_paths.TEMP_DIR, "auditionr_undo")
        PITCH_DIR = os.path.join(managed_paths.TEMP_DIR, "auditionr_pitch")
        PITCH_STATE_FILE = os.path.join(PITCH_DIR, "state.json")
    else:
        UNDO_DIR = ""
        PITCH_DIR = ""
        PITCH_STATE_FILE = ""
    _MANAGED_AUDIO_ROOTS = [
        p
        for p in (
            managed_paths.EXPORTS_DIR,
            managed_paths.ARCHIVE_DIR,
            managed_paths.SAVED_DIR,
            managed_paths.TRASH_DIR,
        )
        if p
    ]


def _socket_broadcast(event: str, payload) -> None:
    """Emit to all connected clients (required from HTTP request handlers)."""
    if _socketio:
        # Flask-SocketIO: omit `to` to reach every client; do not pass `broadcast` (unsupported).
        _socketio.emit(event, payload)


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
        resolved = _resolve_managed_audio_path(file)
        availability[file] = _has_undo_snapshot(resolved) if resolved else False
    return availability


def _emit_folder_counts():
    if _socketio:
        _socket_broadcast("folder_counts", get_auditionr_folder_counts())


def _emit_configs_to_clients():
    """Broadcast presets, patterns, and processing catalog (shortcuts) to connected clients."""
    if not _socketio:
        return
    _socket_broadcast(
        "configs",
        {
            "presets": get_presets(),
            "patterns": get_patterns(),
            "processingActions": get_processing_actions_payload(),
        },
    )


def _resolve_managed_audio_path(path_value: str) -> str | None:
    """Resolve UI/API path tokens to a concrete managed audio file path."""
    raw = str(path_value or "").strip()
    if not raw:
        return None

    if raw.startswith("/exports/"):
        slug = raw[len("/exports/") :].strip()
        candidate = os.path.join(managed_paths.EXPORTS_DIR, normalize_path_basename(slug))
    elif raw.startswith("/archive/"):
        slug = raw[len("/archive/") :].strip()
        candidate = os.path.join(managed_paths.ARCHIVE_DIR, normalize_path_basename(slug))
    elif raw.startswith("/saved/"):
        slug = raw[len("/saved/") :].strip()
        candidate = os.path.join(managed_paths.SAVED_DIR, normalize_path_basename(slug))
    elif raw.startswith("/trash/"):
        slug = raw[len("/trash/") :].strip()
        candidate = os.path.join(managed_paths.TRASH_DIR, normalize_path_basename(slug))
    else:
        candidate = raw

    abs_candidate = os.path.abspath(candidate)
    for root in _MANAGED_AUDIO_ROOTS:
        abs_root = os.path.abspath(root)
        if abs_candidate == abs_root or abs_candidate.startswith(abs_root + os.sep):
            return abs_candidate
    return None


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
    safe_name = normalize_path_basename(filename)
    return send_from_directory(managed_paths.EXPORTS_DIR, safe_name)


def skip_file():
    params = request.get_json() or {}
    if not params["path"]:
        return jsonify({"error": "File path is required."}), 400

    file_path = _resolve_managed_audio_path(params["path"])
    if not file_path:
        return jsonify({"error": "File path is not allowed."}), 400

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist."}), 404

    if not os.path.exists(managed_paths.ARCHIVE_DIR):
        os.makedirs(managed_paths.ARCHIVE_DIR)

    file_name = os.path.basename(file_path)
    _clear_undo_snapshot(file_path)
    _clear_pitch_state_for_file(file_path)

    shutil.move(file_path, os.path.join(managed_paths.ARCHIVE_DIR, file_name))

    _socket_broadcast("exports", {"files": get_latest_exports()})
    _emit_folder_counts()
    return jsonify({"success": "File moved to archive."}), 200


def unarchive_files():
    if not os.path.exists(managed_paths.ARCHIVE_DIR):
        return jsonify({"error": "Archive does not exist"}), 404

    if not os.path.exists(managed_paths.EXPORTS_DIR):
        return jsonify({"error": "Exports do not exist"}), 404

    move_all_files(managed_paths.ARCHIVE_DIR, managed_paths.EXPORTS_DIR)

    _socket_broadcast("exports", {"files": get_latest_exports()})
    _emit_folder_counts()
    return jsonify({"success": "Files moved back from archive"}), 200


def empty_trash():
    if not os.path.exists(managed_paths.TRASH_DIR):
        return jsonify({"error": "Trash does not exist"}), 404

    delete_all_files(managed_paths.TRASH_DIR)

    _emit_folder_counts()
    return jsonify({"success": "Trash has been emptied"}), 200


def reprocess():
    params = request.get_json() or {}
    if not params["path"]:
        return jsonify({"error": "File path is required."}), 400

    file_path = _resolve_managed_audio_path(params["path"])
    if not file_path:
        return jsonify({"error": "File path is not allowed."}), 400

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist."}), 404

    apply_effect(file_path, params["effect"])

    _socket_broadcast("exports", {"files": get_latest_exports()})
    _emit_folder_counts()
    _socket_broadcast("status", {"done": True})
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

    file_path = _resolve_managed_audio_path(params["path"])
    if not file_path:
        return jsonify({"error": "File path is not allowed."}), 400

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist."}), 404

    if not os.path.exists(managed_paths.TRASH_DIR):
        os.makedirs(managed_paths.TRASH_DIR)

    file_name = os.path.basename(file_path)
    _clear_undo_snapshot(file_path)
    _clear_pitch_state_for_file(file_path)

    shutil.move(file_path, os.path.join(managed_paths.TRASH_DIR, file_name))

    _socket_broadcast("exports", {"files": get_latest_exports()})
    _emit_folder_counts()
    return jsonify({"success": "File moved to trash."}), 200


def duplicate_file():
    params = request.get_json() or {}
    if not params.get("path"):
        return jsonify({"error": "File path is required."}), 400

    file_path = _resolve_managed_audio_path(params["path"])
    if not file_path:
        return jsonify({"error": "File path is not allowed."}), 400
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
    # copy2 preserves source mtime; the queue is ordered by mtime (newest first),
    # so the duplicate would often sort below the original. Bump mtime to now.
    _now = time.time()
    os.utime(duplicate_path, (_now, _now))

    _socket_broadcast("exports", {"files": get_latest_exports()})
    _emit_folder_counts()
    return jsonify({"success": f"File duplicated as {duplicate_name}"}), 200


def refresh_configs():
    _emit_configs_to_clients()
    return jsonify({"success": "Refreshed configurations"}), 200


def prepare_drag_copy():
    """Copy a managed sample into saved/ with a _copied suffix for stable DAW references."""
    params = request.get_json() or {}
    if not params.get("path"):
        return jsonify({"error": "File path is required."}), 400

    file_path = _resolve_managed_audio_path(params["path"])
    if not file_path:
        return jsonify({"error": "File path is not allowed."}), 400

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist."}), 404

    os.makedirs(managed_paths.SAVED_DIR, exist_ok=True)
    dest_name = allocate_dragged_saved_filename(os.path.basename(file_path), managed_paths.SAVED_DIR)
    dest_path = os.path.join(managed_paths.SAVED_DIR, dest_name)
    shutil.copy2(file_path, dest_path)

    return jsonify(
        {
            "success": True,
            "absPath": os.path.abspath(dest_path),
            "savedPath": f"/saved/{dest_name}",
            "name": dest_name.replace(".wav", ""),
        }
    ), 200


def save_file():
    params = request.get_json() or {}
    if not params["path"]:
        return jsonify({"error": "File path is required."}), 400

    file_path = _resolve_managed_audio_path(params["path"])
    if not file_path:
        return jsonify({"error": "File path is not allowed."}), 400

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist."}), 404

    if not os.path.exists(managed_paths.SAVED_DIR):
        os.makedirs(managed_paths.SAVED_DIR)

    file_name = os.path.basename(file_path)
    _clear_undo_snapshot(file_path)
    _clear_pitch_state_for_file(file_path)

    shutil.move(file_path, os.path.join(managed_paths.SAVED_DIR, file_name))

    _socket_broadcast("exports", {"files": get_latest_exports()})
    _emit_folder_counts()
    return jsonify({"success": "File moved to saved."}), 200


def process_file():
    params = request.get_json() or {}
    if not params["path"]:
        return jsonify({"error": "File path is required."}), 400

    file_path = _resolve_managed_audio_path(params["path"])
    if not file_path:
        return jsonify({"error": "File path is not allowed."}), 400

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist."}), 404

    processing_spec = (params.get("processing_spec") or "").strip()
    command = (params.get("command") or "").strip()

    if not processing_spec and not command:
        return jsonify({"error": "File command or processing_spec is required."}), 400

    skip_keys = frozenset({"path", "command", "files", "processing_spec"})
    inner_params = {k: v for k, v in params.items() if k not in skip_keys}

    inner_command = command
    if processing_spec:
        try:
            action = parse_single_processing_spec(processing_spec)
            inner_command = action.get("command", "")
            inner_params = dict(action.get("params") or {})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    if inner_command != "undo_last_edit":
        _save_undo_snapshot(file_path)
    if inner_command != "pitch_shift_sample":
        _clear_pitch_state_for_file(file_path)

    match inner_command:
        case "undo_last_edit":
            snapshot = _undo_snapshot_path(file_path)
            if not os.path.exists(snapshot):
                return jsonify({"error": "No undo snapshot available"}), 400
            shutil.copy2(snapshot, file_path)
            _clear_undo_snapshot(file_path)
            _clear_pitch_state_for_file(file_path)
        case "pitch_shift_sample":
            _apply_pitch_with_fixed_base(file_path, inner_params.get("semitones", 0))
        case "reverse_sample":
            reverse_sample(file_path)
        case "double_loop_sample":
            double_loop_sample(file_path)
        case "granularize_sample":
            apply_granular_synthesis(file_path)
        case _:
            try:
                apply_processing_command(file_path, inner_command, inner_params)
            except ValueError:
                return jsonify({"error": "Command not recognized"}), 400

    files = get_latest_exports(sort_override=params["files"])
    undo_available = _undo_availability_for_files(files)
    _socket_broadcast(
        "exports",
        {
            "files": files,
            # Keep updated_path in the same shape as entries returned by get_latest_exports()
            # so the frontend can detect and refresh the modified waveform.
            "updated_path": file_path,
            "undo_available": undo_available,
        },
    )
    _emit_folder_counts()
    return jsonify({"success": f"File processed with {inner_command}"}), 200


def undo_status():
    params = request.get_json() or {}
    if not params.get("path"):
        return jsonify({"error": "File path is required."}), 400

    file_path = _resolve_managed_audio_path(params["path"])
    if not file_path:
        return jsonify({"error": "File path is not allowed."}), 400
    return jsonify({"can_undo": _has_undo_snapshot(file_path)}), 200


def reveal_in_explorer():
    """Reveal the .wav file in the system file manager (Finder, Explorer, or Linux)."""
    params = request.get_json() or {}
    if not params.get("path"):
        return jsonify({"error": "File path is required."}), 400

    file_path = _resolve_managed_audio_path(params["path"])
    if not file_path:
        return jsonify({"error": "File path is not allowed."}), 400

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


def _normalize_post_processing_spec_from_payload(payload: dict) -> str | None:
    """Match beat/drone: list of commands joins to comma-separated spec."""
    post_processing = payload.get("postProcessing")
    if isinstance(post_processing, list):
        joined = ",".join(str(p).strip() for p in post_processing if str(p).strip())
        return joined if joined else None
    if post_processing is None:
        return None
    s = str(post_processing).strip()
    return s if s else None


def _optional_trimmed(payload: dict, key: str) -> str | None:
    raw = payload.get(key)
    if raw is None:
        return None
    s = str(raw).strip()
    return s if s else None


def _positive_int_field(payload: dict, key: str, default: int) -> int:
    try:
        v = int(payload.get(key, default))
    except (TypeError, ValueError):
        v = default
    return max(1, v)


def _float_field(payload: dict, key: str, default: float) -> float:
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def _optional_bool_field(payload: dict, key: str) -> bool | None:
    if key not in payload:
        return None
    raw = payload.get(key)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    if isinstance(raw, (int, float)):
        return bool(raw)
    return None


def _optional_float_field(payload: dict, key: str) -> float | None:
    if key not in payload or payload.get(key) is None:
        return None
    try:
        return float(payload[key])
    except (TypeError, ValueError):
        return None


def _optional_int_field(payload: dict, key: str) -> int | None:
    if key not in payload or payload.get(key) is None:
        return None
    try:
        return int(payload[key])
    except (TypeError, ValueError):
        return None


def _apply_drone_post_processing_to_wavs(
    wav_paths: list[str], post_processing: str | None
) -> None:
    if not wav_paths:
        return
    spec = (
        post_processing.strip()
        if isinstance(post_processing, str) and post_processing.strip()
        else None
    )
    try:
        actions = parse_post_processing_spec(spec)
    except ValueError as e:
        raise ValueError(str(e)) from e

    def _pp_step_banner(i: int, total: int, action: dict) -> None:
        label = action.get("token") or action.get("command", "")
        print(with_process_drone_sample_prompt(f"[{i}/{total}] {label}"))

    def _pp_normalize_banner() -> None:
        print(with_process_drone_sample_prompt("normalize"))

    for wav_path in wav_paths:
        print(process_drone_sample_header())
        print(with_process_drone_sample_prompt(os.path.basename(wav_path)))
        apply_post_processing_actions(
            wav_path,
            actions,
            on_before_chain_step=_pp_step_banner,
            on_before_finalize_normalize=_pp_normalize_banner,
        )
        print(f"{BLUE}│{RESET}")


def _parse_drone_custom_notes(raw) -> list[str] | None:
    """Validate Auditionr custom-note payloads for ``generate_drone_midi(notes=...)``."""
    import pretty_midi

    if raw is None:
        return None
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, list):
        items = [str(part).strip() for part in raw if str(part).strip()]
    else:
        return None
    if not items:
        return None

    validated: list[str] = []
    seen: set[str] = set()
    for item in items:
        try:
            pretty_midi.note_name_to_number(item)
        except Exception as exc:
            raise ValueError(f"Invalid custom note '{item}'") from exc
        if item in seen:
            continue
        seen.add(item)
        validated.append(item)
    return validated or None


def _build_drone_midi_kwargs_from_payload(
    payload: dict,
    *,
    preview: bool = False,
    iteration_index: int = 0,
    chart_pool_last_key: str | None = None,
) -> tuple[dict, str | None]:
    """Shared MIDI generation kwargs for drone export and live preview."""
    from dronmakr.apps.drone_chart_pool import chart_entry_key, parse_drone_chart_selection, pick_chart_entry

    pattern_raw = (payload.get("pattern") or "").strip()
    pattern = pattern_raw or None
    if pattern:
        allowed = set(get_patterns())
        if pattern not in allowed:
            raise ValueError(f"Unknown MIDI pattern '{pattern}'")

    length_bars = coerce_drone_midi_length_bars(payload.get("lengthBars"))
    padded_silence_bars = coerce_drone_midi_padding_bars(payload.get("paddedSilenceBars"))
    tempo_bpm = coerce_drone_midi_tempo_bpm(payload.get("tempo"))

    midi_kwargs = {
        "pattern": pattern,
        "shift_octave_down": None,
        "shift_root_note": None,
        "num_bars": length_bars,
        "padded_silence_bars": 0 if preview else padded_silence_bars,
        "tempo_bpm": tempo_bpm,
    }

    next_chart_pool_last_key = chart_pool_last_key
    chart_selection = parse_drone_chart_selection(payload.get("chartSelection"))
    if chart_selection:
        picked = pick_chart_entry(chart_selection, iteration_index, chart_pool_last_key)
        midi_kwargs["chart_entry"] = picked
        next_chart_pool_last_key = chart_entry_key(picked)
    else:
        custom_notes = _parse_drone_custom_notes(payload.get("customNotes"))
        if custom_notes:
            midi_kwargs["notes"] = custom_notes
        else:
            midi_kwargs["filters"] = {}

    return midi_kwargs, next_chart_pool_last_key


def _parse_drone_instrument_selection(payload: dict) -> tuple[str | None, dict | None]:
    """Return legacy instrument name and optional inline plug-in selection."""
    raw = payload.get("instrumentSelection")
    if not isinstance(raw, dict):
        return (payload.get("instrument") or "").strip() or None, None
    kind = (raw.get("kind") or "").strip().lower()
    if kind == "pool":
        items = raw.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("Instrument pool must include at least one item.")
        mode = (raw.get("mode") or "random").strip().lower()
        if mode not in ("round_robin", "random", "random_unique"):
            raise ValueError("Instrument pool mode must be round_robin, random, or random_unique.")
        groups = raw.get("groups")
        return None, {
            "kind": "pool",
            "mode": mode,
            "items": items,
            "groups": groups if isinstance(groups, list) else [],
        }
    if kind == "patch":
        name = (raw.get("name") or "").strip()
        return name or None, None
    if kind == "plugin":
        plugin_path = (raw.get("pluginPath") or raw.get("plugin_path") or "").strip()
        if not plugin_path:
            raise ValueError("Instrument plug-in selection is missing pluginPath.")
        return None, {
            "kind": "plugin",
            "pluginPath": plugin_path,
            "presetPath": raw.get("presetPath") or raw.get("preset_path") or "",
            "label": raw.get("label") or raw.get("name") or "",
            "pluginName": raw.get("pluginName") or raw.get("plugin_name") or "",
        }
    if kind == "faust":
        faust_id = (raw.get("faustId") or raw.get("faust_id") or "").strip()
        if not faust_id:
            raise ValueError("Faust instrument selection is missing faustId.")
        from dronmakr.audio.faust_library import faust_path_for_id, list_faust_instruments

        label = (raw.get("label") or raw.get("name") or "").strip()
        if not label:
            match = next((entry for entry in list_faust_instruments() if entry["id"] == faust_id), None)
            label = match["label"] if match else faust_id
        return None, {
            "kind": "faust",
            "faustId": faust_id,
            "pluginPath": faust_path_for_id(faust_id),
            "label": label,
        }
    return (payload.get("instrument") or "").strip() or None, None


def _parse_drone_fx_state(payload: dict) -> tuple[list | None, str]:
    """Return FX slots and mode: ``disabled``, ``random``, or ``explicit``."""
    if payload.get("fxEnabled") is False:
        return None, "disabled"

    if "fxSlots" not in payload:
        return None, "random"

    slots = _parse_drone_fx_slots(payload)
    if slots is None:
        return None, "random"
    if any(slots):
        return slots, "explicit"
    return None, "random"


def _parse_drone_fx_slots(payload: dict) -> list | None:
    """Return FX slot list when the UI sends explicit slots; otherwise None (legacy effect field)."""
    if "fxSlots" not in payload:
        return None
    raw_slots = payload.get("fxSlots")
    if not isinstance(raw_slots, list):
        return None
    slots: list = []
    for raw in raw_slots[:5]:
        if not raw:
            slots.append(None)
            continue
        if not isinstance(raw, dict):
            slots.append(None)
            continue
        kind = (raw.get("kind") or "").strip().lower()
        if kind == "patch":
            name = (raw.get("name") or "").strip()
            if name:
                slots.append({"kind": "patch", "name": name})
            else:
                slots.append(None)
        elif kind == "plugin":
            plugin_path = (raw.get("pluginPath") or raw.get("plugin_path") or "").strip()
            if not plugin_path:
                slots.append(None)
            else:
                slots.append(
                    {
                        "kind": "plugin",
                        "pluginPath": plugin_path,
                        "presetPath": raw.get("presetPath") or raw.get("preset_path") or "",
                        "label": raw.get("label") or raw.get("name") or "",
                        "pluginName": raw.get("pluginName") or raw.get("plugin_name") or "",
                    }
                )
        elif kind == "faust":
            faust_id = (raw.get("faustId") or raw.get("faust_id") or "").strip()
            plugin_path = (raw.get("pluginPath") or raw.get("plugin_path") or "").strip()
            if plugin_path:
                from dronmakr.audio.faust_fx_library import faust_fx_id_from_path, is_faust_fx_path

                if is_faust_fx_path(plugin_path) and not faust_id:
                    faust_id = faust_fx_id_from_path(plugin_path)
            if not faust_id:
                slots.append(None)
                continue
            if not plugin_path:
                from dronmakr.audio.faust_fx_library import faust_fx_path_for_id, list_faust_effects

                plugin_path = faust_fx_path_for_id(faust_id)
            label = (raw.get("label") or raw.get("name") or "").strip()
            if not label:
                match = next((entry for entry in list_faust_effects() if entry["id"] == faust_id), None)
                label = match["label"] if match else faust_id
            slots.append(
                {
                    "kind": "faust",
                    "faustId": faust_id,
                    "pluginPath": plugin_path,
                    "label": label,
                }
            )
        else:
            slots.append(None)
    while len(slots) < 5:
        slots.append(None)
    return slots


def _resolve_drone_editor_instrument_spec(
    payload: dict,
    instruments: list[dict],
    *,
    edit_role: str,
    edit_plugin_path: str,
    edit_preset_path: str | None,
) -> tuple[str, str | None]:
    if edit_role == "instrument":
        return edit_plugin_path, edit_preset_path or None

    _instrument, instrument_selection = _parse_drone_instrument_selection(payload)
    if isinstance(instrument_selection, dict) and instrument_selection.get("kind") == "pool":
        from dronmakr.apps.drone_instrument_pool import resolve_effective_instrument_selection

        instrument_selection = resolve_effective_instrument_selection(
            instrument_selection,
            iteration_index=0,
        )
    if isinstance(instrument_selection, dict) and instrument_selection.get("kind") in ("plugin", "faust"):
        plugin_path = (
            instrument_selection.get("pluginPath")
            or instrument_selection.get("plugin_path")
            or ""
        ).strip()
        if plugin_path:
            if instrument_selection.get("kind") == "faust":
                return plugin_path, None
            preset_path = (
                instrument_selection.get("presetPath")
                or instrument_selection.get("preset_path")
                or ""
            ).strip()
            return plugin_path, preset_path or None

    if _instrument:
        for preset in instruments:
            if preset.get("name") == _instrument:
                return preset["plugin_path"], preset.get("preset_path")

    if instruments:
        pick = random.choice(instruments)
        return pick["plugin_path"], pick.get("preset_path")

    from dronmakr.audio.faust_library import list_faust_instruments, pick_random_faust_instrument_preset

    if list_faust_instruments():
        faust_pick = pick_random_faust_instrument_preset()
        return faust_pick["plugin_path"], faust_pick.get("preset_path")

    raise ValueError(
        "No instrument available for live preview — pick or save an instrument first."
    )


def _resolve_drone_editor_fx_specs(
    payload: dict,
    fx_presets: list[dict],
    *,
    edit_role: str,
    edit_plugin_path: str,
    edit_preset_path: str | None,
    fx_slot_index: int | None,
) -> list[tuple[str, str | None]]:
    _effect = (payload.get("effect") or "").strip() or None
    fx_slots, fx_mode = _parse_drone_fx_state(payload)
    if fx_mode == "disabled":
        return []

    specs: list[tuple[str, str | None]] = []
    if fx_slots is None:
        if edit_role == "effect":
            return [(edit_plugin_path, edit_preset_path or None)]
        effect_preset = None
        if _effect and _effect.lower() != "none":
            effect_preset = next((p for p in fx_presets if p.get("name") == _effect), None)
        elif fx_presets:
            effect_preset = random.choice(fx_presets)
        if effect_preset is not None:
            for step in effect_slot_entries(effect_preset):
                specs.append((step["plugin_path"], step.get("preset_path")))
        return specs

    for idx, slot in enumerate(fx_slots[:5]):
        if edit_role == "effect" and fx_slot_index == idx:
            specs.append((edit_plugin_path, edit_preset_path or None))
            continue
        if not slot:
            continue
        if slot.get("kind") == "patch":
            patch_name = (slot.get("name") or "").strip()
            patch = next((p for p in fx_presets if p.get("name") == patch_name), None)
            if patch is None:
                continue
            for step in effect_slot_entries(patch):
                specs.append((step["plugin_path"], step.get("preset_path")))
            continue
        if slot.get("kind") == "plugin":
            plugin_path = (slot.get("pluginPath") or slot.get("plugin_path") or "").strip()
            if plugin_path:
                preset_path = slot.get("presetPath") or slot.get("preset_path") or ""
                specs.append((plugin_path, preset_path or None))
    return specs


def _prepare_drone_editor_preview(
    payload: dict,
    *,
    edit_plugin_path: str,
    edit_role: str,
    edit_preset_path: str | None,
    fx_slot_index: int | None,
) -> dict:
    from dronmakr.core.utils import PRESETS_INDEX_MISSING_MSG, resolve_presets_index_path

    presets_path = resolve_presets_index_path()
    if not presets_path:
        raise FileNotFoundError(PRESETS_INDEX_MISSING_MSG)

    with open(presets_path, "r", encoding="utf-8") as f:
        presets = json.load(f)
    if not isinstance(presets, list):
        raise ValueError(f"{presets_path} must contain a JSON array of preset objects.")

    instruments = [p for p in presets if isinstance(p, dict) and p.get("type") == "instrument"]
    fx_presets = [
        p for p in presets if isinstance(p, dict) and p.get("type") in ("effect", "effect_chain")
    ]

    preview_payload = dict(payload)
    preview_payload["pattern"] = DRONE_AUDIO_PREVIEW_PATTERN
    preview_payload["lengthBars"] = DRONE_AUDIO_PREVIEW_BARS
    midi_kwargs, _chart_pool_last_key = _build_drone_midi_kwargs_from_payload(preview_payload, preview=True)
    midi_kwargs["pattern"] = DRONE_AUDIO_PREVIEW_PATTERN
    midi_obj, _chart_label, render_duration_sec, _pattern_used = generate_drone_midi(
        **midi_kwargs,
        quiet=True,
    )
    midi_temp = write_drone_midi_temp(midi_obj)

    instrument_path, instrument_state_path = _resolve_drone_editor_instrument_spec(
        preview_payload,
        instruments,
        edit_role=edit_role,
        edit_plugin_path=edit_plugin_path,
        edit_preset_path=edit_preset_path,
    )
    fx_specs = _resolve_drone_editor_fx_specs(
        preview_payload,
        fx_presets,
        edit_role=edit_role,
        edit_plugin_path=edit_plugin_path,
        edit_preset_path=edit_preset_path,
        fx_slot_index=fx_slot_index,
    )

    return {
        "midi_path": midi_temp,
        "duration_sec": render_duration_sec,
        "instrument_path": instrument_path,
        "instrument_state_path": instrument_state_path or "",
        "fx_specs": [[path, state or ""] for path, state in fx_specs],
    }


def _run_generate_drone(payload: dict) -> list[str]:
    """`generate-drone` CLI parity (omits UI for --shift-octave-down, --shift-root-note, --dry-run, --log-server, --play)."""
    from dronmakr.core.utils import PRESETS_INDEX_MISSING_MSG, resolve_presets_index_path

    presets_path = resolve_presets_index_path()
    if not presets_path:
        raise FileNotFoundError(PRESETS_INDEX_MISSING_MSG)

    instrument, instrument_selection = _parse_drone_instrument_selection(payload)
    effect = (payload.get("effect") or "").strip() or None
    fx_slots, fx_mode = _parse_drone_fx_state(payload)
    if fx_mode == "disabled":
        effect = "none"
        fx_slots = None
    elif fx_mode == "random":
        fx_slots = None

    try:
        iterations = max(1, min(50, int(payload.get("iterations", 1))))
    except (TypeError, ValueError):
        iterations = 1

    post_processing = payload.get("postProcessing")
    if isinstance(post_processing, list):
        post_processing_spec = ",".join(
            str(p).strip() for p in post_processing if str(p).strip()
        )
    else:
        post_processing_spec = (post_processing or "").strip() or None

    musical_style_mode = (payload.get("musicalStyleMode") or "custom").strip().lower()
    custom_notes = None
    chart_selection = payload.get("chartSelection")
    if musical_style_mode == "custom" and not chart_selection:
        custom_notes = _parse_drone_custom_notes(payload.get("customNotes"))

    results: list[str] = []
    pool_last_key: str | None = None
    chart_pool_last_key: str | None = None
    for iteration in range(iterations):
        midi_kwargs, chart_pool_last_key = _build_drone_midi_kwargs_from_payload(
            payload,
            preview=False,
            iteration_index=iteration,
            chart_pool_last_key=chart_pool_last_key,
        )
        midi_obj, selected_chart, render_duration_sec, _pattern_used = generate_drone_midi(
            **midi_kwargs,
            quiet=True,
        )
        chart_label = "custom" if custom_notes else selected_chart
        tempo_bpm = int(midi_kwargs.get("tempo_bpm") or 120)
        base_sample_name = (
            f"{generate_drone_name()}_-_{chart_label}_-_{tempo_bpm}bpm_-_{generate_id()}"
        )
        sample_name = format_name(f"drone___{base_sample_name}")
        output_path = f"{managed_paths.EXPORTS_DIR}/{sample_name}"
        midi_temp = write_drone_midi_temp(midi_obj)
        iteration_selection = instrument_selection
        if isinstance(instrument_selection, dict) and instrument_selection.get("kind") == "pool":
            from dronmakr.apps.drone_instrument_pool import (
                instrument_item_key,
                pick_pool_item,
                pool_item_to_instrument_selection,
            )

            items = instrument_selection.get("items") or []
            mode = instrument_selection.get("mode") or "random"
            picked = pick_pool_item(items, mode, iteration, pool_last_key)
            pool_last_key = instrument_item_key(picked)
            iteration_selection = pool_item_to_instrument_selection(picked)
        try:
            generated_sample = generate_drone_sample(
                input_path=midi_temp,
                output_path=f"{output_path}.wav",
                presets_path=presets_path,
                instrument=instrument,
                effect=effect,
                render_duration_sec=render_duration_sec,
                tempo_bpm=tempo_bpm,
                instrument_selection=iteration_selection,
                fx_slots=fx_slots,
            )
        finally:
            try:
                os.remove(midi_temp)
            except OSError:
                pass
        results.append(generated_sample)

    wav_files = [f for f in results if f.endswith(".wav")]
    _apply_drone_post_processing_to_wavs(wav_files, post_processing_spec)

    return results


def _run_drone_audio_preview(payload: dict) -> tuple[str, float, str]:
    """Render a short looping WAV preview of the current instrument + FX chain."""
    from dronmakr.core.utils import PRESETS_INDEX_MISSING_MSG, resolve_presets_index_path

    presets_path = resolve_presets_index_path()
    if not presets_path:
        raise FileNotFoundError(PRESETS_INDEX_MISSING_MSG)

    preview_payload = dict(payload)
    preview_payload["pattern"] = DRONE_AUDIO_PREVIEW_PATTERN
    preview_payload["lengthBars"] = DRONE_AUDIO_PREVIEW_BARS

    instrument, instrument_selection = _parse_drone_instrument_selection(preview_payload)
    if isinstance(instrument_selection, dict) and instrument_selection.get("kind") == "pool":
        from dronmakr.apps.drone_instrument_pool import resolve_effective_instrument_selection

        instrument_selection = resolve_effective_instrument_selection(
            instrument_selection,
            iteration_index=0,
        )
    effect = (preview_payload.get("effect") or "").strip() or None
    fx_slots, fx_mode = _parse_drone_fx_state(preview_payload)
    if fx_mode == "disabled":
        effect = "none"
        fx_slots = None
    elif fx_mode == "random":
        fx_slots = None

    midi_kwargs, _chart_pool_last_key = _build_drone_midi_kwargs_from_payload(preview_payload, preview=True)
    midi_kwargs["pattern"] = DRONE_AUDIO_PREVIEW_PATTERN
    midi_obj, chart_label, render_duration_sec, _pattern_used = generate_drone_midi(
        **midi_kwargs,
        quiet=True,
    )

    midi_temp = write_drone_midi_temp(midi_obj)
    fd, wav_temp = tempfile.mkstemp(suffix=".wav", prefix="dronmakr_drone_preview_")
    os.close(fd)
    try:
        generate_drone_sample(
            input_path=midi_temp,
            output_path=wav_temp,
            presets_path=presets_path,
            instrument=instrument,
            effect=effect,
            render_duration_sec=render_duration_sec,
            tempo_bpm=midi_kwargs.get("tempo_bpm"),
            instrument_selection=instrument_selection,
            fx_slots=fx_slots,
        )
        return wav_temp, render_duration_sec, chart_label
    finally:
        try:
            os.remove(midi_temp)
        except OSError:
            pass


def _run_generate_bass(subcommand: str, payload: dict) -> list[str]:
    """`generate-bass` CLI parity from Auditionr JSON body."""
    post_spec = _normalize_post_processing_spec_from_payload(payload)
    iterations = max(1, _positive_int_field(payload, "iterations", 1))
    paths: list[str] = []

    if subcommand == "reese":
        tempo = max(1, _positive_int_field(payload, "tempo", 170))
        bars = max(1, _positive_int_field(payload, "bars", 4))
        sound = _optional_trimmed(payload, "sound")
        movement = _optional_trimmed(payload, "movement")
        distortion = _optional_trimmed(payload, "distortion")
        fx = _optional_trimmed(payload, "fx")
        disable = _optional_trimmed(payload, "disable")
        for _ in range(iterations):
            config = parse_reese_config(
                sound=sound,
                movement=movement,
                distortion=distortion,
                fx=fx,
                disable=disable,
            )
            beat_name = generate_beat_name()
            name_parts = ["reese", beat_name, f"{tempo}bpm", f"{bars}bars", generate_id()]
            sample_name = format_name("___".join(name_parts))
            output_path = f"{managed_paths.EXPORTS_DIR}/{sample_name}.wav"
            output_path, _ = generate_reese_sample(
                tempo=tempo, bars=bars, output=output_path, config=config
            )
            paths.append(output_path)
            print(with_prompt(f"generated: {output_path}"))
    elif subcommand == "donk":
        tempo = max(1, _positive_int_field(payload, "tempo", 120))
        bars = max(1, _positive_int_field(payload, "bars", 1))
        sound = _optional_trimmed(payload, "sound")
        for _ in range(iterations):
            config = parse_donk_config(sound=sound)
            beat_name = generate_beat_name()
            name_parts = ["donk", beat_name, f"{tempo}bpm", f"{bars}bars", generate_id()]
            sample_name = format_name("___".join(name_parts))
            output_path = f"{managed_paths.EXPORTS_DIR}/{sample_name}.wav"
            output_path, _ = generate_donk_sample(
                tempo=tempo, bars=bars, output=output_path, config=config
            )
            paths.append(output_path)
            print(with_prompt(f"generated: {output_path}"))
    else:
        raise ValueError(f"Unknown bass subcommand: {subcommand}")

    _apply_drone_post_processing_to_wavs(paths, post_spec)
    return paths


def _wash_config_from_payload(payload: dict):
    percussion = _optional_trimmed(payload, "percussion")
    if percussion is None:
        percussion = _optional_trimmed(payload, "percussionType")
    library = (
        _optional_trimmed(payload, "library")
        or _optional_trimmed(payload, "sampleLibrary")
        or ""
    ).strip() or None
    return parse_wash_config(
        library=library,
        percussion=percussion,
        reverb_enabled=_optional_bool_field(payload, "reverbEnabled"),
        reverb_wet_level=_optional_float_field(payload, "reverbWetLevel"),
        reverb_length_sec=_optional_float_field(payload, "reverbLengthSec"),
        reverb_decay_sec=_optional_float_field(payload, "reverbDecaySec"),
        reverb_early_reflections=_optional_int_field(payload, "reverbEarlyReflections"),
        reverb_highpass_hz=_optional_float_field(payload, "reverbHighpassHz"),
        reverb_tail_diffusion=_optional_float_field(payload, "reverbTailDiffusion"),
        delay_enabled=_optional_bool_field(payload, "delayEnabled"),
        delay_division=_optional_trimmed(payload, "delayDivision"),
        delay_feedback=_optional_float_field(payload, "delayFeedback"),
        delay_mix=_optional_float_field(payload, "delayMix"),
        paulstretch_enabled=_optional_bool_field(payload, "paulstretchEnabled"),
        stretch=_optional_float_field(payload, "stretch"),
        window_size=_optional_float_field(payload, "windowSize"),
    )


def _sweep_config_from_payload(payload: dict):
    return parse_sweep_config(
        voice=_optional_trimmed(payload, "voice"),
        pitch_min=_optional_float_field(payload, "pitchMin"),
        pitch_max=_optional_float_field(payload, "pitchMax"),
        curve_shape=_optional_trimmed(payload, "curveShape"),
        curve_peak_position=_optional_float_field(payload, "curvePeakPosition")
        if payload.get("curvePeakPosition") is not None
        else _optional_float_field(payload, "curvePeak"),
        filter_enabled=_optional_bool_field(payload, "filterEnabled"),
        filter_type=_optional_trimmed(payload, "filterType"),
        filter_cutoff_low=_optional_int_field(payload, "filterCutoffLow"),
        filter_cutoff_high=_optional_int_field(payload, "filterCutoffHigh"),
        tremolo_enabled=_optional_bool_field(payload, "tremoloEnabled"),
        tremolo_rate_min=_optional_float_field(payload, "tremoloRateMin"),
        tremolo_rate_max=_optional_float_field(payload, "tremoloRateMax"),
        tremolo_depth=_optional_float_field(payload, "tremoloDepth"),
        phaser_enabled=_optional_bool_field(payload, "phaserEnabled"),
        phaser_rate_min=_optional_float_field(payload, "phaserRateMin"),
        phaser_rate_max=_optional_float_field(payload, "phaserRateMax"),
        phaser_depth=_optional_float_field(payload, "phaserDepth"),
        phaser_centre=_optional_float_field(payload, "phaserCentre"),
        phaser_feedback=_optional_float_field(payload, "phaserFeedback"),
        phaser_mix=_optional_float_field(payload, "phaserMix"),
        chorus_enabled=_optional_bool_field(payload, "chorusEnabled"),
        chorus_rate_min=_optional_float_field(payload, "chorusRateMin"),
        chorus_rate_max=_optional_float_field(payload, "chorusRateMax"),
        chorus_depth=_optional_float_field(payload, "chorusDepth"),
        chorus_delay=_optional_float_field(payload, "chorusDelay"),
        chorus_mix=_optional_float_field(payload, "chorusMix"),
        flanger_enabled=_optional_bool_field(payload, "flangerEnabled"),
        flanger_rate_min=_optional_float_field(payload, "flangerRateMin"),
        flanger_rate_max=_optional_float_field(payload, "flangerRateMax"),
        flanger_depth=_optional_float_field(payload, "flangerDepth"),
        flanger_delay=_optional_float_field(payload, "flangerDelay"),
        flanger_feedback=_optional_float_field(payload, "flangerFeedback"),
        flanger_mix=_optional_float_field(payload, "flangerMix"),
        gain_enabled=_optional_bool_field(payload, "gainEnabled"),
        gain_min=_optional_float_field(payload, "gainMin"),
        gain_max=_optional_float_field(payload, "gainMax"),
    )


def _run_generate_sweep(payload: dict) -> list[str]:
    """Generate sweep samples from Auditionr JSON body."""
    post_spec = _normalize_post_processing_spec_from_payload(payload)
    tempo = max(1, _positive_int_field(payload, "tempo", 120))
    iterations = max(1, _positive_int_field(payload, "iterations", 1))
    bars = max(1, _positive_int_field(payload, "bars", 8))
    paths: list[str] = []

    for _ in range(iterations):
        config = _sweep_config_from_payload(payload)
        beat_name = generate_beat_name()
        name_parts = [
            "sweep",
            beat_name,
            f"{tempo}bpm",
            f"{bars}bars",
            generate_id(),
        ]
        sample_name = format_name("___".join(name_parts))
        output_path = f"{managed_paths.EXPORTS_DIR}/{sample_name}.wav"
        output_path, _ = generate_sweep_sample(
            tempo=tempo, bars=bars, output=output_path, config=config
        )
        paths.append(output_path)
        print(with_prompt(f"generated: {output_path}"))

    _apply_drone_post_processing_to_wavs(paths, post_spec)
    return paths


def _run_generate_wash(payload: dict) -> list[str]:
    """Generate wash samples from Auditionr JSON body."""
    post_spec = _normalize_post_processing_spec_from_payload(payload)
    tempo = max(1, _positive_int_field(payload, "tempo", 120))
    iterations = max(1, _positive_int_field(payload, "iterations", 1))
    bars = max(1, _positive_int_field(payload, "bars", 8))
    config = _wash_config_from_payload(payload)
    library = config.get("library")
    original_preset_name = None
    paths: list[str] = []

    if library:
        original_preset_name = get_active_drum_path_preset_name()
        ok, result = set_active_drum_path_preset(library)
        if not ok:
            raise ValueError(result)
        print(with_prompt(f"Using drum path preset: {result}"))
    try:
        for _ in range(iterations):
            beat_name = generate_beat_name()
            perc_label = config.get("percussion") or "random"
            name_parts = [
                "wash",
                str(perc_label),
                beat_name,
                f"{tempo}bpm",
                f"{bars}bars",
                generate_id(),
            ]
            sample_name = format_name("___".join(name_parts))
            output_path = f"{managed_paths.EXPORTS_DIR}/{sample_name}.wav"
            output_path, _ = generate_wash_sample(
                tempo=tempo,
                bars=bars,
                output=output_path,
                config=config,
            )
            paths.append(output_path)
            print(with_prompt(f"generated: {output_path}"))
    finally:
        if original_preset_name is not None:
            set_active_drum_path_preset(original_preset_name)

    _apply_drone_post_processing_to_wavs(paths, post_spec)
    return paths


def _run_generate_transition(subcommand: str, payload: dict) -> list[str]:
    """Backward-compatible wrapper for legacy transition + subcommand API."""
    if subcommand == "wash":
        return _run_generate_wash(payload)
    if subcommand == "sweep":
        return _run_generate_sweep(payload)
    raise ValueError(f"Unknown transition subcommand: {subcommand}")


def _load_beat_patterns_for_generate() -> dict:
    path = get_managed_file("config", "beat-patterns.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _load_drum_kits_for_generate() -> dict:
    path = get_managed_file("config", "drum-kits.json")
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
                output_path = f"{managed_paths.EXPORTS_DIR}/{sample_name}.wav"

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
                apply_post_processing_actions(
                    output_path, post_actions, normalize_peak_db=BEAT_EXPORT_PEAK_DB
                )
                paths.append(output_path)
    finally:
        if original_preset_name:
            set_active_drum_path_preset(original_preset_name)

    return paths


def _handle_drone_patch_detail():
    name = (request.args.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400
    try:
        from dronmakr.apps.generatr_plugins import get_drone_patch_detail

        return jsonify(get_drone_patch_detail(name))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _handle_drone_plugin_scan_status():
    try:
        from dronmakr.apps.generatr_plugins import get_drone_plugin_scan_status

        return jsonify(get_drone_plugin_scan_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _handle_drone_plugin_scan_start():
    data = request.get_json(silent=True) or {}
    force = bool(data.get("force"))
    try:
        from dronmakr.apps.generatr_plugins import start_drone_plugin_scan

        return jsonify(start_drone_plugin_scan(force=force))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _handle_drone_plugin_list_editor():
    if request.method == "GET":
        role = (request.args.get("role") or "instrument").strip().lower()
        if role not in ("instrument", "effect"):
            return jsonify({"error": "role must be instrument or effect"}), 400
        try:
            from dronmakr.apps.generatr_plugins import get_drone_plugin_list_editor_payload

            return jsonify(get_drone_plugin_list_editor_payload(role))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    data = request.get_json() or {}
    role = (data.get("role") or "instrument").strip().lower()
    if role not in ("instrument", "effect"):
        return jsonify({"error": "role must be instrument or effect"}), 400
    allowed = data.get("allowedLabels") or data.get("allowed") or []
    if not isinstance(allowed, list):
        return jsonify({"error": "allowedLabels must be a list"}), 400
    try:
        from dronmakr.apps.generatr_plugins import save_drone_plugin_list_editor

        return jsonify(save_drone_plugin_list_editor(role, allowed))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _handle_drone_plugin_picker():
    role = (request.args.get("role") or "instrument").strip().lower()
    if role not in ("instrument", "effect"):
        return jsonify({"error": "role must be instrument or effect"}), 400
    try:
        from dronmakr.apps.generatr_plugins import get_drone_picker_payload

        return jsonify(get_drone_picker_payload(role))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _handle_drone_plugin_editor():
    data = request.get_json() or {}
    plugin_path = (data.get("pluginPath") or data.get("plugin_path") or "").strip()
    preset_path = (data.get("presetPath") or data.get("preset_path") or "").strip() or None
    role = (data.get("role") or "instrument").strip().lower()
    if not plugin_path:
        return jsonify({"error": "pluginPath is required"}), 400
    if role not in ("instrument", "effect"):
        return jsonify({"error": "role must be instrument or effect"}), 400
    fx_slot_index = data.get("fxSlotIndex")
    if fx_slot_index is not None and fx_slot_index != "":
        try:
            fx_slot_index = int(fx_slot_index)
        except (TypeError, ValueError):
            return jsonify({"error": "fxSlotIndex must be an integer"}), 400
    else:
        fx_slot_index = None
    editor_preview = None
    try:
        from dronmakr.audio.audio_worker import delegate_open_drone_plugin_editor_if_needed
        from dronmakr.apps.generatr_plugins import open_drone_plugin_editor_capture

        preview_context = data.get("previewContext")
        if isinstance(preview_context, dict):
            editor_preview = _prepare_drone_editor_preview(
                preview_context,
                edit_plugin_path=plugin_path,
                edit_role=role,
                edit_preset_path=preset_path,
                fx_slot_index=fx_slot_index,
            )

        result = delegate_open_drone_plugin_editor_if_needed(
            plugin_path,
            role,
            preset_path,
            editor_preview=editor_preview,
        )
        if result is None:
            result = open_drone_plugin_editor_capture(
                plugin_path,
                role,
                preset_path,
                editor_preview=editor_preview,
            )
            editor_preview = None
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if editor_preview and editor_preview.get("midi_path"):
            try:
                os.remove(editor_preview["midi_path"])
            except OSError:
                pass


def _handle_drone_midi_patterns():
    include_previews = (request.args.get("previews") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    return jsonify({"patterns": get_patterns_catalog(include_previews=include_previews)})


def _read_drone_midi_import_bytes() -> bytes:
    upload = request.files.get("file") if request.files else None
    if upload and upload.filename:
        data = upload.read()
        if data:
            return data
    raise ValueError("Missing MIDI file upload")


def _handle_drone_midi_import():
    try:
        data = _read_drone_midi_import_bytes()
        notes = extract_drone_piano_notes_from_midi_bytes(data)
        return jsonify({"notes": notes, "count": len(notes)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _handle_drone_midi_preview():
    data = request.get_json() or {}
    try:
        midi_kwargs, _chart_pool_last_key = _build_drone_midi_kwargs_from_payload(data, preview=True)
        midi_obj, chart_label, render_duration_sec, pattern_id = generate_drone_midi(
            **midi_kwargs,
            quiet=True,
        )
        description = next(
            (
                item["description"]
                for item in get_patterns_catalog()
                if item["id"] == pattern_id
            ),
            "",
        )
        return jsonify(
            {
                "preview": build_midi_preview_payload(midi_obj),
                "pattern": pattern_id,
                "patternDisplayName": format_pattern_display_name(pattern_id),
                "description": description,
                "chart": chart_label,
                "renderDurationSec": render_duration_sec,
                "tempoBpm": midi_kwargs.get("tempo_bpm"),
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _handle_drone_audio_preview():
    data = request.get_json() or {}
    wav_path = ""
    try:
        wav_path, render_duration_sec, chart_label = _run_drone_audio_preview(data)
        response = send_file(
            wav_path,
            mimetype="audio/wav",
            as_attachment=False,
            download_name="drone-preview.wav",
        )
        response.headers["X-Drone-Preview-Duration"] = str(render_duration_sec)
        response.headers["X-Drone-Preview-Chart"] = chart_label or ""
        return response
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if wav_path:
            try:
                os.remove(wav_path)
            except OSError:
                pass


def _handle_drone_save_preset():
    data = request.get_json() or {}
    role = (data.get("role") or "instrument").strip().lower()
    name = (data.get("name") or "").strip()
    if role not in ("instrument", "effect"):
        return jsonify({"error": "role must be instrument or effect"}), 400
    if not name:
        return jsonify({"error": "name is required"}), 400
    try:
        from dronmakr.apps.generatr_plugins import save_drone_preset

        result = save_drone_preset(
            role=role,
            name=name,
            instrument_selection=data.get("instrumentSelection"),
            fx_slots=data.get("fxSlots"),
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _handle_drone_delete_preset():
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    preset_id = (data.get("id") or data.get("presetId") or "").strip()
    if not name and not preset_id:
        return jsonify({"error": "name or id is required"}), 400
    try:
        from dronmakr.apps.generatr_plugins import delete_drone_preset

        result = delete_drone_preset(name=name or None, preset_id=preset_id or None)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _handle_drone_chord_scale_catalog():
    return jsonify({"charts": get_chord_scale_catalog()})


def _handle_api_generate_options():
    ensure_settings()
    preset_index = get_presets()
    pick = get_chord_scale_picklists()
    return jsonify(
        {
            "patterns": sorted(_load_beat_patterns_for_generate().keys()),
            "kits": sorted(_load_drum_kits_for_generate().keys()),
            "processingActions": get_processing_actions_payload(),
            "droneMidiPatternCatalog": get_patterns_catalog(),
            "droneInstruments": preset_index.get("instruments", []),
            "droneEffects": preset_index.get("effects", []),
            "droneChordScaleRoots": pick["roots"],
            "droneChordScaleTags": pick["tags"],
            "droneChordScaleChartNames": pick["chartNames"],
            "drumPathsConfigured": has_configured_drum_paths(),
            "pluginPathsConfigured": has_configured_plugin_paths(),
        }
    )


def _normalize_generate_type(gen_type: str, subcommand: str) -> str:
    """Map API type (+ optional legacy subcommand) to a supported generator type."""
    g = (gen_type or "").strip().lower()
    sub = (subcommand or "").strip().lower()
    if g == "beat":
        return "drumpattern"
    if g == "transition":
        if sub == "wash":
            return "wash"
        if sub == "sweep":
            return "sweep"
        raise ValueError("Transition requires subcommand: sweep or wash")
    return g


def _handle_api_generate():
    """
    POST /api/generatr/generate
    Body: { "type": "drone" | "bass" | "sweep" | "wash" | "drumpattern", "subcommand": optional }
    Returns: { "paths": [...], "error": null } or { "paths": [], "error": "..." }
    """
    ensure_settings()
    data = request.get_json() or {}
    gen_type_raw = (data.get("type") or "").strip().lower()
    subcommand = (data.get("subcommand") or "").strip().lower()

    if not gen_type_raw:
        return jsonify(
            {
                "paths": [],
                "error": "Missing type (drone, bass, sweep, wash, or drumpattern)",
            }
        ), 400

    try:
        gen_type = _normalize_generate_type(gen_type_raw, subcommand)
    except ValueError as e:
        return jsonify({"paths": [], "error": str(e)}), 400

    supported = ("drone", "bass", "sweep", "wash", "drumpattern")
    if gen_type not in supported:
        return jsonify({"paths": [], "error": f"Unknown type: {gen_type_raw}"}), 400
    if gen_type == "bass" and subcommand not in ("reese", "donk"):
        return jsonify({"paths": [], "error": "Bass requires subcommand: reese or donk"}), 400

    from dronmakr.core.utils import refresh_managed_path_constants

    refresh_managed_path_constants()
    if not managed_paths.EXPORTS_DIR:
        return jsonify({"paths": [], "error": "Storage folder is not configured. Complete onboarding first."}), 400

    try:
        print(f"{RED}│{RESET} generate: {gen_type}" + (f" {subcommand}" if subcommand and gen_type == "bass" else ""))
        if gen_type == "drone":
            paths = _run_generate_drone(data)
        elif gen_type == "bass":
            paths = _run_generate_bass(subcommand, data)
        elif gen_type == "sweep":
            paths = _run_generate_sweep(data)
        elif gen_type == "wash":
            paths = _run_generate_wash(data)
        else:
            paths = _run_generate_beat(data)
        print(f"{RED}■ generate completed{RESET}")
        _socket_broadcast("exports", {"files": get_latest_exports()})
        _emit_folder_counts()
        return jsonify({"paths": paths, "error": None})
    except Exception as e:
        print(with_prompt(f"generate error: {e}"))
        return jsonify({"paths": [], "error": str(e)}), 500


def _handle_request_exports():
    """Socket handler: client requests latest exports and folder counts (e.g. after generate)."""
    _socket_broadcast("exports", {"files": get_latest_exports()})
    _emit_folder_counts()


def _handle_api_post_processing_shortcuts():
    """GET raw shortcuts JSON; POST append; DELETE ?name=…"""
    if request.method == "GET":
        try:
            return jsonify(read_post_processing_shortcuts_document())
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    if request.method == "POST":
        data = request.get_json() or {}
        name = str(data.get("name") or "").strip()
        command = str(data.get("command") or "").strip()
        try:
            append_post_processing_shortcut(name, command)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        _emit_configs_to_clients()
        return jsonify({"success": True}), 200

    if request.method == "DELETE":
        name = request.args.get("name") or ""
        try:
            removed = remove_post_processing_shortcut(name)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        if not removed:
            return jsonify({"error": "Shortcut not found"}), 404
        _emit_configs_to_clients()
        return jsonify({"success": True}), 200

    return jsonify({"error": "Method not allowed"}), 405


def register_auditionr(app, socketio):
    """Register auditionr routes on the given Flask app. Socketio is used for emits."""
    global _socketio
    warm_chord_scale_picklists()
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
        "/prepare-drag-copy",
        "auditionr_prepare_drag_copy",
        prepare_drag_copy,
        methods=["POST"],
    )
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
        "/api/generatr/drone-plugin-scan",
        "auditionr_api_drone_plugin_scan",
        _handle_drone_plugin_scan_start,
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/generatr/drone-plugin-scan-status",
        "auditionr_api_drone_plugin_scan_status",
        _handle_drone_plugin_scan_status,
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/generatr/drone-plugin-list",
        "auditionr_api_drone_plugin_list",
        _handle_drone_plugin_list_editor,
        methods=["GET", "POST"],
    )
    app.add_url_rule(
        "/api/generatr/drone-patch",
        "auditionr_api_drone_patch_detail",
        _handle_drone_patch_detail,
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/generatr/drone-plugin-picker",
        "auditionr_api_drone_plugin_picker",
        _handle_drone_plugin_picker,
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/generatr/drone-plugin-editor",
        "auditionr_api_drone_plugin_editor",
        _handle_drone_plugin_editor,
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/generatr/drone-chord-scale-catalog",
        "auditionr_api_drone_chord_scale_catalog",
        _handle_drone_chord_scale_catalog,
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/generatr/drone-midi-patterns",
        "auditionr_api_drone_midi_patterns",
        _handle_drone_midi_patterns,
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/generatr/drone-midi-import",
        "auditionr_api_drone_midi_import",
        _handle_drone_midi_import,
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/generatr/drone-midi-preview",
        "auditionr_api_drone_midi_preview",
        _handle_drone_midi_preview,
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/generatr/drone-audio-preview",
        "auditionr_api_drone_audio_preview",
        _handle_drone_audio_preview,
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/generatr/drone-save-preset",
        "auditionr_api_drone_save_preset",
        _handle_drone_save_preset,
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/generatr/drone-delete-preset",
        "auditionr_api_drone_delete_preset",
        _handle_drone_delete_preset,
        methods=["POST"],
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
    app.add_url_rule(
        "/api/post-processing-shortcuts",
        "auditionr_api_post_processing_shortcuts",
        _handle_api_post_processing_shortcuts,
        methods=["GET", "POST", "DELETE"],
    )


if __name__ == "__main__":
    from backend_server import main

    main()
