"""
folysplitr routes and helpers.
"""

from __future__ import annotations

import mimetypes
import os
import shutil
import subprocess
import tempfile
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from flask import jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
from process_sample import trim_sample_start, trim_sample_end, reverse_sample
from processing_actions import apply_processing_command, get_processing_actions_payload

ROOT_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT_DIR / "recordings"
SPLITS_DIR = ROOT_DIR / "splits"
FOLYSPLITR_UNDO_DIR = ROOT_DIR / "temp" / "folysplitr_undo"
MAX_RECORDING_BYTES = 300 * 1024 * 1024
SPLIT_CATEGORIES = [
    "kick",
    "hihat",
    "perc",
    "tom",
    "snare",
    "shaker",
    "clap",
    "cymbal",
    "fx",
    "synth",
    "instrument",
    "misc",
    "trash",
    "archive",
]

_ALLOWED_EXTENSIONS = {
    ".wav",
    ".webm",
    ".ogg",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
}

_MIME_TO_EXTENSION = {
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/x-wav": ".wav",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
    "audio/flac": ".flac",
}


def ensure_recordings_dir() -> Path:
    """Ensure recordings directory exists."""
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return RECORDINGS_DIR


def ensure_splits_dirs() -> Path:
    """Ensure splits root and category directories exist."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for category in SPLIT_CATEGORIES:
        (SPLITS_DIR / category).mkdir(parents=True, exist_ok=True)
    return SPLITS_DIR


def _resolve_audio_path(file_ref: str) -> Path | None:
    cleaned = (file_ref or "").strip().lstrip("/")
    if not cleaned:
        return None
    if cleaned.startswith("recordings/"):
        candidate = ROOT_DIR / cleaned
    elif cleaned.startswith("splits/"):
        candidate = ROOT_DIR / cleaned
    else:
        candidate = RECORDINGS_DIR / cleaned
    try:
        resolved = candidate.resolve(strict=False)
    except OSError:
        return None
    if not str(resolved).startswith(str(ROOT_DIR.resolve())):
        return None
    return resolved


def _public_audio_payload(file_path: Path) -> dict:
    rel = file_path.relative_to(ROOT_DIR).as_posix()
    if rel.startswith("recordings/"):
        url = "/" + rel
    elif rel.startswith("splits/"):
        url = "/" + rel
    else:
        url = f"/recordings/{file_path.name}"
    return {
        "path": rel,
        "filename": file_path.name,
        "url": url,
        "size": file_path.stat().st_size,
        "folder": file_path.parent.name,
    }


def _safe_split_bounds(start_s: float, end_s: float, duration_s: float) -> tuple[float, float]:
    start = max(0.0, min(float(start_s), duration_s))
    end = max(0.0, min(float(end_s), duration_s))
    if end < start:
        start, end = end, start
    return start, end


def _sanitize_markers(markers: list[float], duration_s: float) -> list[float]:
    out = [0.0, float(duration_s)]
    for marker in markers:
        try:
            t = float(marker)
        except (TypeError, ValueError):
            continue
        t = max(0.0, min(float(duration_s), t))
        if 0.0 < t < duration_s:
            out.append(t)
    out.sort()
    dedup: list[float] = []
    for t in out:
        if not dedup or abs(t - dedup[-1]) > 0.005:
            dedup.append(t)
    return dedup


def _ensure_folysplitr_undo_dir() -> Path:
    FOLYSPLITR_UNDO_DIR.mkdir(parents=True, exist_ok=True)
    return FOLYSPLITR_UNDO_DIR


def _undo_snapshot_path_for(file_path: Path) -> Path:
    _ensure_folysplitr_undo_dir()
    key = hashlib.sha1(str(file_path.resolve()).encode("utf-8")).hexdigest()
    return FOLYSPLITR_UNDO_DIR / f"{key}.wav"


def _save_undo_snapshot(file_path: Path) -> None:
    if not file_path.exists() or not file_path.is_file():
        return
    snapshot = _undo_snapshot_path_for(file_path)
    shutil.copy2(file_path, snapshot)


def _guess_extension(filename: str, content_type: str) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext in _ALLOWED_EXTENSIONS:
        return ext
    if content_type in _MIME_TO_EXTENSION:
        return _MIME_TO_EXTENSION[content_type]
    guessed = mimetypes.guess_extension(content_type or "")
    if guessed and guessed.lower() in _ALLOWED_EXTENSIONS:
        return guessed.lower()
    return ".webm"


def _convert_to_wav(source_path: Path, target_path: Path) -> tuple[bool, str]:
    """Convert any supported input audio to WAV using ffmpeg."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return False, "ffmpeg is required for WAV conversion but was not found."
    try:
        result = subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-i",
                str(source_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "48000",
                "-ac",
                "2",
                str(target_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return False, f"Failed to run ffmpeg: {exc}"

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if detail:
            detail = " " + detail.splitlines()[-1]
        return False, f"Audio conversion failed.{detail}"
    return True, ""


def register_folysplitr(app):
    """Register folysplitr API routes."""

    @app.route("/api/folysplitr/recordings", methods=["POST"])
    def api_folysplitr_recording_upload():
        ensure_recordings_dir()
        ensure_splits_dirs()
        incoming = request.files.get("file")
        if incoming is None:
            return jsonify({"ok": False, "error": "Missing file"}), 400

        incoming.seek(0, 2)
        size = incoming.tell()
        incoming.seek(0)
        if size <= 0:
            return jsonify({"ok": False, "error": "Recording is empty"}), 400
        if size > MAX_RECORDING_BYTES:
            return jsonify({"ok": False, "error": "Recording exceeds size limit"}), 400

        original_name = secure_filename(incoming.filename or "").strip()
        content_type = (incoming.content_type or "").strip().lower()
        extension = _guess_extension(original_name, content_type)

        stem = Path(original_name).stem if original_name else "recording"
        stem = secure_filename(stem).strip() or "recording"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_name = f"{timestamp}_{stem}.wav"
        save_path = RECORDINGS_DIR / saved_name

        with tempfile.TemporaryDirectory(prefix="folysplitr_") as temp_dir:
            temp_input = Path(temp_dir) / f"incoming{extension}"
            incoming.save(temp_input)
            ok, error_message = _convert_to_wav(temp_input, save_path)
            if not ok:
                return jsonify({"ok": False, "error": error_message}), 500

        return jsonify(
            {
                "ok": True,
                "filename": saved_name,
                "url": f"/recordings/{saved_name}",
                "size": save_path.stat().st_size,
            }
        )

    @app.route("/api/folysplitr/recordings", methods=["GET"])
    def api_folysplitr_recordings_list():
        ensure_recordings_dir()
        ensure_splits_dirs()
        files = []
        audio_files: list[Path] = []
        # Splitr queue is recordings-only. Files moved to splits/* leave the queue.
        for file in RECORDINGS_DIR.glob("*.wav"):
            if file.is_file():
                audio_files.append(file)
        for file in sorted(audio_files, key=lambda p: p.stat().st_mtime, reverse=True):
            files.append(_public_audio_payload(file))
        return jsonify({"ok": True, "files": files})

    @app.route("/recordings/<path:filename>")
    def folysplitr_recording_file(filename):
        ensure_recordings_dir()
        return send_from_directory(RECORDINGS_DIR, filename)

    @app.route("/splits/<path:filename>")
    def folysplitr_split_file(filename):
        ensure_splits_dirs()
        return send_from_directory(SPLITS_DIR, filename)

    @app.route("/api/folysplitr/processing-actions", methods=["GET"])
    def api_folysplitr_processing_actions():
        payload = get_processing_actions_payload()
        payload["splitCategories"] = list(SPLIT_CATEGORIES)
        return jsonify({"ok": True, **payload})

    @app.route("/api/folysplitr/duplicate", methods=["POST"])
    def api_folysplitr_duplicate():
        ensure_recordings_dir()
        ensure_splits_dirs()
        params = request.get_json(silent=True) or {}
        source = _resolve_audio_path(params.get("path", ""))
        if source is None or not source.exists() or not source.is_file():
            return jsonify({"ok": False, "error": "File does not exist"}), 404
        stem, ext = source.stem, source.suffix
        base_name = f"{stem}-copy{ext}"
        target = source.parent / base_name
        counter = 2
        while target.exists():
            target = source.parent / f"{stem}-copy{counter}{ext}"
            counter += 1
        shutil.copy2(source, target)
        return jsonify({"ok": True, "file": _public_audio_payload(target)})

    @app.route("/api/folysplitr/process", methods=["POST"])
    def api_folysplitr_process():
        ensure_recordings_dir()
        ensure_splits_dirs()
        params = request.get_json(silent=True) or {}
        source = _resolve_audio_path(params.get("path", ""))
        command = (params.get("command") or "").strip()
        if source is None or not source.exists() or not source.is_file():
            return jsonify({"ok": False, "error": "File does not exist"}), 404
        if not command:
            return jsonify({"ok": False, "error": "Missing command"}), 400
        try:
            _save_undo_snapshot(source)
            if command == "trim_sample_start":
                trim_sample_start(str(source), float(params.get("seconds", 0)))
            elif command == "trim_sample_end":
                trim_sample_end(str(source), float(params.get("seconds", 0)))
            elif command == "reverse_sample":
                reverse_sample(str(source))
            else:
                apply_processing_command(str(source), command, params)
        except (ValueError, TypeError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": f"Failed to process sample: {exc}"}), 500
        return jsonify({"ok": True, "file": _public_audio_payload(source)})

    @app.route("/api/folysplitr/undo", methods=["POST"])
    def api_folysplitr_undo():
        params = request.get_json(silent=True) or {}
        source = _resolve_audio_path(params.get("path", ""))
        if source is None or not source.exists() or not source.is_file():
            return jsonify({"ok": False, "error": "File does not exist"}), 404
        snapshot = _undo_snapshot_path_for(source)
        if not snapshot.exists():
            return jsonify({"ok": False, "error": "No audio undo available"}), 400
        try:
            shutil.copy2(snapshot, source)
            snapshot.unlink(missing_ok=True)
        except OSError as exc:
            return jsonify({"ok": False, "error": f"Undo failed: {exc}"}), 500
        return jsonify({"ok": True, "file": _public_audio_payload(source)})

    @app.route("/api/folysplitr/organize", methods=["POST"])
    def api_folysplitr_organize():
        ensure_recordings_dir()
        ensure_splits_dirs()
        params = request.get_json(silent=True) or {}
        source = _resolve_audio_path(params.get("path", ""))
        destination = (params.get("destination") or "").strip().lower()
        if source is None or not source.exists() or not source.is_file():
            return jsonify({"ok": False, "error": "File does not exist"}), 404
        if destination not in SPLIT_CATEGORIES:
            return jsonify({"ok": False, "error": "Invalid split destination"}), 400
        target_dir = SPLITS_DIR / destination
        target_dir.mkdir(parents=True, exist_ok=True)
        target_name = secure_filename(source.name) or source.name
        target = target_dir / target_name
        counter = 2
        while target.exists():
            stem, ext = os.path.splitext(target_name)
            target = target_dir / f"{stem}-{counter}{ext}"
            counter += 1
        shutil.move(str(source), str(target))
        return jsonify({"ok": True, "file": _public_audio_payload(target)})

    @app.route("/api/folysplitr/split", methods=["POST"])
    def api_folysplitr_split():
        ensure_recordings_dir()
        ensure_splits_dirs()
        params = request.get_json(silent=True) or {}
        source = _resolve_audio_path(params.get("path", ""))
        if source is None or not source.exists() or not source.is_file():
            return jsonify({"ok": False, "error": "File does not exist"}), 404

        try:
            audio, sample_rate = sf.read(str(source))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": f"Could not read audio: {exc}"}), 500

        if audio.size == 0:
            return jsonify({"ok": False, "error": "Audio file is empty"}), 400
        total_samples = audio.shape[0]
        duration = total_samples / float(sample_rate)
        markers = params.get("markers", [])
        if not isinstance(markers, list):
            return jsonify({"ok": False, "error": "markers must be a list"}), 400
        bounds = _sanitize_markers(markers, duration)
        segments = []
        for i in range(len(bounds) - 1):
            start, end = _safe_split_bounds(bounds[i], bounds[i + 1], duration)
            if (end - start) >= 0.01:
                segments.append((start, end))
        segments.sort(key=lambda pair: pair[0])
        if not segments:
            return jsonify({"ok": False, "error": "No valid split segments"}), 400

        stem, ext = source.stem, source.suffix
        created_files: list[Path] = []
        for idx, (start, end) in enumerate(segments, start=1):
            start_i = int(round(start * sample_rate))
            end_i = int(round(end * sample_rate))
            if end_i <= start_i:
                continue
            segment = audio[start_i:end_i]
            target = source.parent / f"{stem}-split-{idx:02d}{ext}"
            counter = 2
            while target.exists():
                target = source.parent / f"{stem}-split-{idx:02d}-{counter}{ext}"
                counter += 1
            sf.write(str(target), segment, sample_rate)
            created_files.append(target)

        if not created_files:
            return jsonify({"ok": False, "error": "No split files were created"}), 400
        original_path = source.relative_to(ROOT_DIR).as_posix()
        try:
            source.unlink()
        except OSError as exc:
            return jsonify({"ok": False, "error": f"Failed to remove original file: {exc}"}), 500

        files_payload = [_public_audio_payload(path) for path in created_files]
        return jsonify(
            {
                "ok": True,
                "files": files_payload,
                "activeFile": files_payload[0],
                "originalPath": original_path,
            }
        )

    @app.route("/api/folysplitr/markers/autodetect", methods=["POST"])
    def api_folysplitr_markers_autodetect():
        params = request.get_json(silent=True) or {}
        source = _resolve_audio_path(params.get("path", ""))
        if source is None or not source.exists() or not source.is_file():
            return jsonify({"ok": False, "error": "File does not exist"}), 404
        mode = str(params.get("mode", "transient")).strip().lower()
        try:
            threshold = float(params.get("threshold", 0.35))
        except (TypeError, ValueError):
            threshold = 0.35
        threshold = max(0.01, min(1.0, threshold))
        try:
            min_gap_ms = float(params.get("min_gap_ms", 60))
        except (TypeError, ValueError):
            min_gap_ms = 60.0
        min_gap_ms = max(10.0, min(500.0, min_gap_ms))
        try:
            beat_division = int(params.get("beat_division", 4))
        except (TypeError, ValueError):
            beat_division = 4
        if beat_division not in (1, 2, 4, 8, 16):
            beat_division = 4
        try:
            tempo_bpm = float(params.get("tempo_bpm", 120))
        except (TypeError, ValueError):
            tempo_bpm = 120.0
        tempo_bpm = max(40.0, min(260.0, tempo_bpm))

        try:
            audio, sample_rate = sf.read(str(source), always_2d=True)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": f"Could not read audio: {exc}"}), 500
        if audio.size == 0:
            return jsonify({"ok": False, "markers": []})

        mono = np.mean(np.abs(audio), axis=1).astype(np.float32)
        peak = float(np.max(mono)) if mono.size else 0.0
        if peak <= 1e-12:
            return jsonify({"ok": True, "markers": []})
        norm = mono / peak
        refractory = max(1, int(sample_rate * (min_gap_ms / 1000.0)))
        marker_indexes: list[int] = []
        duration = audio.shape[0] / float(sample_rate)

        if mode == "level":
            smooth_window = max(1, int(sample_rate * 0.02))
            if smooth_window > 1:
                kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
                norm = np.convolve(norm, kernel, mode="same")
            i = 0
            n = norm.shape[0]
            while i < n:
                if norm[i] >= threshold:
                    marker_indexes.append(i)
                    i += refractory
                    continue
                i += 1
        elif mode == "transient":
            flux = np.diff(norm, prepend=norm[0])
            flux = np.maximum(flux, 0.0)
            flux_peak = float(np.max(flux)) if flux.size else 0.0
            if flux_peak > 1e-12:
                flux = flux / flux_peak
            i = 1
            n = flux.shape[0]
            while i < n:
                if flux[i] >= threshold:
                    marker_indexes.append(i)
                    i += refractory
                    continue
                i += 1
        elif mode == "beat":
            envelope = np.abs(np.diff(norm, prepend=norm[0]))
            env_peak = float(np.max(envelope)) if envelope.size else 0.0
            if env_peak > 1e-12:
                envelope = envelope / env_peak
            energy = float(np.mean(envelope)) if envelope.size else 0.0
            if energy <= 1e-6:
                marker_indexes = []
            else:
                beats_per_second = tempo_bpm / 60.0
                seconds_per_beat = 1.0 / max(1e-6, beats_per_second)
                step_seconds = seconds_per_beat / max(1, beat_division)
                step = max(1, int(round(sample_rate * step_seconds)))
                marker_indexes = list(range(0, len(norm), step))
        else:
            return jsonify({"ok": False, "error": f"Unknown autodetect mode: {mode}"}), 400

        duration = audio.shape[0] / float(sample_rate)
        markers = [0.0]
        for idx in marker_indexes:
            t = idx / float(sample_rate)
            if t > 0.01 and (not markers or abs(t - markers[-1]) > 0.02):
                markers.append(t)
        if duration > 0:
            markers.append(duration)

        cleaned = []
        for t in sorted(markers):
            if not cleaned or abs(t - cleaned[-1]) > 0.02:
                cleaned.append(float(t))
        return jsonify({"ok": True, "markers": cleaned})

    @app.route("/api/folysplitr/trash/empty", methods=["POST"])
    def api_folysplitr_empty_trash():
        ensure_splits_dirs()
        trash_dir = SPLITS_DIR / "trash"
        removed = 0
        for wav in trash_dir.glob("*.wav"):
            try:
                wav.unlink()
                removed += 1
            except OSError:
                continue
        return jsonify({"ok": True, "removed": removed})

    @app.route("/api/folysplitr/archive/unarchive", methods=["POST"])
    def api_folysplitr_unarchive():
        ensure_recordings_dir()
        ensure_splits_dirs()
        archive_dir = SPLITS_DIR / "archive"
        moved = 0
        for wav in sorted(archive_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime):
            target = RECORDINGS_DIR / wav.name
            counter = 2
            while target.exists():
                target = RECORDINGS_DIR / f"{wav.stem}-{counter}{wav.suffix}"
                counter += 1
            try:
                shutil.move(str(wav), str(target))
                moved += 1
            except OSError:
                continue
        return jsonify({"ok": True, "moved": moved})
