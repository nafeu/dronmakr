"""
folysplitr routes and helpers.
"""

from __future__ import annotations

import mimetypes
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from flask import jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

ROOT_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT_DIR / "recordings"
MAX_RECORDING_BYTES = 300 * 1024 * 1024

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
        files = []
        for file in sorted(
            RECORDINGS_DIR.glob("*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            if not file.is_file():
                continue
            files.append(
                {
                    "filename": file.name,
                    "url": f"/recordings/{file.name}",
                    "size": file.stat().st_size,
                }
            )
        return jsonify({"ok": True, "files": files})

    @app.route("/recordings/<path:filename>")
    def folysplitr_recording_file(filename):
        ensure_recordings_dir()
        return send_from_directory(RECORDINGS_DIR, filename)
