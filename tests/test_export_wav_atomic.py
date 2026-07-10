"""Export WAV writes must not rely on /tmp when destination is on another mount."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dronmakr.generate.generate_sample import _write_validated_export_wav


def test_write_validated_export_wav_writes_in_destination_directory(tmp_path: Path):
    exports_dir = tmp_path / "exports"
    exports_dir.mkdir(parents=True)
    output_path = exports_dir / "drone_test.wav"
    audio = np.zeros((2000, 2), dtype=np.float32)
    audio[100:400, :] = 0.35

    _write_validated_export_wav(str(output_path), audio, 44100)

    assert output_path.is_file()
    leftovers = list(exports_dir.glob("dronmakr_export_*"))
    assert leftovers == []


def test_write_validated_export_wav_uses_destination_dir_for_tempfile(tmp_path: Path):
    exports_dir = tmp_path / "exports"
    exports_dir.mkdir(parents=True)
    output_path = exports_dir / "drone_test.wav"
    audio = np.zeros((2000, 2), dtype=np.float32)
    audio[100:400, :] = 0.35
    captured: dict = {}

    real_mkstemp = tempfile.mkstemp

    def spy_mkstemp(**kwargs):
        captured.update(kwargs)
        return real_mkstemp(**kwargs)

    with patch("dronmakr.generate.generate_sample.tempfile.mkstemp", side_effect=spy_mkstemp):
        _write_validated_export_wav(str(output_path), audio, 44100)

    assert captured.get("dir") == str(exports_dir.resolve())
