"""Filter post-processing must audibly change typical drone-heavy material."""

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import soundfile as sf

from dronmakr.processing.processing_actions import (
    apply_processing_command,
    parse_single_processing_spec,
)


def _write_drone_like_wav(path: str) -> None:
    sr = 44100
    t = np.linspace(0, 2, sr * 2, dtype=np.float32)
    mono = (
        0.4 * np.sin(2 * np.pi * 55 * t)
        + 0.25 * np.sin(2 * np.pi * 110 * t)
        + 0.2 * np.sin(2 * np.pi * 220 * t)
        + 0.15 * np.sin(2 * np.pi * 880 * t)
    )
    sf.write(path, np.column_stack([mono, mono]), sr, subtype="PCM_16")


def _max_diff(path_a: str, path_b: str) -> float:
    a, _ = sf.read(path_a, dtype="float32", always_2d=True)
    b, _ = sf.read(path_b, dtype="float32", always_2d=True)
    return float(np.max(np.abs(a - b)))


def test_filter_bracket_defaults_are_audible_on_drone_material():
    fd, source = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        _write_drone_like_wav(source)
        specs = [
            "filter:[kind=lpf]",
            "filter:[kind=hpf]",
            "filter:[kind=bpf][low_hz=300][high_hz=6000]",
            "filter:[kind=lpf][cutoff_hz=800]",
            "filter:[kind=lpf][cutoff_hz=400]",
            "filter:lpf--",
        ]
        for spec in specs:
            action = parse_single_processing_spec(spec)
            target = source + ".out.wav"
            shutil.copy2(source, target)
            apply_processing_command(target, action["command"], action.get("params") or {})
            diff = _max_diff(source, target)
            assert diff > 0.01, f"{spec} did not change audio enough (max_diff={diff})"
            os.unlink(target)
    finally:
        os.unlink(source)
