import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from dronmakr.audio.process_sample import apply_transpose_pitch_by_resampling_inplace
from dronmakr.processing.processing_actions import parse_single_processing_spec


def _estimate_fundamental(mono: np.ndarray, sample_rate: int) -> float:
    x = np.asarray(mono, dtype=np.float64).reshape(-1)
    x -= x.mean()
    corr = np.correlate(x, x, mode="full")[len(x) - 1 :]
    min_lag = int(sample_rate / 2000)
    max_lag = int(sample_rate / 200)
    if max_lag <= min_lag:
        return 0.0
    lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
    return float(sample_rate / lag)


def test_resample_transpose_raises_pitch_and_shortens():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tone.wav"
        sr = 44100
        t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(path, tone, sr)

        apply_transpose_pitch_by_resampling_inplace(str(path), 12)

        out, out_sr = sf.read(path, dtype="float32")
        assert out_sr == sr
        mono = out[:, 0] if out.ndim > 1 else out
        assert len(mono) < sr * 0.6
        freq = _estimate_fundamental(mono, sr)
        assert 820 < freq < 940


def test_pitch_bracket_spec_resample_mode():
    parsed = parse_single_processing_spec("pitch:[mode=resample][semitones=7][cents=0]")
    assert parsed["command"] == "pitch_shift_transpose_sample"
    assert parsed["params"]["semitones"] == 7.0
