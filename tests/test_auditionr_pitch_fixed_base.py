import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from dronmakr.apps import auditionr as aud
from dronmakr.core import paths as managed_paths
from dronmakr.core.utils import refresh_managed_path_constants


def _bootstrap_auditionr_paths(tmp_root: str) -> None:
    import dronmakr.core.utils as utils

    utils.FILES_ROOT = tmp_root
    utils.TEMP_DIR = os.path.join(tmp_root, "temp")
    utils.EXPORTS_DIR = os.path.join(tmp_root, "exports")
    os.makedirs(utils.TEMP_DIR, exist_ok=True)
    os.makedirs(utils.EXPORTS_DIR, exist_ok=True)
    refresh_managed_path_constants()
    aud.refresh_auditionr_paths()


def _write_tone(path: str, freq: float = 440.0, sr: int = 44100) -> None:
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    sf.write(path, 0.5 * np.sin(2 * np.pi * freq * t), sr)


def _estimate_fundamental(path: str) -> float:
    mono, sr = sf.read(path, dtype="float32")
    if mono.ndim > 1:
        mono = mono[:, 0]
    mono = mono - mono.mean()
    corr = np.correlate(mono, mono, mode="full")[len(mono) - 1 :]
    min_lag = int(sr / 2000)
    max_lag = int(sr / 200)
    lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
    return float(sr / lag)


@pytest.fixture()
def pitch_env(tmp_path):
    _bootstrap_auditionr_paths(str(tmp_path))
    wav_path = tmp_path / "exports" / "tone.wav"
    _write_tone(str(wav_path))
    yield str(wav_path)
    aud._clear_pitch_state_for_file(str(wav_path))


def test_pitch_fixed_base_applies_absolute_target(pitch_env):
    base_freq = _estimate_fundamental(pitch_env)

    aud._apply_pitch_with_fixed_base(pitch_env, 12, mode="preserve")
    shifted = _estimate_fundamental(pitch_env)
    assert 820 < shifted < 940
    assert abs(shifted - base_freq * 2) < 40


def test_pitch_fixed_base_repeat_same_target_is_idempotent(pitch_env):
    aud._apply_pitch_with_fixed_base(pitch_env, 12, mode="preserve")
    first = _estimate_fundamental(pitch_env)
    aud._apply_pitch_with_fixed_base(pitch_env, 12, mode="preserve")
    second = _estimate_fundamental(pitch_env)
    assert abs(second - first) < 5


def test_pitch_fixed_base_zero_restores_base_snapshot(pitch_env):
    base_freq = _estimate_fundamental(pitch_env)
    aud._apply_pitch_with_fixed_base(pitch_env, 12, mode="preserve")
    aud._apply_pitch_with_fixed_base(pitch_env, 0, mode="preserve")
    restored = _estimate_fundamental(pitch_env)
    assert abs(restored - base_freq) < 5
