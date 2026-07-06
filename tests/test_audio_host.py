import threading
import time

import numpy as np

from dronmakr.audio.audio_host import (
    _wait_for_editor_preview_arm,
    daw_audio_to_samples_channels,
    downmix_audio_for_export,
    samples_channels_to_daw_audio,
)


def test_daw_audio_layout_roundtrip():
    samples_ch = np.random.randn(128, 2).astype(np.float32)
    daw = samples_channels_to_daw_audio(samples_ch)
    assert daw.shape[0] == 2
    back = daw_audio_to_samples_channels(daw)
    assert back.shape == samples_ch.shape
    np.testing.assert_allclose(back, samples_ch, rtol=1e-5, atol=1e-6)


def test_daw_audio_layout_handles_many_output_channels():
    daw = np.random.randn(16, 4096).astype(np.float32)
    out = daw_audio_to_samples_channels(daw)
    assert out.shape == (4096, 16)


def test_downmix_audio_for_export_keeps_stereo():
    audio = np.random.randn(1024, 16).astype(np.float32)
    stereo = downmix_audio_for_export(audio)
    assert stereo.shape == (1024, 2)


def test_wait_for_editor_preview_arm_returns_true_when_armed():
    preview_armed = threading.Event()
    stop = threading.Event()
    preview_armed.set()
    assert _wait_for_editor_preview_arm(preview_armed, stop) is True


def test_wait_for_editor_preview_arm_returns_false_when_stopped_first():
    preview_armed = threading.Event()
    stop = threading.Event()
    stop.set()
    assert _wait_for_editor_preview_arm(preview_armed, stop) is False


def test_wait_for_editor_preview_arm_waits_until_armed():
    preview_armed = threading.Event()
    stop = threading.Event()

    def arm_later() -> None:
        time.sleep(0.05)
        preview_armed.set()

    threading.Thread(target=arm_later, daemon=True).start()
    assert _wait_for_editor_preview_arm(preview_armed, stop) is True
