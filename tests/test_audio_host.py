import threading
import time

import numpy as np

from dronmakr.audio.audio_host import (
    EXPORT_PEAK_LIMIT,
    HEADROOM_GAIN,
    _wait_for_editor_preview_arm,
    daw_audio_to_samples_channels,
    downmix_audio_for_export,
    finalize_rendered_audio,
    limit_audio_peak,
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


def test_limit_audio_peak_scales_hot_signal():
    audio = np.array([[2.0, -1.5], [1.0, 0.5]], dtype=np.float32)
    limited = limit_audio_peak(audio, peak_limit=EXPORT_PEAK_LIMIT)
    assert float(np.max(np.abs(limited))) <= EXPORT_PEAK_LIMIT + 1e-6


def test_limit_audio_peak_leaves_quiet_signal_unchanged():
    audio = np.full((8, 2), 0.25, dtype=np.float32)
    limited = limit_audio_peak(audio, peak_limit=EXPORT_PEAK_LIMIT)
    np.testing.assert_allclose(limited, audio)


def test_scaled_midi_velocity_reduces_chord_velocities():
    from dronmakr.audio.audio_host import _scaled_midi_velocity

    solo = _scaled_midi_velocity(100, polyphony=1, velocity_gain=0.58)
    chord = _scaled_midi_velocity(100, polyphony=4, velocity_gain=0.58)
    assert chord < solo
    assert chord >= 1


def test_finalize_rendered_audio_preserves_level_throughout():
    """Static finalize must not pump gain after the first peak (no limiter)."""
    t = np.linspace(0, 1, 4000, dtype=np.float32)
    tone = 0.85 * np.sin(2 * np.pi * 5 * t)
    audio = np.column_stack([tone, tone])
    out = finalize_rendered_audio(audio, 44100, headroom_gain=0.4)
    rms_start = float(np.sqrt(np.mean(out[:400] ** 2)))
    rms_later = float(np.sqrt(np.mean(out[1600:2400] ** 2)))
    assert rms_start > 1e-6
    assert abs(rms_later - rms_start) / rms_start < 0.12


def test_finalize_rendered_audio_tames_hot_polyphonic_render():
    audio = np.full((1000, 2), 4.0, dtype=np.float32)
    out = finalize_rendered_audio(audio, 44100, headroom_gain=HEADROOM_GAIN)
    assert float(np.max(np.abs(out))) <= EXPORT_PEAK_LIMIT + 1e-5
