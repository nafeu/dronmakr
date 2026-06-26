import numpy as np

from dronmakr.audio.audio_host import (
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
