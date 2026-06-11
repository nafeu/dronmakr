import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from process_sample import pad_sample
from processing_actions import action_from_bracket_segment, parse_single_processing_spec


def test_pad_sample_after():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.wav"
        sr = 44100
        audio = np.ones(sr, dtype=np.float32) * 0.5
        sf.write(path, audio, sr)

        pad_sample(str(path), 0.5, side="after")

        out, out_sr = sf.read(path)
        assert out_sr == sr
        assert len(out) == int(sr * 1.5)
        assert np.max(np.abs(out[:sr] - 0.5)) < 1e-6
        assert np.max(np.abs(out[sr:])) < 1e-6


def test_pad_sample_before():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.wav"
        sr = 22050
        audio = np.ones(int(sr * 0.5), dtype=np.float32) * 0.25
        sf.write(path, audio, sr)

        pad_sample(str(path), 1.0, side="before")

        out, _ = sf.read(path)
        assert len(out) == len(audio) * 2
        assert np.max(np.abs(out[: len(audio)])) < 1e-6
        assert np.max(np.abs(out[len(audio) :] - 0.25)) < 1e-6


def test_padding_bracket_spec():
    action = action_from_bracket_segment("padding:[side=after][amount=0.25]")
    assert action["command"] == "pad_sample"
    assert action["params"]["amount"] == 0.25
    assert action["params"]["side"] == "after"

    parsed = parse_single_processing_spec("padding:[side=before][amount=1]")
    assert parsed["command"] == "pad_sample"
    assert parsed["params"]["amount"] == 1.0
    assert parsed["params"]["side"] == "before"
