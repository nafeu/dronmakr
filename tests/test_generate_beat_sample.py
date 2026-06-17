import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from dronmakr.generate.generate_sample import (
    BEAT_EXPORT_PEAK_DB,
    BEAT_EXPORT_SAMPLE_RATE,
    _finalize_beat_export_mix,
    _load_beat_hit,
    _mix_hit_into_buffer,
    generate_beat_sample,
)


class GenerateBeatSampleTests(unittest.TestCase):
    def test_export_skips_rows_without_kit_paths(self):
        sample = np.full((100, 1), 0.5, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            output_path = os.path.join(tmp, "beat.wav")
            pattern = {
                "_meta": {
                    "gridSize": "1/16",
                    "timeSignature": [4, 4],
                    "length": 1,
                },
                "kick": [1, 0, 0, 0] + [0] * 12,
                "prca": [0, 1, 0, 0] + [0] * 12,
            }

            with patch(
                "dronmakr.generate.generate_sample._load_beat_hit",
                return_value=sample,
            ):
                result = generate_beat_sample(
                    bpm=120,
                    bars=1,
                    output=output_path,
                    humanize=False,
                    style="",
                    swing=0.0,
                    play=False,
                    pattern_config={
                        "gridSize": "1/16",
                        "timeSignature": [4, 4],
                        "length": 1,
                    },
                    kit_paths={"kick": "/tmp/kick.wav"},
                    pattern_data=pattern,
                    loops=1,
                )

            self.assertEqual(result, output_path)
            self.assertTrue(os.path.isfile(output_path))
            audio, sr = sf.read(output_path, dtype="float32")
            self.assertEqual(sr, BEAT_EXPORT_SAMPLE_RATE)
            self.assertGreater(float(np.max(np.abs(audio))), 0.0)

    def test_float_mix_preserves_overlapping_transients(self):
        hit = np.ones((64, 1), dtype=np.float32) * 0.9
        mix = np.zeros((128, 1), dtype=np.float32)
        for _ in range(8):
            _mix_hit_into_buffer(mix, hit, 0, 1.0)
        self.assertAlmostEqual(float(np.max(np.abs(mix))), 7.2, places=5)

    def test_load_beat_hit_resamples_to_export_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hit.wav")
            t = np.linspace(0, 0.01, 480, endpoint=False, dtype=np.float32)
            sf.write(path, np.sin(2 * np.pi * 440 * t), 48000, subtype="PCM_16")
            hit = _load_beat_hit(path)
            self.assertEqual(hit.shape[0], 441)
            self.assertEqual(hit.dtype, np.float32)

    def test_finalize_beat_export_mix_peaks_at_target(self):
        mix = np.zeros((1000, 1), dtype=np.float32)
        mix[0] = 0.25
        out = _finalize_beat_export_mix(mix)
        target = 10 ** (BEAT_EXPORT_PEAK_DB / 20.0)
        self.assertAlmostEqual(float(np.max(np.abs(out))), target, places=5)


if __name__ == "__main__":
    unittest.main()
