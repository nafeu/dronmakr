import os
import tempfile
import unittest
from unittest.mock import patch

from pydub import AudioSegment

from generate_sample import generate_beat_sample


class GenerateBeatSampleTests(unittest.TestCase):
    def test_export_skips_rows_without_kit_paths(self):
        sample = AudioSegment.silent(duration=50, frame_rate=44100)

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
                "generate_sample.load_sample_from_path",
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


if __name__ == "__main__":
    unittest.main()
