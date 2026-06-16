import unittest
from pathlib import Path
from unittest.mock import patch

from dronmakr.apps.folysplitr import _recordings_queue_sort_key


class FolysplitrRecordingsSortTests(unittest.TestCase):
    def test_split_siblings_sort_chronologically_within_same_mtime(self):
        mtime = 1_700_000_000.0
        paths = [
            Path("rec-split-03.wav"),
            Path("rec-split-01.wav"),
            Path("rec-split-02.wav"),
        ]

        def fake_stat(self):
            return type("Stat", (), {"st_mtime": mtime})()

        with patch.object(Path, "stat", fake_stat):
            ordered = sorted(paths, key=_recordings_queue_sort_key)
        self.assertEqual(
            [path.name for path in ordered],
            ["rec-split-01.wav", "rec-split-02.wav", "rec-split-03.wav"],
        )

    def test_newer_recording_session_sorts_before_older(self):
        older = Path("older.wav")
        newer = Path("newer-split-01.wav")

        def fake_stat(self):
            mtime = 200.0 if self.name.startswith("newer") else 100.0
            return type("Stat", (), {"st_mtime": mtime})()

        with patch.object(Path, "stat", fake_stat):
            ordered = sorted([older, newer], key=_recordings_queue_sort_key)
        self.assertEqual([path.name for path in ordered], ["newer-split-01.wav", "older.wav"])


if __name__ == "__main__":
    unittest.main()
