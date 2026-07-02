import unittest
from pathlib import Path
from unittest.mock import patch

from dronmakr.apps.folysplitr import _recordings_queue_sort_key, _sort_recordings_queue


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
            ordered = _sort_recordings_queue(paths)
        self.assertEqual(
            [path.name for path in ordered],
            ["rec-split-01.wav", "rec-split-02.wav", "rec-split-03.wav"],
        )

    def test_split_siblings_sort_chronologically_when_mtimes_differ(self):
        paths = [
            Path("rec-split-05.wav"),
            Path("rec-split-01.wav"),
            Path("rec-split-03.wav"),
        ]

        def fake_stat(self):
            idx = int(self.stem.rsplit("-", 1)[-1])
            return type("Stat", (), {"st_mtime": float(idx)})()

        with patch.object(Path, "stat", fake_stat):
            ordered = _sort_recordings_queue(paths)
        self.assertEqual(
            [path.name for path in ordered],
            ["rec-split-01.wav", "rec-split-03.wav", "rec-split-05.wav"],
        )

    def test_newer_recording_session_sorts_before_older(self):
        older = Path("older.wav")
        newer = Path("newer-split-01.wav")

        def fake_stat(self):
            mtime = 200.0 if self.name.startswith("newer") else 100.0
            return type("Stat", (), {"st_mtime": mtime})()

        with patch.object(Path, "stat", fake_stat):
            ordered = _sort_recordings_queue([older, newer])
        self.assertEqual([path.name for path in ordered], ["newer-split-01.wav", "older.wav"])

    def test_legacy_sort_key_still_orders_split_indices(self):
        mtime = 1_700_000_000.0
        paths = [Path("rec-split-02.wav"), Path("rec-split-01.wav")]

        def fake_stat(self):
            return type("Stat", (), {"st_mtime": mtime})()

        with patch.object(Path, "stat", fake_stat):
            ordered = sorted(paths, key=_recordings_queue_sort_key)
        self.assertEqual([path.name for path in ordered], ["rec-split-01.wav", "rec-split-02.wav"])


if __name__ == "__main__":
    unittest.main()
