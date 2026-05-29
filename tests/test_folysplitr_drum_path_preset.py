import json
import os
import tempfile
import unittest
from unittest.mock import patch

from settings import (
    DRUM_PATH_PRESET_NAME_KEY,
    FOLYSPLITR_DRUM_PATH_PRESET_NAME,
    build_folysplitr_drum_path_preset,
    get_drum_path_presets,
    set_files_root,
)


class FolysplitrDrumPathPresetTests(unittest.TestCase):
    def test_build_folysplitr_drum_path_preset_maps_splits_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            files_root = os.path.join(tmp, "dronmakr-files")
            preset = build_folysplitr_drum_path_preset(files_root)
            self.assertEqual(
                preset["DRUM_KICK_PATHS"],
                os.path.abspath(os.path.join(files_root, "splits", "kick")),
            )
            self.assertEqual(
                preset["DRUM_SNARE_PATHS"],
                os.path.abspath(os.path.join(files_root, "splits", "snare")),
            )
            self.assertEqual(
                preset["DRUM_CYMBAL_PATHS"],
                os.path.abspath(os.path.join(files_root, "splits", "cymbal")),
            )

    def test_set_files_root_creates_folysplitr_preset(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = os.path.join(tmp, "settings.json")
            files_root = os.path.join(tmp, "dronmakr-files")
            with patch("settings.SETTINGS_PATH", settings_path):
                resolved = set_files_root(files_root)
                self.assertEqual(resolved, os.path.abspath(files_root))

                with open(settings_path, "r", encoding="utf-8") as handle:
                    saved = json.load(handle)

                presets = get_drum_path_presets(saved)
                self.assertIn(FOLYSPLITR_DRUM_PATH_PRESET_NAME, presets)
                self.assertEqual(
                    presets[FOLYSPLITR_DRUM_PATH_PRESET_NAME],
                    build_folysplitr_drum_path_preset(files_root),
                )

                entries = saved["DRUM_PATH_PRESETS"]
                folysplitr_entries = [
                    entry
                    for entry in entries
                    if entry.get(DRUM_PATH_PRESET_NAME_KEY)
                    == FOLYSPLITR_DRUM_PATH_PRESET_NAME
                ]
                self.assertEqual(len(folysplitr_entries), 1)

                for category in ("kick", "snare", "hihat", "clap"):
                    self.assertTrue(
                        os.path.isdir(os.path.join(files_root, "splits", category))
                    )


if __name__ == "__main__":
    unittest.main()
