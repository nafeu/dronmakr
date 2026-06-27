"""Built-in Faust instruments when presets.json has no saved instruments."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dronmakr.audio.faust_library import (
    is_faust_instrument_path,
    list_faust_instruments,
    pick_random_faust_instrument_preset,
)
from dronmakr.generate.generate_sample import generate_drone_sample


class TestFaustInstrumentFallback(unittest.TestCase):
    def test_pick_random_faust_instrument_preset(self):
        preset = pick_random_faust_instrument_preset()
        self.assertTrue(is_faust_instrument_path(preset["plugin_path"]))
        self.assertTrue(preset.get("name"))

    def test_generate_drone_sample_without_saved_instruments(self):
        with tempfile.TemporaryDirectory() as tmp:
            presets_path = Path(tmp) / "presets.json"
            presets_path.write_text("[]", encoding="utf-8")
            midi_path = Path(tmp) / "input.mid"
            # Minimal valid MIDI header (type 0, 0 tracks — enough for path existence).
            midi_path.write_bytes(b"MThd\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00")
            output_path = Path(tmp) / "out.wav"

            fake_audio = np.zeros((2, 44100), dtype=np.float32)

            with patch(
                "dronmakr.generate.generate_sample.render_midi_chain_from_paths",
                return_value=fake_audio,
            ), patch(
                "dronmakr.generate.generate_sample._write_validated_export_wav",
            ), patch(
                "dronmakr.generate.generate_sample.midi_musical_end_seconds",
                return_value=1.0,
            ), patch(
                "dronmakr.audio.audio_worker.delegate_generate_drone_sample_if_needed",
                return_value=None,
            ), patch(
                "dronmakr.presets.preset_authoring.assert_plugin_role_for_slot",
            ):
                result = generate_drone_sample(
                    input_path=str(midi_path),
                    output_path=str(output_path),
                    presets_path=str(presets_path),
                    instrument=None,
                    effect="none",
                    render_duration_sec=1.0,
                    instrument_selection=None,
                    fx_slots=None,
                )

            self.assertTrue(str(result).endswith(".wav"))
            self.assertTrue(list_faust_instruments())


if __name__ == "__main__":
    unittest.main()
