"""Tests for Edit Samples plugin-patch processing spec."""

from __future__ import annotations

from dronmakr.processing.processing_actions import (
    _load_fx_patch_options_for_edit_ui,
    get_processing_actions_payload,
    parse_single_processing_spec,
)


def test_parse_plugin_patch_spec():
    parsed = parse_single_processing_spec("plugin_patch:[patch=My Plate Chain]")
    assert parsed["command"] == "apply_plugin_patch_to_sample"
    assert parsed["params"]["patch"] == "My Plate Chain"


def test_processing_payload_includes_plugin_patch_type():
    payload = get_processing_actions_payload()
    keys = {row["key"] for row in payload["types"]}
    assert "plugin_patch" in keys
    assert isinstance(payload["pluginPatchOptions"], list)


def test_load_fx_patch_options_returns_list():
    options = _load_fx_patch_options_for_edit_ui()
    assert isinstance(options, list)
    for row in options:
        assert row.get("name")
        assert row.get("label")


def test_finalize_fx_processed_audio_preserves_level():
    import numpy as np

    from dronmakr.audio.process_sample import _finalize_fx_processed_audio

    processed = np.column_stack(
        [np.full(1000, 0.5, dtype=np.float32), np.full(1000, 0.5, dtype=np.float32)]
    )
    out = _finalize_fx_processed_audio(processed, target_samples=1000)
    assert out.shape == (1000, 2)
    assert float(np.max(np.abs(out))) == 0.5


def test_apply_plugin_patch_keeps_audible_level(tmp_path):
    import json

    import numpy as np
    import soundfile as sf

    from dronmakr.audio.process_sample import apply_plugin_patch_to_sample

    wav_path = tmp_path / "input.wav"
    presets_path = tmp_path / "presets.json"
    sr = 44100
    t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False, dtype=np.float32)
    stereo = np.column_stack([0.45 * np.sin(2 * np.pi * 220 * t)] * 2)
    sf.write(wav_path, stereo, sr, subtype="PCM_16")
    presets_path.write_text(
        json.dumps(
            [
                {
                    "name": "Test Wash",
                    "type": "effect",
                    "plugin_path": "faustfx:cathedral_wash",
                    "plugin_name": "Faust",
                    "preset_path": "",
                }
            ]
        ),
        encoding="utf-8",
    )

    apply_plugin_patch_to_sample(str(wav_path), "Test Wash", presets_path=str(presets_path))

    result, _ = sf.read(wav_path, dtype="float32", always_2d=True)
    peak = float(np.max(np.abs(result)))
    assert peak > 0.05, f"Expected audible output, got peak {peak}"
