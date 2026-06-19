from dronmakr.core.utils import (
    _default_subfolder_key,
    _export_subfolder_counts,
    _export_subfolder_slug_from_stem,
)


def test_collection_slug_uses_generated_name():
    stem = "drumpattern___vaelthorn___170bpm___4bars___abc123"
    assert _export_subfolder_slug_from_stem(stem, "collection") == "vaelthorn"


def test_style_slug_uses_style_segment():
    stem = "drumpattern___vaelthorn___170bpm___4bars___abc123"
    assert _export_subfolder_slug_from_stem(stem, "style") == "170bpm_4bars"


def test_collection_fallback_uses_sample_type():
    assert _export_subfolder_slug_from_stem("plain_recording", "collection", "bass") == "bass"


def test_style_fallback_uses_sample_type():
    assert _export_subfolder_slug_from_stem("plain_recording", "style", "drone") == "drone"


def test_collection_fallback_two_part_stem():
    assert _export_subfolder_slug_from_stem("drumpattern___mybeat", "collection") == "mybeat"


def test_collection_fallback_misc_for_other_type():
    assert _export_subfolder_slug_from_stem("plain_recording", "collection", "other") == "plain"


def test_style_fallback_misc_for_other_type():
    assert _export_subfolder_slug_from_stem("plain_recording", "style", "other") == "plain"


def test_subfolder_counts_include_fallback_keys():
    ordered = [
        {"name": "alpha", "type": "bass"},
        {"name": "beta", "type": "bass"},
        {"name": "gamma", "type": "drone"},
    ]
    counts = _export_subfolder_counts(ordered, "style")
    assert counts["bass"] == 2
    assert counts["drone"] == 1


def test_default_subfolder_key_collection_prefers_type():
    assert _default_subfolder_key("random.wav", "collection", "sweep") == "sweep"
