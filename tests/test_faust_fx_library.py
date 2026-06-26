"""Tests for built-in Faust FX library catalog and paths."""

from __future__ import annotations

import pytest

from dronmakr.audio import faust_fx_library as fxl


def test_list_faust_effects_has_ten_entries():
    effects = fxl.list_faust_effects()
    assert len(effects) == 17
    ids = {entry["id"] for entry in effects}
    assert "warm_saturation" in ids
    assert "cathedral_wash" in ids
    assert "abyss_wash" in ids


def test_list_faust_fx_categories_groups_effects():
    categories = fxl.list_faust_fx_categories()
    assert categories
    total = sum(len(cat.get("effects") or []) for cat in categories)
    assert total == 17
    reverb = next(cat for cat in categories if cat["id"] == "reverb")
    assert len(reverb["effects"]) == 7


def test_faust_fx_path_roundtrip():
    path = fxl.faust_fx_path_for_id("ghost_echo")
    assert path == "faustfx:ghost_echo"
    assert fxl.is_faust_fx_path(path)
    assert fxl.faust_fx_id_from_path(path) == "ghost_echo"


def test_resolve_faust_fx_dsp_path_exists():
    path = fxl.resolve_faust_fx_dsp_path("diffuse_cloud")
    assert path.endswith("resources/faust/fx/diffuse_cloud.dsp")


def test_faust_fx_path_exists():
    assert fxl.faust_fx_path_exists("faustfx:tape_wobble")
    assert not fxl.faust_fx_path_exists("faustfx:missing_effect")


def test_unknown_faust_fx_raises():
    with pytest.raises(ValueError, match="Unknown Faust library effect"):
        fxl.resolve_faust_fx_dsp_path("not_an_effect")


def test_wash_reverbs_render_usable_audio_in_fx_chain(tmp_path):
    import numpy as np
    import pretty_midi
    import time

    from dronmakr.audio.audio_host import (
        _build_plugin_graph,
        _inject_midi_notes_from_file,
        _render_output_is_usable,
        create_engine,
        load_instrument,
        load_plugin,
    )

    midi_path = tmp_path / "plate_test.mid"
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0, end=2))
    pm.instruments.append(instrument)
    pm.write(str(midi_path))

    wash_ids = [entry["id"] for entry in fxl.list_faust_effects() if entry["id"].endswith("_wash")]
    assert wash_ids

    for fx_id in wash_ids:
        engine = create_engine()
        inst = load_instrument(engine, "faust:sine_osc", name="instrument")
        fx = load_plugin(engine, fxl.faust_fx_path_for_id(fx_id), name="fx_0")
        assert fx.get_num_input_channels() <= 3, (
            f"{fx_id} exposes {fx.get_num_input_channels()} inputs; FX chain expects stereo in."
        )
        engine.load_graph(_build_plugin_graph(inst, [fx]))
        engine.set_bpm(120)
        _inject_midi_notes_from_file(inst, str(midi_path))
        time.sleep(0.2)
        engine.render(2.25)
        audio = engine.get_audio()
        assert _render_output_is_usable(audio), f"{fx_id} produced empty/invalid audio"
        assert np.isfinite(audio).all(), f"{fx_id} produced non-finite samples"
