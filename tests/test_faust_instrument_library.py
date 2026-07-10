"""Tests for built-in Faust instrument library I/O and render sanity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import pytest

from dronmakr.audio import faust_library as fl
from dronmakr.audio.audio_host import (
    _render_output_is_usable,
    create_engine,
    load_instrument,
    render_midi_chain_from_paths,
)


def test_all_faust_instruments_expose_synth_io():
    for entry in fl.list_faust_instruments():
        engine = create_engine()
        inst = load_instrument(engine, fl.faust_path_for_id(entry["id"]), name="instrument")
        assert inst.get_num_input_channels() == fl.FAUST_INSTRUMENT_INPUT_CHANNELS, (
            f"{entry['id']} has {inst.get_num_input_channels()} inputs; "
            "instrument slot expects a 0-input synth."
        )
        assert inst.get_num_output_channels() == fl.FAUST_INSTRUMENT_OUTPUT_CHANNELS, (
            f"{entry['id']} has {inst.get_num_output_channels()} outputs; "
            "instrument slot expects stereo out."
        )
        assert inst.num_voices == fl.FAUST_POLYPHONY_VOICES


@pytest.fixture
def midi_with_note(tmp_path: Path) -> str:
    midi_path = tmp_path / "note.mid"
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0, end=2))
    pm.instruments.append(instrument)
    pm.write(str(midi_path))
    return str(midi_path)


def test_all_faust_instruments_render_usable_audio(midi_with_note: str):
    for entry in fl.list_faust_instruments():
        audio = render_midi_chain_from_paths(
            fl.faust_path_for_id(entry["id"]),
            None,
            [],
            midi_with_note,
            duration_sec=2.0,
        )
        assert _render_output_is_usable(audio), f"{entry['id']} produced empty/invalid audio"
        assert np.isfinite(audio).all(), f"{entry['id']} produced non-finite samples"


def test_faust_instrument_dsp_files_use_stereo_process_line():
    for entry in fl.list_faust_instruments():
        dsp_path = Path(fl.resolve_faust_dsp_path(entry["id"]))
        text = dsp_path.read_text(encoding="utf-8")
        assert "freq = nentry" in text, f"{entry['id']} missing freq nentry"
        assert "gain = nentry" in text, f"{entry['id']} missing gain nentry"
        assert 'gate = button("gate")' in text, f"{entry['id']} missing gate button"
        assert "<: _, _" in text, f"{entry['id']} missing stereo split (`<: _, _`)"
