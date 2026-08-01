import pretty_midi

from dronmakr.apps.auditionr import resolve_export_midi_abs_path
from dronmakr.generate.generate_midi import build_midi_preview_payload, save_drone_midi_export


def test_build_midi_preview_payload_normalizes_note_timing():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
    instrument.notes.append(pretty_midi.Note(velocity=50, pitch=72, start=1.0, end=2.0))
    pm.instruments.append(instrument)

    preview = build_midi_preview_payload(pm)
    assert preview["durationSec"] == 2.0
    assert len(preview["events"]) == 2
    assert preview["events"][0]["start"] == 0.0
    assert preview["events"][1]["end"] == 1.0


def test_build_midi_preview_payload_honors_total_duration_for_trailing_silence():
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=2.0))
    pm.instruments.append(instrument)

    preview = build_midi_preview_payload(pm, total_duration_sec=4.0)
    assert preview["durationSec"] == 4.0
    assert preview["events"][0]["end"] == 0.5


def test_resolve_export_midi_abs_path_and_preview_payload(tmp_path, monkeypatch):
    import dronmakr.core.utils as managed_paths

    midi_dir = tmp_path / "midi"
    midi_dir.mkdir(parents=True)
    monkeypatch.setattr(managed_paths, "MIDI_DIR", str(midi_dir))
    monkeypatch.setattr(managed_paths, "refresh_managed_path_constants", lambda: None)

    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=0.5))
    pm.instruments.append(instrument)
    save_drone_midi_export(pm, "drone_test1234")

    abs_path = resolve_export_midi_abs_path("drone_test1234")
    assert abs_path and abs_path.endswith("drone_test1234.mid")

    loaded = pretty_midi.PrettyMIDI(abs_path)
    preview = build_midi_preview_payload(loaded)
    assert len(preview["events"]) == 1
