import pretty_midi
import pytest

from dronmakr.generate.generate_midi import save_drone_midi_export


def test_save_drone_midi_export_writes_under_midi_dir(tmp_path, monkeypatch):
    files_root = tmp_path / "files"
    midi_dir = files_root / "midi"
    midi_dir.mkdir(parents=True)

    import dronmakr.core.utils as utils

    monkeypatch.setattr(utils, "MIDI_DIR", str(midi_dir))
    monkeypatch.setattr(utils, "refresh_managed_path_constants", lambda: None)

    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=1.0))
    pm.instruments.append(instrument)

    path = save_drone_midi_export(pm, "abc12345")
    assert path == str(midi_dir / "abc12345.mid")
    assert (midi_dir / "abc12345.mid").is_file()


def test_save_drone_midi_export_rejects_empty_id(tmp_path, monkeypatch):
    midi_dir = tmp_path / "midi"
    midi_dir.mkdir()

    import dronmakr.core.utils as utils

    monkeypatch.setattr(utils, "MIDI_DIR", str(midi_dir))
    monkeypatch.setattr(utils, "refresh_managed_path_constants", lambda: None)

    pm = pretty_midi.PrettyMIDI()
    with pytest.raises(ValueError, match="unique_id"):
        save_drone_midi_export(pm, "   ")
