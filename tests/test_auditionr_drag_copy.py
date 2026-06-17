import os

from dronmakr.core.utils import (
    allocate_dragged_saved_filename,
    get_saved_files,
    is_dragged_saved_artifact,
)


def test_is_dragged_saved_artifact():
    assert is_dragged_saved_artifact("drone_foo_dragged.wav")
    assert is_dragged_saved_artifact("drone_foo_dragged_2.wav")
    assert not is_dragged_saved_artifact("drone_foo.wav")


def test_allocate_dragged_saved_filename_empty_dir(tmp_path):
    saved_dir = tmp_path / "saved"
    saved_dir.mkdir()
    assert allocate_dragged_saved_filename("drone_a.wav", str(saved_dir)) == "drone_a_dragged.wav"


def test_allocate_dragged_saved_filename_collision(tmp_path):
    saved_dir = tmp_path / "saved"
    saved_dir.mkdir()
    (saved_dir / "drone_a_dragged.wav").write_bytes(b"x")
    assert allocate_dragged_saved_filename("drone_a.wav", str(saved_dir)) == "drone_a_dragged_1.wav"


def test_allocate_dragged_saved_filename_multiple_collisions(tmp_path):
    saved_dir = tmp_path / "saved"
    saved_dir.mkdir()
    (saved_dir / "drone_a_dragged.wav").write_bytes(b"x")
    (saved_dir / "drone_a_dragged_1.wav").write_bytes(b"x")
    assert allocate_dragged_saved_filename("drone_a.wav", str(saved_dir)) == "drone_a_dragged_2.wav"


def test_get_saved_files_excludes_dragged_artifacts(monkeypatch, tmp_path):
    saved_dir = tmp_path / "saved"
    saved_dir.mkdir()
    (saved_dir / "approved.wav").write_bytes(b"a")
    (saved_dir / "draft_dragged.wav").write_bytes(b"b")
    (saved_dir / "draft_dragged_1.wav").write_bytes(b"c")

    import dronmakr.core.utils as utils

    monkeypatch.setattr(utils, "SAVED_DIR", str(saved_dir))

    names = {item["name"] for item in get_saved_files()}
    assert names == {"approved"}
