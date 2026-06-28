"""Editor session .ddstate paths must resolve after audio-worker capture."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from dronmakr.apps import generatr_plugins as gp
from dronmakr.core.utils import refresh_managed_path_constants


@pytest.fixture
def files_root(tmp_path):
    root = tmp_path / "dronmakr-files"
    root.mkdir()
    (root / "config").mkdir()

    def managed_file(*parts: str) -> str:
        return str(root.joinpath(*parts))

    with patch("dronmakr.core.utils.has_configured_files_root", return_value=True), patch(
        "dronmakr.core.utils.get_files_root_path", return_value=root
    ), patch("dronmakr.core.paths.get_managed_file", side_effect=managed_file), patch(
        "dronmakr.core.utils.get_managed_file", side_effect=managed_file
    ):
        refresh_managed_path_constants()
        yield root


def test_generatr_session_dir_uses_files_root_temp(files_root):
    session_dir = gp.generatr_session_dir()
    assert session_dir == str((files_root / "temp" / "generatr-plugin-sessions").resolve())
    assert os.path.isdir(session_dir)


def test_resolve_preset_state_path_finds_legacy_relative_session_file(files_root, monkeypatch, tmp_path):
    legacy_dir = tmp_path / "generatr-plugin-sessions"
    legacy_dir.mkdir()
    legacy_file = legacy_dir / "legacy-test.ddstate"
    legacy_file.write_bytes(b"ddstate")
    monkeypatch.chdir(tmp_path)

    resolved = gp._resolve_preset_state_path("generatr-plugin-sessions/legacy-test.ddstate")
    assert resolved == str(legacy_file.resolve())


def test_persist_preset_state_copies_session_file(files_root):
    session_dir = gp.generatr_session_dir()
    source = Path(session_dir) / "capture.ddstate"
    source.write_bytes(b"plugin-state")
    dest = gp._persist_preset_state(str(source), "My Patch")
    assert os.path.isfile(dest)
    assert Path(dest).read_bytes() == b"plugin-state"
