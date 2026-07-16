"""Unit tests for DRONMAKR_TEST isolation."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest


@pytest.fixture()
def test_env(tmp_path, monkeypatch):
    root = tmp_path / "test-root"
    root.mkdir()
    monkeypatch.setenv("DRONMAKR_TEST", "1")
    monkeypatch.setenv("DRONMAKR_TEST_ROOT", str(root))
    monkeypatch.setenv("DRONMAKR_TEST_FILES_ROOT", str(root / "files"))
    yield root


def test_settings_path_isolated_in_test_mode(test_env, monkeypatch):
    import importlib

    import dronmakr.core.settings as settings_mod

    importlib.reload(settings_mod)
    assert settings_mod.SETTINGS_PATH == str(test_env / "config" / "settings.json")


def test_updater_disabled_in_test_mode(monkeypatch):
    monkeypatch.setenv("DRONMAKR_TEST", "1")
    from dronmakr.core.updater import fetch_update_info_throttled

    with patch("dronmakr.core.updater.check_for_update") as mocked:
        assert fetch_update_info_throttled(force=True) is None
        mocked.assert_not_called()


def test_bootstrap_seeds_plugin_paths(test_env, monkeypatch):
    import importlib

    import dronmakr.core.settings as settings_mod
    import dronmakr.core.test_mode as test_mode_mod

    importlib.reload(settings_mod)
    importlib.reload(test_mode_mod)

    test_mode_mod.bootstrap_test_settings_if_needed()
    with open(settings_mod.SETTINGS_PATH, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    assert data.get("PLUGIN_PATHS")
