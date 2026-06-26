"""Tests for FX picker payload including Faust library."""

from __future__ import annotations

from unittest.mock import patch

from dronmakr.apps import generatr_plugins as gp


@patch("dronmakr.apps.generatr_plugins._load_presets_list", return_value=[])
@patch("dronmakr.apps.generatr_plugins.plugin_settings_tuple", return_value=([""], []))
def test_effect_picker_includes_faust_library(_mock_settings, _mock_presets):
    payload = gp.get_drone_picker_payload("effect")
    assert payload["role"] == "effect"
    assert len(payload["library"]) == 17
    assert payload["libraryCategories"]
    assert payload["libraryCategories"][0].get("effects")
    assert "instruments" not in payload["libraryCategories"][0]
