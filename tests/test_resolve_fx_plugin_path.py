from unittest.mock import patch

import pytest

from dronmakr.presets import preset_authoring as pa


@patch("dronmakr.presets.preset_authoring.os.path.exists", return_value=True)
@patch("dronmakr.presets.preset_authoring.list_installed_plugins")
@patch("dronmakr.presets.preset_authoring.plugin_settings_tuple")
def test_resolve_fx_plugin_path_prefers_fx_sibling(
    mock_settings_tuple,
    mock_list_installed,
    _mock_exists,
):
    mock_settings_tuple.return_value = (["/plugins"], [])
    mock_list_installed.return_value = [
        "/plugins/Reaktor 6.vst3",
        "/plugins/Reaktor 6 FX.vst3",
    ]

    resolved = pa.resolve_fx_plugin_path("/plugins/Reaktor 6.vst3")

    assert resolved == "/plugins/Reaktor 6 FX.vst3"


@patch("dronmakr.presets.preset_authoring.os.path.exists", return_value=True)
@patch("dronmakr.presets.preset_authoring.list_installed_plugins")
@patch("dronmakr.presets.preset_authoring.plugin_settings_tuple")
def test_resolve_fx_plugin_path_prefers_fm8_fx_sibling(
    mock_settings_tuple,
    mock_list_installed,
    _mock_exists,
):
    mock_settings_tuple.return_value = (["/plugins"], [])
    mock_list_installed.return_value = [
        "/plugins/FM8.vst3",
        "/plugins/FM8 FX.vst3",
    ]

    resolved = pa.resolve_fx_plugin_path("/plugins/FM8.vst3")

    assert resolved == "/plugins/FM8 FX.vst3"


@patch("dronmakr.presets.preset_authoring.os.path.exists", return_value=True)
@patch("dronmakr.presets.preset_authoring.list_installed_plugins")
@patch("dronmakr.presets.preset_authoring.plugin_settings_tuple")
def test_resolve_fx_plugin_path_matches_attached_fx_stem(
    mock_settings_tuple,
    mock_list_installed,
    _mock_exists,
):
    mock_settings_tuple.return_value = (["/plugins"], [])
    mock_list_installed.return_value = [
        "/plugins/FM8.component",
        "/plugins/FM8FX.vst3",
    ]

    resolved = pa.resolve_fx_plugin_path("/plugins/FM8.component")

    assert resolved == "/plugins/FM8FX.vst3"


@patch("dronmakr.presets.preset_authoring.os.path.exists", return_value=True)
@patch("dronmakr.presets.preset_authoring.list_installed_plugins")
@patch("dronmakr.presets.preset_authoring.plugin_settings_tuple")
def test_resolve_fx_plugin_path_keeps_explicit_fx(
    mock_settings_tuple,
    mock_list_installed,
    _mock_exists,
):
    mock_settings_tuple.return_value = (["/plugins"], [])
    mock_list_installed.return_value = [
        "/plugins/Reaktor 6.vst3",
        "/plugins/Reaktor 6 FX.vst3",
    ]

    resolved = pa.resolve_fx_plugin_path("/plugins/Reaktor 6 FX.vst3")

    assert resolved == "/plugins/Reaktor 6 FX.vst3"


def test_resolve_fx_plugin_path_passes_through_faust_fx():
    resolved = pa.resolve_fx_plugin_path("faustfx:ghost_echo")
    assert resolved == "faustfx:ghost_echo"


def test_resolve_fx_plugin_path_rejects_faust_instrument():
    with pytest.raises(ValueError, match="Faust instrument"):
        pa.resolve_fx_plugin_path("faust:sine_osc")
