from unittest.mock import patch

from dronmakr.apps import generatr_plugins as gp


@patch("dronmakr.apps.generatr_plugins.save_presets_json")
@patch("dronmakr.apps.generatr_plugins.load_presets_json")
@patch("dronmakr.apps.generatr_plugins.name_exists")
@patch("dronmakr.apps.generatr_plugins._persist_preset_state", return_value="/tmp/new.ddstate")
@patch("dronmakr.apps.generatr_plugins._resolve_drone_selection")
@patch("dronmakr.apps.generatr_plugins._plugin_path_exists", return_value=True)
def test_save_drone_preset_rejects_duplicate_without_overwrite(
    _mock_exists_path,
    mock_resolve,
    _mock_persist,
    mock_name_exists,
    mock_load,
    mock_save,
):
    mock_name_exists.return_value = True
    mock_resolve.return_value = ("/plugins/Synth.vst3", "Synth", "/tmp/old.ddstate", "Synth")

    try:
        gp.save_drone_preset(
            role="instrument",
            name="My Patch",
            instrument_selection={"kind": "plugin", "pluginPath": "/plugins/Synth.vst3"},
        )
    except ValueError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    mock_save.assert_not_called()


@patch("dronmakr.presets.preset_authoring.delete_preset_by_name", return_value=True)
@patch("dronmakr.apps.generatr_plugins.save_presets_json")
@patch("dronmakr.apps.generatr_plugins.load_presets_json", return_value=[])
@patch("dronmakr.apps.generatr_plugins.name_exists", return_value=True)
@patch("dronmakr.apps.generatr_plugins._persist_preset_state", return_value="/tmp/new.ddstate")
@patch("dronmakr.apps.generatr_plugins._resolve_drone_selection")
@patch("dronmakr.apps.generatr_plugins._plugin_path_exists", return_value=True)
def test_save_drone_preset_overwrites_existing_name(
    _mock_exists_path,
    mock_resolve,
    _mock_persist,
    _mock_name_exists,
    mock_load,
    mock_save,
    mock_delete,
):
    mock_resolve.return_value = ("/plugins/Synth.vst3", "Synth", "/tmp/old.ddstate", "Synth")

    result = gp.save_drone_preset(
        role="instrument",
        name="My Patch",
        instrument_selection={"kind": "plugin", "pluginPath": "/plugins/Synth.vst3"},
        overwrite=True,
    )

    mock_delete.assert_called_once_with("My Patch")
    mock_save.assert_called_once()
    assert result["name"] == "My Patch"
