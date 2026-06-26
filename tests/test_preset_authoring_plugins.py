from unittest.mock import patch

from dronmakr.presets import preset_authoring as pa


@patch("dronmakr.presets.preset_authoring.os.path.exists", return_value=True)
@patch("dronmakr.presets.preset_authoring.list_installed_plugins")
@patch("dronmakr.presets.preset_authoring.plugin_settings_tuple")
def test_list_all_installed_plugin_entries_scans_paths_only(
    mock_settings_tuple,
    mock_list_installed,
    _mock_exists,
):
    mock_settings_tuple.return_value = (["/plugins"], [])
    mock_list_installed.return_value = [
        "/plugins/Kontakt 8.vst3",
        "/plugins/Ozone 11.vst3",
    ]

    entries = pa.list_all_installed_plugin_entries()

    assert len(entries) == 2
    labels = {entry["label"] for entry in entries}
    assert "Kontakt 8" in labels
    assert "Ozone 11" in labels
    assert all(entry.get("displayLabel") for entry in entries)


@patch("dronmakr.presets.preset_authoring.get_setting")
@patch("dronmakr.presets.preset_authoring.list_all_installed_plugin_entries")
def test_list_installed_plugin_entries_empty_allowlist_shows_all(
    mock_list_all,
    mock_get_setting,
):
    mock_list_all.return_value = [
        {"label": "Kontakt 8", "path": "/plugins/Kontakt 8.vst3", "displayLabel": "Kontakt 8"},
        {"label": "Ozone 11", "path": "/plugins/Ozone 11.vst3", "displayLabel": "Ozone 11"},
    ]
    mock_get_setting.return_value = ""

    entries = pa.list_installed_plugin_entries(role="instrument")

    assert len(entries) == 2


@patch("dronmakr.presets.preset_authoring.get_setting")
@patch("dronmakr.presets.preset_authoring.list_all_installed_plugin_entries")
def test_list_installed_plugin_entries_filters_by_allowlist(
    mock_list_all,
    mock_get_setting,
):
    mock_list_all.return_value = [
        {"label": "Kontakt 8", "path": "/plugins/Kontakt 8.vst3", "displayLabel": "Kontakt 8"},
        {"label": "Ozone 11", "path": "/plugins/Ozone 11.vst3", "displayLabel": "Ozone 11"},
    ]
    mock_get_setting.return_value = "Kontakt 8"

    entries = pa.list_installed_plugin_entries(role="instrument")

    assert [entry["label"] for entry in entries] == ["Kontakt 8"]


@patch("dronmakr.core.settings.save_settings")
@patch("dronmakr.core.settings.load_settings")
def test_save_allowed_plugins_for_role_persists_allowlist(
    mock_load_settings,
    mock_save_settings,
):
    mock_load_settings.return_value = {
        "INSTRUMENT_ALLOW_LIST": "",
        "FX_ALLOW_LIST": "",
        "IGNORE_PLUGINS": "Kontakt,Ozone 11",
    }

    result = pa.save_allowed_plugins_for_role("instrument", ["Kontakt 8", "Reaktor 6"])

    assert result == "Kontakt 8,Reaktor 6"
    mock_save_settings.assert_called_once()
    saved = mock_save_settings.call_args[0][0]
    assert saved["INSTRUMENT_ALLOW_LIST"] == "Kontakt 8,Reaktor 6"
    assert saved["IGNORE_PLUGINS"] == ""


def test_assert_plugin_role_for_slot_is_noop():
    pa.assert_plugin_role_for_slot("/plugins/Anything.vst3", "instrument")
