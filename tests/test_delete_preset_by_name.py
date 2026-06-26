from unittest.mock import patch

from dronmakr.presets import preset_authoring as pa


@patch("dronmakr.presets.preset_authoring.save_presets_json")
@patch("dronmakr.presets.preset_authoring.load_presets_json")
def test_delete_preset_by_name(mock_load, mock_save):
    mock_load.return_value = [
        {"id": "abc", "name": "My Patch", "type": "instrument", "preset_path": "/tmp/x.ddstate"},
        {"id": "def", "name": "Other", "type": "effect", "preset_path": "/tmp/y.ddstate"},
    ]

    deleted = pa.delete_preset_by_name("My Patch")

    assert deleted is True
    mock_save.assert_called_once()
    saved = mock_save.call_args[0][0]
    assert len(saved) == 1
    assert saved[0]["name"] == "Other"
