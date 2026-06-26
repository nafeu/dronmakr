import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dronmakr.core import utils


class TestManagedConfigBootstrap(unittest.TestCase):
    def test_creates_all_config_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            files_root = Path(tmp) / "dronmakr-files"
            files_root.mkdir()

            def managed_file(*parts: str) -> str:
                return str(files_root.joinpath(*parts))

            with patch(
                "dronmakr.core.settings.has_configured_files_root",
                return_value=True,
            ), patch(
                "dronmakr.core.paths.get_files_root_path",
                return_value=files_root,
            ), patch(
                "dronmakr.core.paths.get_managed_file",
                side_effect=managed_file,
            ), patch(
                "dronmakr.core.utils.get_managed_file",
                side_effect=managed_file,
            ):
                utils.refresh_managed_path_constants()
                utils.ensure_managed_config_files()

            config_dir = files_root / "config"
            presets = json.loads((config_dir / "presets.json").read_text(encoding="utf-8"))
            self.assertEqual(presets, [])
            self.assertTrue((config_dir / "post-processing-shortcuts.json").is_file())
            self.assertTrue((config_dir / "beat-patterns.json").is_file())
            self.assertTrue((config_dir / "drum-kits.json").is_file())


if __name__ == "__main__":
    unittest.main()
