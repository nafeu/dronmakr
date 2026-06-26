import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dronmakr.core import utils
from dronmakr.core.settings import get_files_root, has_configured_files_root, set_files_root


class TestFilesRootOnboarding(unittest.TestCase):
    def test_no_default_home_directory_without_settings(self):
        with patch("dronmakr.core.settings.load_settings", return_value={"FILES_ROOT": ""}):
            self.assertFalse(has_configured_files_root())
            self.assertEqual(get_files_root(allow_default=False), "")

    def test_get_managed_dir_empty_when_unconfigured(self):
        with patch("dronmakr.core.settings.load_settings", return_value={"FILES_ROOT": ""}):
            utils.refresh_managed_path_constants()
            self.assertEqual(utils.EXPORTS_DIR, "")

    def test_set_files_root_creates_user_chosen_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "example-storage"
            with patch("dronmakr.core.settings.SETTINGS_PATH", str(Path(tmp) / "settings.json")):
                with patch("dronmakr.core.settings.load_settings", return_value={"FILES_ROOT": ""}):
                    with patch("dronmakr.core.settings.save_settings") as save_mock:
                        resolved = set_files_root(str(storage))
            self.assertEqual(resolved, os.path.abspath(str(storage)))
            self.assertTrue(storage.is_dir())
            self.assertTrue((storage / "exports").is_dir())
            self.assertTrue((storage / "config").is_dir())
            save_mock.assert_called_once()

    def test_home_dronmakr_files_not_created_on_path_refresh(self):
        home_default = Path.home() / "dronmakr-files"
        existed_before = home_default.exists()
        with patch("dronmakr.core.settings.load_settings", return_value={"FILES_ROOT": ""}):
            utils.refresh_managed_path_constants()
        if not existed_before:
            self.assertFalse(home_default.exists())


if __name__ == "__main__":
    unittest.main()
