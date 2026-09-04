"""Tests for subprocess-based folder picker (must not invoke Tkinter from Flask thread)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import dronmakr.core.native_folder_picker as nfp


class NativeFolderPickerTest(unittest.TestCase):
    def test_darwin_uses_osascript(self) -> None:
        fake = MagicMock(return_value=MagicMock(returncode=0, stdout="/tmp/picked/\n"))

        with patch.object(nfp.sys, "platform", "darwin"):
            with patch("dronmakr.core.native_folder_picker.subprocess.run", fake):
                r = nfp.pick_folder_subprocess()

        self.assertEqual(r.status, "ok")
        self.assertEqual(fake.call_args[0][0][0], "osascript")
        self.assertIn("/tmp/picked", r.path)

    def test_darwin_passes_initial_dir_to_osascript(self) -> None:
        fake = MagicMock(return_value=MagicMock(returncode=0, stdout="/tmp/picked/\n"))

        with patch.object(nfp.sys, "platform", "darwin"):
            with patch("dronmakr.core.native_folder_picker.subprocess.run", fake):
                with patch("dronmakr.core.native_folder_picker.os.path.isdir", return_value=True):
                    nfp.pick_folder_subprocess("/tmp/picked")

        script = fake.call_args[0][0][2]
        self.assertIn("default location", script)
        self.assertIn("/tmp/picked", script)

    def test_linux_first_zenity_cancel_returns_cancelled(self) -> None:
        fake = MagicMock(return_value=MagicMock(returncode=1, stdout=""))

        with patch.object(nfp.sys, "platform", "linux"):
            with patch("dronmakr.core.native_folder_picker.subprocess.run", fake):
                r = nfp.pick_folder_subprocess()

        self.assertEqual(r.status, "cancelled")
        zenity_argv = fake.call_args[0][0]
        self.assertEqual(zenity_argv[0], "zenity")

    def test_win32_passes_initial_dir_via_env(self) -> None:
        fake = MagicMock(return_value=MagicMock(returncode=0, stdout=r"F:\Samples\Kicks\n"))

        with patch.object(nfp.sys, "platform", "win32"):
            with patch("dronmakr.core.native_folder_picker.subprocess.run", fake) as run_mock:
                with patch("dronmakr.core.native_folder_picker.os.path.isdir", return_value=True):
                    r = nfp.pick_folder_subprocess(r"F:\Samples\Kicks")

        self.assertEqual(r.status, "ok")
        env = run_mock.call_args.kwargs.get("env") or run_mock.call_args[1].get("env")
        self.assertEqual(env.get("DRONMAKR_PICKER_INITIAL"), r"F:\Samples\Kicks")


if __name__ == "__main__":
    unittest.main()
