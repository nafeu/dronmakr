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

    def test_linux_first_zenity_cancel_returns_cancelled(self) -> None:
        fake = MagicMock(return_value=MagicMock(returncode=1, stdout=""))

        with patch.object(nfp.sys, "platform", "linux"):
            with patch("dronmakr.core.native_folder_picker.subprocess.run", fake):
                r = nfp.pick_folder_subprocess()

        self.assertEqual(r.status, "cancelled")
        zenity_argv = fake.call_args[0][0]
        self.assertEqual(zenity_argv[0], "zenity")


if __name__ == "__main__":
    unittest.main()
