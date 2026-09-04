"""Tests for comma-separated path parsing (Windows backslashes, escaped commas)."""

from __future__ import annotations

import unittest

from dronmakr.core.settings import parse_escaped_csv


class ParseEscapedCsvTest(unittest.TestCase):
    def test_windows_path_single_backslashes(self) -> None:
        self.assertEqual(
            parse_escaped_csv(r"F:\Samples\Kicks"),
            [r"F:\Samples\Kicks"],
        )

    def test_windows_path_double_backslashes(self) -> None:
        self.assertEqual(
            parse_escaped_csv(r"F:\\Samples\\Kicks"),
            [r"F:\Samples\Kicks"],
        )

    def test_multiple_windows_paths(self) -> None:
        self.assertEqual(
            parse_escaped_csv(r"F:\Samples\Kicks,D:\Drums\Snares"),
            [r"F:\Samples\Kicks", r"D:\Drums\Snares"],
        )

    def test_escaped_comma_in_path(self) -> None:
        self.assertEqual(
            parse_escaped_csv(r"C:\path1\,withcomma,C:\path2"),
            [r"C:\path1,withcomma", r"C:\path2"],
        )

    def test_unix_paths_unchanged(self) -> None:
        self.assertEqual(
            parse_escaped_csv("/home/user/samples,/tmp/drums"),
            ["/home/user/samples", "/tmp/drums"],
        )


if __name__ == "__main__":
    unittest.main()
