"""Test-mode helpers for local E2E runs (DRONMAKR_TEST=1)."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

_TEST_ENV = "DRONMAKR_TEST"
_TEST_ROOT_ENV = "DRONMAKR_TEST_ROOT"
_TEST_PLUGIN_PATHS_ENV = "DRONMAKR_TEST_PLUGIN_PATHS"
_TEST_FILES_ROOT_ENV = "DRONMAKR_TEST_FILES_ROOT"
_LOGGER = logging.getLogger("dronmakr.test_mode")


def is_test_mode() -> bool:
    return os.environ.get(_TEST_ENV) == "1"


def activate_test_mode() -> Path:
    """
    Ensure isolated temp dirs and verbose logging when DRONMAKR_TEST=1.

    Safe to call multiple times; returns the active test root.
    """
    if not is_test_mode():
        return Path(".")

    root = os.environ.get(_TEST_ROOT_ENV)
    if not root:
        root = tempfile.mkdtemp(prefix="dronmakr-test-")
        os.environ[_TEST_ROOT_ENV] = root

    test_root = Path(root)
    test_root.mkdir(parents=True, exist_ok=True)
    (test_root / "config").mkdir(parents=True, exist_ok=True)
    (test_root / "logs").mkdir(parents=True, exist_ok=True)

    files_root = os.environ.get(_TEST_FILES_ROOT_ENV)
    if not files_root:
        files_root = str(test_root / "files")
        os.environ[_TEST_FILES_ROOT_ENV] = files_root

    logging.basicConfig(level=logging.DEBUG, force=False)
    _LOGGER.setLevel(logging.DEBUG)
    _LOGGER.debug("test mode active root=%s files_root=%s", test_root, files_root)
    return test_root


def test_data_root() -> Path | None:
    if not is_test_mode():
        return None
    activate_test_mode()
    return Path(os.environ[_TEST_ROOT_ENV])


def test_files_root() -> str | None:
    if not is_test_mode():
        return None
    activate_test_mode()
    return os.environ.get(_TEST_FILES_ROOT_ENV)


def test_plugin_paths_csv() -> str | None:
    if not is_test_mode():
        return None
    explicit = os.environ.get(_TEST_PLUGIN_PATHS_ENV, "").strip()
    if explicit:
        return explicit
    from dronmakr.presets.plugin_default_paths import default_plugin_paths_csv

    return default_plugin_paths_csv()


def get_test_mode_info() -> dict:
    if not is_test_mode():
        return {"testMode": False}
    root = activate_test_mode()
    from dronmakr.core.settings import SETTINGS_PATH, get_files_root, load_settings

    settings = load_settings()
    return {
        "testMode": True,
        "testRoot": str(root),
        "settingsPath": SETTINGS_PATH,
        "filesRoot": get_files_root(settings=settings, allow_default=False),
        "pluginPaths": settings.get("PLUGIN_PATHS", ""),
    }


def bootstrap_test_settings_if_needed() -> None:
    """Pre-seed PLUGIN_PATHS in test mode; leave FILES_ROOT for onboarding UI."""
    if not is_test_mode():
        return
    from dronmakr.core.settings import FILES_ROOT_KEY, load_settings, save_settings

    settings = load_settings()
    plugin_paths = test_plugin_paths_csv()
    if plugin_paths and not (settings.get("PLUGIN_PATHS") or "").strip():
        settings["PLUGIN_PATHS"] = plugin_paths
        save_settings(settings)
        _LOGGER.debug("seeded PLUGIN_PATHS for test mode")

    if (os.environ.get("DRONMAKR_TEST_AUTO_FILES_ROOT") or "").strip() == "1":
        files_root = test_files_root()
        if files_root and not (settings.get(FILES_ROOT_KEY) or "").strip():
            from dronmakr.core.settings import set_files_root

            set_files_root(files_root)
            _LOGGER.debug("auto-configured FILES_ROOT=%s", files_root)
