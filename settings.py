"""
Central settings management. Uses config/settings.json instead of .env.
On startup, migrates from .env if settings.json does not exist.
"""

import json
import os

SETTINGS_PATH = "config/settings.json"
DEFAULT_KEYS = [
    "PLUGIN_PATHS",
    "QT_LOGGING_RULES",
    "IMKLogLevel",
    "ASSERT_INSTRUMENT",
    "IGNORE_PLUGINS",
    "CUSTOM_PLUGINS",
    "DRUM_KICK_PATHS",
    "DRUM_HIHAT_PATHS",
    "DRUM_PERC_PATHS",
    "DRUM_TOM_PATHS",
    "DRUM_SNARE_PATHS",
    "DRUM_SHAKER_PATHS",
    "DRUM_CLAP_PATHS",
    "DRUM_CYMBAL_PATHS",
]

_default_values = {
    "PLUGIN_PATHS": "",
    "QT_LOGGING_RULES": "*.debug=false",
    "IMKLogLevel": "none",
    "ASSERT_INSTRUMENT": "",
    "IGNORE_PLUGINS": "",
    "CUSTOM_PLUGINS": "",
    "DRUM_KICK_PATHS": "",
    "DRUM_HIHAT_PATHS": "",
    "DRUM_PERC_PATHS": "",
    "DRUM_TOM_PATHS": "",
    "DRUM_SNARE_PATHS": "",
    "DRUM_SHAKER_PATHS": "",
    "DRUM_CLAP_PATHS": "",
    "DRUM_CYMBAL_PATHS": "",
}


def _migrate_from_env() -> dict:
    """Load values from .env file if it exists."""
    result = dict(_default_values)
    env_path = ".env"
    if not os.path.exists(env_path):
        return result
    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key in DEFAULT_KEYS:
                        result[key] = value
    except Exception:
        pass
    return result


def _ensure_settings_file() -> None:
    """Ensure config/settings.json exists. Create from .env or defaults if missing."""
    if os.path.exists(SETTINGS_PATH):
        return
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    settings = _migrate_from_env()
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


def load_settings() -> dict:
    """Load settings from config/settings.json. Creates file from .env if missing."""
    _ensure_settings_file()
    with open(SETTINGS_PATH, "r") as f:
        data = json.load(f)
    # Merge with defaults so new keys get default values
    out = dict(_default_values)
    for k, v in data.items():
        if isinstance(v, str):
            out[k] = v
    return out


def get_setting(key: str, default: str = "") -> str:
    """Get a single setting value. Returns default if key missing or not a string."""
    settings = load_settings()
    return settings.get(key, default) if isinstance(settings.get(key), str) else default


def save_settings(settings: dict) -> None:
    """Save settings to config/settings.json."""
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


def ensure_settings() -> None:
    """Call at startup (CLI or webui) to ensure settings.json exists."""
    _ensure_settings_file()
