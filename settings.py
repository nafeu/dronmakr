"""
Central settings management. Uses config/settings.json instead of .env.
On startup, migrates from .env if settings.json does not exist.
"""

import json
import os

SETTINGS_PATH = "config/settings.json"
DRUM_PATH_KEYS = [
    "DRUM_KICK_PATHS",
    "DRUM_HIHAT_PATHS",
    "DRUM_PERC_PATHS",
    "DRUM_TOM_PATHS",
    "DRUM_SNARE_PATHS",
    "DRUM_SHAKER_PATHS",
    "DRUM_CLAP_PATHS",
    "DRUM_CYMBAL_PATHS",
]
DEFAULT_DRUM_PATH_PRESET_NAME = "Default"
DRUM_PATH_PRESET_NAME_KEY = "DRUM_PATH_PRESET"
DEFAULT_KEYS = [
    "PLUGIN_PATHS",
    "QT_LOGGING_RULES",
    "IMKLogLevel",
    "ASSERT_INSTRUMENT",
    "IGNORE_PLUGINS",
    "CUSTOM_PLUGINS",
    *DRUM_PATH_KEYS,
    "ACTIVE_DRUM_PATH_PRESET",
    "DRUM_PATH_PRESETS",
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
    "ACTIVE_DRUM_PATH_PRESET": DEFAULT_DRUM_PATH_PRESET_NAME,
    "DRUM_PATH_PRESETS": [
        {
            DRUM_PATH_PRESET_NAME_KEY: DEFAULT_DRUM_PATH_PRESET_NAME,
            **{key: "" for key in DRUM_PATH_KEYS},
        }
    ],
}


def _normalize_preset_name(name: str | None) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip()


def parse_escaped_csv(value: str | None) -> list[str]:
    """
    Split comma-separated values where commas can be escaped as '\,'.
    Also supports escaping backslash as '\\'.
    """
    if not isinstance(value, str) or not value:
        return []
    out: list[str] = []
    buf: list[str] = []
    escaped = False
    for ch in value:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == ",":
            item = "".join(buf).strip()
            if item:
                out.append(item)
            buf = []
            continue
        buf.append(ch)
    if escaped:
        buf.append("\\")
    item = "".join(buf).strip()
    if item:
        out.append(item)
    return out


def _coerce_preset_paths(value: dict | None) -> dict[str, str]:
    out: dict[str, str] = {}
    data = value if isinstance(value, dict) else {}
    for key in DRUM_PATH_KEYS:
        v = data.get(key, "")
        out[key] = v if isinstance(v, str) else ""
    return out


def _coerce_preset_entry(value: dict | None) -> dict[str, str]:
    data = value if isinstance(value, dict) else {}
    name = _normalize_preset_name(data.get(DRUM_PATH_PRESET_NAME_KEY))
    entry: dict[str, str] = {DRUM_PATH_PRESET_NAME_KEY: name}
    entry.update(_coerce_preset_paths(data))
    return entry


def _default_preset_paths_from_flat(data: dict | None) -> dict[str, str]:
    src = data if isinstance(data, dict) else {}
    out: dict[str, str] = {}
    for key in DRUM_PATH_KEYS:
        v = src.get(key, "")
        out[key] = v if isinstance(v, str) else ""
    return out


def _normalize_drum_presets_inplace(settings: dict) -> None:
    raw_presets = settings.get("DRUM_PATH_PRESETS")
    normalized_entries: list[dict[str, str]] = []
    seen_names: set[str] = set()

    if isinstance(raw_presets, list):
        for raw_entry in raw_presets:
            entry = _coerce_preset_entry(raw_entry)
            name = entry.get(DRUM_PATH_PRESET_NAME_KEY, "")
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            normalized_entries.append(entry)
    elif isinstance(raw_presets, dict):
        # Backward compatibility: old map style {"PresetName": {"DRUM_*": "..."}}
        for raw_name, raw_paths in raw_presets.items():
            name = _normalize_preset_name(raw_name)
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            entry = {DRUM_PATH_PRESET_NAME_KEY: name}
            entry.update(_coerce_preset_paths(raw_paths))
            normalized_entries.append(entry)

    # If no preset payload exists, migrate from legacy flat DRUM_* keys.
    if not normalized_entries:
        default_entry = {DRUM_PATH_PRESET_NAME_KEY: DEFAULT_DRUM_PATH_PRESET_NAME}
        default_entry.update(_default_preset_paths_from_flat(settings))
        normalized_entries.append(default_entry)

    if DEFAULT_DRUM_PATH_PRESET_NAME not in seen_names:
        default_entry = {DRUM_PATH_PRESET_NAME_KEY: DEFAULT_DRUM_PATH_PRESET_NAME}
        default_entry.update({key: "" for key in DRUM_PATH_KEYS})
        normalized_entries.insert(0, default_entry)
        seen_names.add(DEFAULT_DRUM_PATH_PRESET_NAME)

    active = _normalize_preset_name(settings.get("ACTIVE_DRUM_PATH_PRESET"))
    if not active or active not in seen_names:
        active = DEFAULT_DRUM_PATH_PRESET_NAME

    settings["DRUM_PATH_PRESETS"] = normalized_entries
    settings["ACTIVE_DRUM_PATH_PRESET"] = active

    # Keep legacy flat keys aligned to active preset for backward compatibility.
    active_paths = {}
    for entry in normalized_entries:
        if entry.get(DRUM_PATH_PRESET_NAME_KEY) == active:
            active_paths = _coerce_preset_paths(entry)
            break
    for key in DRUM_PATH_KEYS:
        settings[key] = active_paths.get(key, "")


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
        if isinstance(v, str) or (
            k == "DRUM_PATH_PRESETS" and isinstance(v, (dict, list))
        ):
            out[k] = v
    _normalize_drum_presets_inplace(out)
    return out


def get_setting(key: str, default: str = "") -> str:
    """Get a single setting value. Returns default if key missing or not a string."""
    settings = load_settings()
    if key in DRUM_PATH_KEYS:
        active = get_active_drum_path_preset_name(settings)
        presets = get_drum_path_presets(settings)
        active_paths = presets.get(active, {})
        val = active_paths.get(key, default)
        return val if isinstance(val, str) else default
    return settings.get(key, default) if isinstance(settings.get(key), str) else default


def save_settings(settings: dict) -> None:
    """Save settings to config/settings.json."""
    _normalize_drum_presets_inplace(settings)
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


def ensure_settings() -> None:
    """Call at startup (CLI or webui) to ensure settings.json exists."""
    _ensure_settings_file()


def get_drum_path_presets(settings: dict | None = None) -> dict[str, dict[str, str]]:
    src = settings if isinstance(settings, dict) else load_settings()
    _normalize_drum_presets_inplace(src)
    entries = src.get("DRUM_PATH_PRESETS", [])
    out: dict[str, dict[str, str]] = {}
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = _normalize_preset_name(entry.get(DRUM_PATH_PRESET_NAME_KEY))
            if not name:
                continue
            out[name] = _coerce_preset_paths(entry)
    return out


def get_active_drum_path_preset_name(settings: dict | None = None) -> str:
    src = settings if isinstance(settings, dict) else load_settings()
    _normalize_drum_presets_inplace(src)
    active = src.get("ACTIVE_DRUM_PATH_PRESET", DEFAULT_DRUM_PATH_PRESET_NAME)
    if isinstance(active, str) and active in get_drum_path_presets(src):
        return active
    return DEFAULT_DRUM_PATH_PRESET_NAME


def set_active_drum_path_preset(name: str) -> tuple[bool, str]:
    settings = load_settings()
    presets = get_drum_path_presets(settings)
    normalized_name = _normalize_preset_name(name)
    if not normalized_name:
        return False, "Preset name is required"
    if normalized_name not in presets:
        return False, f'Preset "{normalized_name}" does not exist'
    settings["ACTIVE_DRUM_PATH_PRESET"] = normalized_name
    save_settings(settings)
    return True, normalized_name


def get_drum_path_preset_entries(settings: dict | None = None) -> list[dict[str, str]]:
    src = settings if isinstance(settings, dict) else load_settings()
    _normalize_drum_presets_inplace(src)
    entries = src.get("DRUM_PATH_PRESETS", [])
    out: list[dict[str, str]] = []
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            coerced = _coerce_preset_entry(entry)
            if coerced.get(DRUM_PATH_PRESET_NAME_KEY):
                out.append(coerced)
    return out


def get_all_drum_paths_for_key(key: str, settings: dict | None = None) -> list[str]:
    """Return parsed path roots for `key` from the active drum path preset only."""
    if key not in DRUM_PATH_KEYS:
        return []
    src = settings if isinstance(settings, dict) else load_settings()
    _normalize_drum_presets_inplace(src)
    active = get_active_drum_path_preset_name(src)
    presets = get_drum_path_presets(src)
    entry = presets.get(active, {})
    value = entry.get(key, "")
    if not isinstance(value, str) or not value.strip():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in parse_escaped_csv(value):
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
