"""
Central settings management. Uses config/settings.json instead of .env.
On startup, migrates from .env if settings.json does not exist.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dronmakr._repo import LOGS_ROOT, REPO_ROOT


def _user_data_root() -> Path:
    """Writable location for bundled (PyInstaller) app — cwd is unreliable."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "dronmakr"
    if sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "dronmakr"
    return Path.home() / ".local" / "share" / "dronmakr"


def get_server_logs_dir() -> Path:
    """
    Directory for server-process log files.

    Packaged builds use the same user-data root as ``config/settings.json``; development
    runs use ``./logs`` at the repo root.
    """
    if getattr(sys, "frozen", False):
        return _user_data_root() / "logs"
    return LOGS_ROOT


def _compute_settings_path() -> str:
    """
    Resolve config/settings.json relative to this package, not process cwd.

    CLI and web must read the same file; Flask/desktop are often launched with a
    cwd that is not the repository root (empty PLUGIN_PATHS in the UI otherwise).
    """
    if getattr(sys, "frozen", False):
        cfg = _user_data_root() / "config"
    else:
        cfg = REPO_ROOT / "config"
    return str(cfg / "settings.json")


SETTINGS_PATH = _compute_settings_path()


def _settings_env_path() -> Path:
    """Repository .env used for first-run migration."""
    return REPO_ROOT / ".env"


def _settings_env_example_path() -> Path:
    """Repository .env template."""
    return REPO_ROOT / ".env-sample"
DEFAULT_FILES_ROOT_DIRNAME = "dronmakr-files"
FILES_ROOT_KEY = "FILES_ROOT"
MANAGED_SUBDIRS = [
    "presets",
    "exports",
    "archive",
    "saved",
    "recordings",
    "splits",
    "trash",
    "packages",
    "history",
    "temp",
    "vst-preset-files",
    "config",
]
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
DEFAULT_DRUM_PATH_PRESET_NAME = "default"
FOLYSPLITR_DRUM_PATH_PRESET_NAME = "folysplitr"
DRUM_PATH_PRESET_NAME_KEY = "DRUM_PATH_PRESET"
# Folysplitr exports splits into category folders under FILES_ROOT/splits/.
DRUM_PATH_KEY_TO_SPLIT_CATEGORY = {
    "DRUM_KICK_PATHS": "kick",
    "DRUM_HIHAT_PATHS": "hihat",
    "DRUM_PERC_PATHS": "perc",
    "DRUM_TOM_PATHS": "tom",
    "DRUM_SNARE_PATHS": "snare",
    "DRUM_SHAKER_PATHS": "shaker",
    "DRUM_CLAP_PATHS": "clap",
    "DRUM_CYMBAL_PATHS": "cymbal",
}
DEFAULT_KEYS = [
    "PLUGIN_PATHS",
    "IGNORE_PLUGINS",
    "INSTRUMENT_ALLOW_LIST",
    "FX_ALLOW_LIST",
    *DRUM_PATH_KEYS,
    "ACTIVE_DRUM_PATH_PRESET",
    "DRUM_PATH_PRESETS",
    FILES_ROOT_KEY,
]

_default_values = {
    "PLUGIN_PATHS": "",
    "IGNORE_PLUGINS": "",
    "INSTRUMENT_ALLOW_LIST": "",
    "FX_ALLOW_LIST": "",
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
    FILES_ROOT_KEY: "",
}


def _normalize_preset_name(name: str | None) -> str:
    if not isinstance(name, str):
        return ""
    normalized = name.strip()
    if normalized.lower() == DEFAULT_DRUM_PATH_PRESET_NAME:
        return DEFAULT_DRUM_PATH_PRESET_NAME
    return normalized


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


def _migrate_legacy_settings_keys(settings: dict) -> None:
    """Copy values from deprecated keys into current allow-list settings."""
    legacy_allow_lists = (
        ("ALLOWED_INSTRUMENT_PLUGINS", "INSTRUMENT_ALLOW_LIST"),
        ("ALLOWED_FX_PLUGINS", "FX_ALLOW_LIST"),
    )
    for old_key, new_key in legacy_allow_lists:
        old_val = settings.get(old_key, "")
        new_val = settings.get(new_key, "")
        if isinstance(old_val, str) and old_val.strip() and not (
            isinstance(new_val, str) and new_val.strip()
        ):
            settings[new_key] = old_val


def _migrate_from_env() -> dict:
    """Load values from .env file if it exists."""
    result = dict(_default_values)
    env_path = str(_settings_env_path())
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


def _default_files_root() -> str:
    return os.path.abspath(os.path.join(os.path.expanduser("~"), DEFAULT_FILES_ROOT_DIRNAME))


def normalize_files_root(path: str | None) -> str:
    if not isinstance(path, str):
        return ""
    cleaned = path.strip()
    if not cleaned:
        return ""
    return os.path.abspath(os.path.expanduser(cleaned))


def get_files_root(settings: dict | None = None, allow_default: bool = False) -> str:
    src = settings if isinstance(settings, dict) else load_settings()
    normalized = normalize_files_root(src.get(FILES_ROOT_KEY, ""))
    if normalized:
        return normalized
    return _default_files_root() if allow_default else ""


def has_configured_files_root(settings: dict | None = None) -> bool:
    return bool(get_files_root(settings=settings, allow_default=False))


def has_configured_drum_paths(settings: dict | None = None) -> bool:
    """True when the active drum path preset has at least one non-empty path list."""
    src = settings if isinstance(settings, dict) else load_settings()
    active = get_active_drum_path_preset_name(src)
    presets = get_drum_path_presets(src)
    entry = presets.get(active, {})
    for key in DRUM_PATH_KEYS:
        value = entry.get(key, "")
        if isinstance(value, str) and value.strip() and parse_escaped_csv(value):
            return True
    return False


def has_configured_plugin_paths(settings: dict | None = None) -> bool:
    """True when PLUGIN_PATHS are set and at least one plug-in is discovered."""
    import glob
    import sys

    src = settings if isinstance(settings, dict) else load_settings()
    plugin_paths_raw = src.get("PLUGIN_PATHS", "")
    plugin_paths = (
        [p.strip() for p in plugin_paths_raw.split(",") if p.strip()]
        if isinstance(plugin_paths_raw, str)
        else []
    )
    if not plugin_paths:
        return False
    discovered: list[str] = []
    for plugin_dir in plugin_paths:
        if not plugin_dir or not os.path.exists(plugin_dir):
            continue
        discovered.extend(glob.glob(os.path.join(plugin_dir, "*.vst3")))
        if sys.platform != "darwin":
            discovered.extend(glob.glob(os.path.join(plugin_dir, "*.vst")))
        discovered.extend(glob.glob(os.path.join(plugin_dir, "*.dll")))
        discovered.extend(glob.glob(os.path.join(plugin_dir, "*.so")))
        discovered.extend(glob.glob(os.path.join(plugin_dir, "*.component")))
    return bool(discovered)


def ensure_managed_files_root(root: str | None = None) -> str:
    resolved_root = normalize_files_root(root) if root is not None else get_files_root(allow_default=True)
    if not resolved_root:
        raise ValueError("FILES_ROOT is not configured")
    os.makedirs(resolved_root, exist_ok=True)
    for subdir in MANAGED_SUBDIRS:
        os.makedirs(os.path.join(resolved_root, subdir), exist_ok=True)
    return resolved_root


def build_folysplitr_drum_path_preset(files_root: str) -> dict[str, str]:
    """Return DRUM_* path values for the folysplitr preset under files_root/splits/."""
    resolved_root = normalize_files_root(files_root)
    if not resolved_root:
        raise ValueError("A valid files root path is required")
    splits_root = os.path.join(resolved_root, "splits")
    return {
        key: os.path.abspath(os.path.join(splits_root, category))
        for key, category in DRUM_PATH_KEY_TO_SPLIT_CATEGORY.items()
    }


def _apply_folysplitr_drum_path_preset(settings: dict, files_root: str) -> None:
    """Ensure splits category dirs and a folysplitr drum path preset exist for files_root."""
    paths = build_folysplitr_drum_path_preset(files_root)
    splits_root = os.path.join(normalize_files_root(files_root), "splits")
    os.makedirs(splits_root, exist_ok=True)
    for category in DRUM_PATH_KEY_TO_SPLIT_CATEGORY.values():
        os.makedirs(os.path.join(splits_root, category), exist_ok=True)

    entries = get_drum_path_preset_entries(settings)
    new_entry = {
        DRUM_PATH_PRESET_NAME_KEY: FOLYSPLITR_DRUM_PATH_PRESET_NAME,
        **paths,
    }
    updated = False
    for index, entry in enumerate(entries):
        if entry.get(DRUM_PATH_PRESET_NAME_KEY) == FOLYSPLITR_DRUM_PATH_PRESET_NAME:
            entries[index] = new_entry
            updated = True
            break
    if not updated:
        entries.append(new_entry)
    settings["DRUM_PATH_PRESETS"] = entries


def ensure_folysplitr_drum_path_preset(files_root: str | None = None) -> dict[str, str]:
    """
    Ensure the folysplitr drum path preset exists and points at files_root/splits/*.
    Returns the preset's DRUM_* path map.
    """
    resolved = (
        normalize_files_root(files_root)
        if files_root is not None
        else get_files_root(allow_default=False)
    )
    if not resolved:
        raise ValueError("FILES_ROOT is not configured")
    paths = build_folysplitr_drum_path_preset(resolved)
    settings = load_settings()
    existing = get_drum_path_presets(settings).get(FOLYSPLITR_DRUM_PATH_PRESET_NAME, {})
    if existing == paths:
        return paths
    _apply_folysplitr_drum_path_preset(settings, resolved)
    save_settings(settings)
    return paths


def set_files_root(path: str) -> str:
    resolved = normalize_files_root(path)
    if not resolved:
        raise ValueError("A valid files root path is required")
    ensure_managed_files_root(resolved)
    settings = load_settings()
    settings[FILES_ROOT_KEY] = resolved
    _apply_folysplitr_drum_path_preset(settings, resolved)
    save_settings(settings)
    try:
        from dronmakr.core.utils import refresh_managed_path_constants

        refresh_managed_path_constants()
    except Exception:
        pass
    return resolved


def _ensure_settings_file() -> None:
    """Ensure config/settings.json exists. Create from .env or defaults if missing."""
    if os.path.exists(SETTINGS_PATH):
        return
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    settings = _migrate_from_env()
    # First-ever file only: sensible HOST OS plugin folders when nothing came from .env.
    pp = settings.get("PLUGIN_PATHS")
    if not (isinstance(pp, str) and pp.strip()):
        from dronmakr.presets.plugin_default_paths import default_plugin_paths_csv

        settings["PLUGIN_PATHS"] = default_plugin_paths_csv()
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
        if k == FILES_ROOT_KEY and isinstance(v, str):
            out[k] = normalize_files_root(v)
        elif isinstance(v, str) or (
            k == "DRUM_PATH_PRESETS" and isinstance(v, (dict, list))
        ):
            out[k] = v
    _normalize_drum_presets_inplace(out)
    _migrate_legacy_settings_keys(out)
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
    settings[FILES_ROOT_KEY] = normalize_files_root(settings.get(FILES_ROOT_KEY, ""))
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
