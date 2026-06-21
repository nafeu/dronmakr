"""Generate Samples drone plug-in picker and editor capture."""

from __future__ import annotations

import json
import os

from dronmakr.audio.audio_host import (
    create_engine,
    load_plugin as load_plugin_on_engine,
    open_plugin_editor,
    plugin_is_effect,
    plugin_is_instrument,
    save_plugin_state,
)
from dronmakr.core.utils import TEMP_DIR, generate_id, resolve_presets_index_path
from dronmakr.presets.preset_authoring import (
    format_plugin_name,
    list_installed_plugin_entries,
    plugin_settings_tuple,
    save_allowed_plugins_for_role,
    scan_plugin_classifications,
)


def generatr_session_dir() -> str:
    path = os.path.join(TEMP_DIR, "generatr-plugin-sessions")
    os.makedirs(path, exist_ok=True)
    return path


def _ensure_plugin_classifications_scanned(*, force: bool = False) -> None:
    from dronmakr.audio.audio_worker import delegate_scan_plugin_classifications_if_needed

    delegate_scan_plugin_classifications_if_needed(force=force)


def _load_presets_list() -> list[dict]:
    presets_path = resolve_presets_index_path()
    if not presets_path or not os.path.isfile(presets_path):
        return []
    with open(presets_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return [p for p in data if isinstance(p, dict)]


def _patch_summary(preset: dict) -> dict:
    return {
        "name": preset.get("name") or "",
        "type": preset.get("type") or "",
        "pluginName": preset.get("plugin_name") or "",
        "pluginPath": preset.get("plugin_path") or "",
        "presetPath": preset.get("preset_path") or "",
        "effects": preset.get("effects") or [],
    }


def _plugin_path_exists(plugin_path: str) -> bool:
    """True if a plug-in bundle or binary exists (VST3/AU bundles are directories on macOS)."""
    path = (plugin_path or "").strip()
    return bool(path) and os.path.exists(path)


def _role_plugins(role: str, *, respect_ignore: bool) -> list[dict]:
    _ensure_plugin_classifications_scanned()
    return list_installed_plugin_entries(role=role, respect_ignore=respect_ignore)


def get_drone_picker_payload(role: str) -> dict:
    """Return saved patches and installed plug-ins for instrument or FX picker."""
    role = (role or "instrument").strip().lower()
    presets = _load_presets_list()
    if role == "instrument":
        patches = [p for p in presets if p.get("type") == "instrument"]
    else:
        patches = [p for p in presets if p.get("type") in ("effect", "effect_chain")]

    plugin_paths, _, _, custom_plugins = plugin_settings_tuple()
    plugins: list[dict] = []
    if plugin_paths and plugin_paths != [""]:
        plugins = [
            {"label": entry["label"], "path": entry["path"]}
            for entry in _role_plugins(role, respect_ignore=True)
            if _plugin_path_exists(entry["path"])
        ]

    return {
        "role": role,
        "patches": [_patch_summary(p) for p in patches if p.get("name")],
        "plugins": plugins,
        "pluginPathsConfigured": bool(plugins) or bool(list_installed_plugin_entries(respect_ignore=False)),
    }


def get_drone_plugin_list_editor_payload(role: str) -> dict:
    """Return allowed vs detected plug-ins for the list editor modal."""
    role = (role or "instrument").strip().lower()
    detected = _role_plugins(role, respect_ignore=False)
    allowed = _role_plugins(role, respect_ignore=True)
    return {
        "role": role,
        "allowed": [{"label": entry["label"], "path": entry["path"]} for entry in allowed],
        "detected": [{"label": entry["label"], "path": entry["path"]} for entry in detected],
    }


def save_drone_plugin_list_editor(role: str, allowed_labels: list[str]) -> dict:
    """Persist allowed plug-ins for a role via ``IGNORE_PLUGINS``."""
    role = (role or "").strip().lower()
    ignore_plugins = save_allowed_plugins_for_role(role, allowed_labels)
    scan_plugin_classifications(force=True)
    return {
        "role": role,
        "ignorePlugins": ignore_plugins,
        **get_drone_plugin_list_editor_payload(role),
    }


def open_drone_plugin_editor_capture(plugin_path: str, role: str) -> dict:
    """Open DawDreamer plug-in editor; save state snapshot when the window closes."""
    plugin_path = os.path.abspath((plugin_path or "").strip())
    role = (role or "instrument").strip().lower()
    if not _plugin_path_exists(plugin_path):
        raise FileNotFoundError(f"Plug-in not found: {plugin_path}")

    engine = create_engine()
    processor = load_plugin_on_engine(engine, plugin_path)

    if role == "instrument" and not plugin_is_instrument(processor):
        raise ValueError(
            "Selected plug-in is not an instrument (it accepts audio input). "
            "Choose a synth or instrument plug-in."
        )
    if role == "effect" and not plugin_is_effect(processor):
        raise ValueError(
            "Selected plug-in is an instrument (no audio input). Choose an FX plug-in."
        )

    open_plugin_editor(processor)

    preset_path = os.path.join(generatr_session_dir(), f"{generate_id()}.ddstate")
    save_plugin_state(processor, preset_path)

    return {
        "kind": "plugin",
        "pluginPath": plugin_path,
        "presetPath": preset_path,
        "label": format_plugin_name(plugin_path),
    }
