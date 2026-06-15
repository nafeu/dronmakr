"""DawDreamer preset authoring shared by Patchcraftr GUI and CLI helpers."""

from __future__ import annotations

import glob
import json
import os
import sys

import audio_host
from audio_host import (
    apply_plugin_state,
    apply_vstpreset_bytes_to_plugin,
    create_engine,
    load_plugin as _load_plugin_on_engine,
    open_plugin_editor,
    plugin_is_effect,
    plugin_is_instrument,
    reload_plugin_preserving_state as _reload_plugin_on_engine,
    save_plugin_state,
    serialize_plugin_preset_bytes,
    write_plugin_state_to_vstpreset,
)
from generate_midi import SUPPORTED_PATTERNS_INFO
from paths import get_managed_file
from settings import get_setting
from utils import (
    MAGENTA,
    RESET,
    generate_id,
    PRESETS_DIR,
    resolve_presets_index_path,
    TEMP_DIR,
    with_patchcraftr_prompt as with_prompt,
)

MAX_CHAIN_SLOTS = 5


class PluginVariantRequired(Exception):
    """Reserved — DawDreamer does not expose multi-factory VST3 variant selection yet."""

    def __init__(self, plugin_path: str, variants: list[str]):
        self.plugin_path = plugin_path
        self.variants = variants
        super().__init__(f"Choose variant for {plugin_path!r}: {variants!r}")


class PresetAuthoringConfigError(RuntimeError):
    pass


def ensure_authoring_dirs() -> None:
    os.makedirs(PRESETS_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)


def plugin_settings_tuple() -> tuple[list[str], list[str], list[str], list[str]]:
    plugin_paths = get_setting("PLUGIN_PATHS", "").split(",")
    assert_instrument = [x.strip() for x in get_setting("ASSERT_INSTRUMENT", "").split(",") if x.strip()]
    ignore_plugins = [x.strip() for x in get_setting("IGNORE_PLUGINS", "").split(",") if x.strip()]
    custom_plugins = [
        plugin.strip()
        for plugin in get_setting("CUSTOM_PLUGINS", "").split(",")
        if plugin.strip()
    ]
    return plugin_paths, assert_instrument, ignore_plugins, custom_plugins


def assert_plugin_paths_configured() -> None:
    plugin_paths, _, _, custom_plugins = plugin_settings_tuple()
    if not plugin_paths or plugin_paths == [""]:
        raise PresetAuthoringConfigError(
            "No VST paths in settings (PLUGIN_PATHS). Set them in Settings."
        )
    discovered = list_installed_plugins(plugin_paths, custom_plugins)
    if not discovered:
        raise PresetAuthoringConfigError(
            "No plug-ins discovered under PLUGIN_PATHS / CUSTOM_PLUGINS."
        )


def build_plugin_label_map() -> dict[str, str]:
    plugin_paths, _, ignore_plugins, custom_plugins = plugin_settings_tuple()
    if not plugin_paths or plugin_paths == [""]:
        return {}
    available = list_installed_plugins(plugin_paths, custom_plugins)
    return {
        format_plugin_name(path): path
        for path in available
        if not any(ignore in format_plugin_name(path) for ignore in ignore_plugins)
    }


def list_installed_plugins(plugin_dirs: list[str], custom_plugins: list[str]) -> list[str]:
    paths = list(custom_plugins)
    for plugin_dir in plugin_dirs:
        plugin_dir = plugin_dir.strip()
        if os.path.exists(plugin_dir):
            paths.extend(glob.glob(os.path.join(plugin_dir, "*.vst3")))
            if sys.platform != "darwin":
                paths.extend(glob.glob(os.path.join(plugin_dir, "*.vst")))
            paths.extend(glob.glob(os.path.join(plugin_dir, "*.dll")))
            paths.extend(glob.glob(os.path.join(plugin_dir, "*.so")))
            paths.extend(glob.glob(os.path.join(plugin_dir, "*.component")))
    return paths


def macos_legacy_vst2_path(plugin_path: str) -> bool:
    """True if ``plugin_path`` is a legacy macOS VST2 bundle (``.vst`` but not ``.vst3``)."""
    if sys.platform != "darwin":
        return False
    base = plugin_path.rstrip("/")
    low = base.lower()
    return low.endswith(".vst") and not low.endswith(".vst3")


def format_plugin_name(plugin_path: str) -> str:
    return (
        os.path.basename(plugin_path)
        .replace(".vst3", "")
        .replace(".vst", "")
        .replace(".component", " (AU)")
    )


def load_plugin(plugin_path: str, plugin_name: str | None = None):
    """Return ``(processor, effective_plugin_name, engine)``."""
    del plugin_name  # metadata only until DawDreamer supports factory variants
    engine = create_engine()
    proc = _load_plugin_on_engine(engine, plugin_path)
    return proc, "", engine


def reload_plugin_preserving_state(
    plugin_path: str, plugin_name_hint: str | None, old_plugin, old_engine
):
    """Fresh processor + prior DawDreamer state file."""
    del plugin_name_hint
    new_proc = _reload_plugin_on_engine(
        old_engine, plugin_path, old_plugin, name=old_plugin.get_name()
    )
    return new_proc, "", old_engine


def _presets_index_path() -> str:
    return resolve_presets_index_path() or get_managed_file("config", "presets.json")


def load_presets_json() -> list:
    presets_path = _presets_index_path()
    if not os.path.exists(presets_path):
        return []
    with open(presets_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_presets_json(presets_data: list) -> None:
    presets_path = _presets_index_path()
    os.makedirs(os.path.dirname(presets_path) or ".", exist_ok=True)
    with open(presets_path, "w", encoding="utf-8") as f:
        json.dump(presets_data, f, indent=4)


def name_exists(
    name: str,
    *,
    exclude_uid: str | None = None,
    types: tuple[str, ...] | None = None,
) -> bool:
    presets_path = _presets_index_path()
    if not os.path.exists(presets_path):
        return False
    with open(presets_path, "r", encoding="utf-8") as f:
        presets_data = json.load(f)
    lowered = name.lower().strip()
    for preset in presets_data:
        if preset.get("name", "").lower() != lowered:
            continue
        if exclude_uid is not None and preset.get("id") == exclude_uid:
            continue
        if types is not None and preset.get("type") not in types:
            continue
        return True
    return False


def delete_preset_by_id(uid: str) -> bool:
    data = load_presets_json()
    kept: list = []
    removed_paths: list[str] = []
    found = False
    for preset in data:
        if preset.get("id") != uid:
            kept.append(preset)
            continue
        found = True
        if preset.get("type") == "instrument":
            p = preset.get("preset_path")
            if p:
                removed_paths.append(p)
        elif preset.get("type") == "effect":
            p = preset.get("preset_path")
            if p:
                removed_paths.append(p)
        elif preset.get("type") == "effect_chain":
            for eff in preset.get("effects") or []:
                p = eff.get("preset_path")
                if p:
                    removed_paths.append(p)
    if not found:
        return False
    save_presets_json(kept)
    for p in removed_paths:
        try:
            if p and os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass
    return True


def effect_chain_tuple_to_json_effects(effect_chain_tuples: list) -> list[dict]:
    return [
        {
            "id": fx[3][1],
            "name": fx[3][0],
            "plugin_path": fx[0],
            "plugin_name": fx[1],
            "preset_path": fx[3][2],
        }
        for fx in effect_chain_tuples
    ]


def upsert_preset_entry(preset_data: dict) -> None:
    data = load_presets_json()
    uid = preset_data.get("id")
    idx = next((i for i, p in enumerate(data) if p.get("id") == uid), None)
    if idx is None:
        data.append(preset_data)
    else:
        data[idx] = preset_data
    save_presets_json(data)


def list_presets(show_chain_plugins: bool = False, show_patterns: bool = False) -> None:
    presets_path = _presets_index_path()
    if not os.path.exists(presets_path):
        print("No presets found.")
        return

    try:
        with open(presets_path, "r", encoding="utf-8") as f:
            presets_data = json.load(f)
    except json.JSONDecodeError:
        print(with_prompt("Error reading preset index file."))
        return

    patterns = []
    instruments: list[tuple[int, str]] = []
    effects_index: list[tuple[int, dict]] = []

    for idx, preset in enumerate(presets_data, start=1):
        if preset.get("type") == "instrument":
            instruments.append((idx, preset["name"]))
        elif preset.get("type") in ("effect", "effect_chain"):
            effects_index.append((idx, preset))

    for idx, pattern in enumerate(SUPPORTED_PATTERNS_INFO, start=1):
        patterns.append((idx, pattern[0], pattern[1]))

    if len(instruments) < 1:
        print(
            with_prompt(
                "No instruments — use Patchcraftr from the desktop tray (Launch patchcraftr)."
            )
        )
        return

    if len(effects_index) < 1:
        print(
            with_prompt(
                "No saved effects — use Patchcraftr from the desktop tray (Launch patchcraftr)."
            )
        )
        return

    longest_pattern_name_length = len(max(patterns, key=lambda x: len(x[1]))[1])
    longest_instrument_name_length = len(max(instruments, key=lambda x: len(x[1]))[1])
    longest_effect_label_length = max(len(item[1]["name"]) for item in effects_index)

    if show_chain_plugins:
        for _idx, effect_preset in effects_index:
            if effect_preset.get("type") != "effect_chain":
                continue
            for plugin in effect_preset.get("effects") or []:
                if len(plugin["name"]) + 2 > longest_effect_label_length:
                    longest_effect_label_length = len(plugin["name"]) + 2

    if show_patterns:
        print(f"{MAGENTA}■ patterns{RESET}")
        print(f"{MAGENTA}│{RESET}")
        for idx, name, desc in patterns:
            desc_spacing = " " * (longest_pattern_name_length - len(name))
            print(f"{MAGENTA}│  {name} {RESET}{desc_spacing}{desc}")
        print(f"{MAGENTA}│{RESET}")

    print(f"{MAGENTA}■ instruments{RESET}")
    print(f"{MAGENTA}│{RESET}")
    for _idx, name in instruments:
        name_spacing = " " * (longest_instrument_name_length - len(name))
        print(f"{MAGENTA}│  {name} {RESET}{name_spacing}")

    print(f"{MAGENTA}│{RESET}")
    if effects_index:
        print(f"{MAGENTA}■ effects (--effect accepts these names){RESET}")
        print(f"{MAGENTA}│{RESET}")
        for _idx, effect_preset in effects_index:
            kind = ""
            if effect_preset.get("type") == "effect":
                kind = " [single]"
            name_spacing = " " * max(
                0,
                longest_effect_label_length - len(effect_preset["name"]) - len(kind),
            )
            print(f"{MAGENTA}│  {effect_preset['name']}{kind}{RESET}{name_spacing}")
            if show_chain_plugins and effect_preset.get("type") == "effect_chain":
                for effect in effect_preset.get("effects") or []:
                    name_spacing_inner = " " * max(
                        0, longest_effect_label_length - len(effect["name"]) - 2
                    )
                    print(f"{MAGENTA}│    {effect['name']} {RESET}{name_spacing_inner}")


def slot_allowed_as_chain_effect(plugin, plugin_name: str, assert_instrument: list[str]) -> bool:
    """FX chain accepts plug-ins that report as effects unless overridden by ASSERT_INSTRUMENT."""
    if plugin_is_effect(plugin) and plugin_name not in assert_instrument:
        return True
    return False


# Backward-compatible aliases during migration
load_pedalboard_plugin = load_plugin
reload_pedalboard_plugin_preserving_state = reload_plugin_preserving_state
macos_pedalboard_rejects_vst2_path = macos_legacy_vst2_path
