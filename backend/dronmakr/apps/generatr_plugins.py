"""Generate Samples drone plug-in picker and editor capture."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import threading
from typing import Any

from dronmakr.audio.audio_host import (
    apply_plugin_state,
    create_engine,
    load_instrument as load_instrument_on_engine,
    load_plugin as load_plugin_on_engine,
    open_plugin_editor,
    plugin_is_effect,
    plugin_is_instrument,
    run_live_preview_during_editor,
    save_plugin_state,
)
from dronmakr.core.paths import get_managed_dir
from dronmakr.core.utils import TEMP_DIR, generate_id, resolve_presets_index_path
from dronmakr.audio.faust_library import list_faust_instruments, list_faust_library_categories
from dronmakr.presets.preset_authoring import (
    format_plugin_name,
    get_plugin_scan_cache_info,
    list_installed_plugin_entries,
    load_presets_json,
    name_exists,
    plugin_scan_cache_needs_initial_scan,
    plugin_settings_tuple,
    read_plugin_scan_progress,
    save_allowed_plugins_for_role,
    save_presets_json,
)


def generatr_session_dir() -> str:
    path = os.path.join(TEMP_DIR, "generatr-plugin-sessions")
    os.makedirs(path, exist_ok=True)
    return path


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
    """True if a plug-in bundle/binary or built-in Faust instrument exists."""
    from dronmakr.audio.faust_library import faust_instrument_path_exists

    path = (plugin_path or "").strip()
    if faust_instrument_path_exists(path):
        return True
    return bool(path) and os.path.exists(path)


def _role_plugins(role: str, *, respect_ignore: bool) -> list[dict]:
    return list_installed_plugin_entries(role=role, respect_ignore=respect_ignore)


def get_drone_plugin_scan_status() -> dict:
    progress = read_plugin_scan_progress()
    cache = get_plugin_scan_cache_info()
    progress["scanning"] = progress.get("status") == "running"
    progress["cacheComplete"] = cache["complete"]
    progress["cache"] = cache
    progress["needsInitialScan"] = plugin_scan_cache_needs_initial_scan()
    return progress


def start_drone_plugin_scan(*, force: bool = False) -> dict:
    from dronmakr.audio.audio_worker import spawn_background_plugin_scan

    progress = read_plugin_scan_progress()
    if progress.get("status") == "running":
        return {**get_drone_plugin_scan_status(), "scanning": True, "started": False}

    if not force and not plugin_scan_cache_needs_initial_scan():
        return {**get_drone_plugin_scan_status(), "started": False, "skipped": True}

    spawn_background_plugin_scan(force=force)
    return {**get_drone_plugin_scan_status(), "started": True}


def get_drone_patch_detail(name: str) -> dict:
    """Return plug-in paths for a saved patch by name."""
    patch_name = (name or "").strip()
    if not patch_name:
        raise ValueError("Patch name is required.")
    presets = _load_presets_list()
    patch = next((p for p in presets if p.get("name") == patch_name), None)
    if patch is None:
        raise ValueError(f"No saved patch named '{patch_name}'.")
    if patch.get("type") == "effect_chain":
        raise ValueError(
            f"Patch '{patch_name}' is an FX chain — open an individual slot from the chain instead."
        )
    summary = _patch_summary(patch)
    if not summary.get("pluginPath"):
        raise ValueError(f"Patch '{patch_name}' has no plug-in path configured.")
    return summary


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
        "library": list_faust_instruments() if role == "instrument" else [],
        "libraryCategories": list_faust_library_categories() if role == "instrument" else [],
        "pluginPathsConfigured": bool(plugins) or bool(list_installed_plugin_entries(respect_ignore=False)),
        "scan": get_drone_plugin_scan_status(),
    }


def get_drone_plugin_list_editor_payload(role: str) -> dict:
    """Return allowed vs detected plug-ins for the list editor modal."""
    role = (role or "instrument").strip().lower()
    detected = _role_plugins(role, respect_ignore=False)
    allowed = _role_plugins(role, respect_ignore=True)
    scan = get_drone_plugin_scan_status()
    return {
        "role": role,
        "allowed": [{"label": entry["label"], "path": entry["path"]} for entry in allowed],
        "detected": [{"label": entry["label"], "path": entry["path"]} for entry in detected],
        "scan": scan,
    }


def save_drone_plugin_list_editor(role: str, allowed_labels: list[str]) -> dict:
    """Persist allowed plug-ins for a role via ``IGNORE_PLUGINS``."""
    role = (role or "instrument").strip().lower()
    ignore_plugins = save_allowed_plugins_for_role(role, allowed_labels)
    return {
        "role": role,
        "ignorePlugins": ignore_plugins,
        **get_drone_plugin_list_editor_payload(role),
    }


def _resolve_edit_processor(
    instrument,
    fx_processors: list[tuple[Any, str]],
    role: str,
    plugin_path: str,
):
    plugin_path = os.path.abspath((plugin_path or "").strip())
    if role == "instrument":
        return instrument
    for fx, path in fx_processors:
        if os.path.abspath(path) == plugin_path:
            return fx
    raise ValueError(f"Edited FX plug-in is not loaded in the preview chain: {plugin_path}")


def _load_editor_preview_chain(
    engine,
    instrument_path: str,
    instrument_state_path: str | None,
    fx_specs: list[tuple[str, str | None]],
):
    instrument = load_instrument_on_engine(engine, instrument_path, name="instrument")
    if instrument_state_path and os.path.isfile(instrument_state_path):
        apply_plugin_state(instrument, instrument_state_path)
    fx_processors: list[tuple[Any, str]] = []
    for idx, (path, state_path) in enumerate(fx_specs):
        abs_path = os.path.abspath(path)
        fx = load_plugin_on_engine(engine, abs_path, name=f"fx_{idx}")
        if state_path and os.path.isfile(state_path):
            apply_plugin_state(fx, state_path)
        fx_processors.append((fx, abs_path))
    return instrument, fx_processors


def open_drone_plugin_editor_capture(
    plugin_path: str,
    role: str,
    preset_path: str | None = None,
    *,
    editor_preview: dict | None = None,
) -> dict:
    """Open DawDreamer plug-in editor; save state snapshot when the window closes."""
    plugin_path = os.path.abspath((plugin_path or "").strip())
    role = (role or "instrument").strip().lower()
    if not _plugin_path_exists(plugin_path):
        raise FileNotFoundError(f"Plug-in not found: {plugin_path}")

    existing_preset = os.path.abspath((preset_path or "").strip()) if preset_path else ""
    preview_midi_path = ""
    try:
        if editor_preview:
            instrument_path = (editor_preview.get("instrument_path") or "").strip()
            if not instrument_path:
                raise ValueError("Editor preview is missing instrument_path.")
            instrument_state_path = (editor_preview.get("instrument_state_path") or "").strip() or None
            fx_specs = [
                (str(item[0]), str(item[1]) if item[1] else None)
                for item in (editor_preview.get("fx_specs") or [])
                if item and item[0]
            ]
            preview_midi_path = (editor_preview.get("midi_path") or "").strip()
            preview_duration = float(editor_preview.get("duration_sec") or 0)
            if not preview_midi_path or not os.path.isfile(preview_midi_path):
                raise ValueError("Editor preview MIDI file is missing.")
            if preview_duration <= 0:
                raise ValueError("Editor preview duration must be positive.")

            engine = create_engine()
            instrument, fx_processors = _load_editor_preview_chain(
                engine,
                instrument_path,
                instrument_state_path,
                fx_specs,
            )
            processor = _resolve_edit_processor(instrument, fx_processors, role, plugin_path)
            if role == "instrument" and not plugin_is_instrument(processor):
                raise ValueError(
                    "Selected plug-in is not an instrument (it accepts audio input). "
                    "Choose a synth or instrument plug-in."
                )
            if role == "effect" and not plugin_is_effect(processor):
                raise ValueError(
                    "Selected plug-in is an instrument (no audio input). Choose an FX plug-in."
                )
            if existing_preset and os.path.isfile(existing_preset):
                apply_plugin_state(processor, existing_preset)

            engine_lock = threading.RLock()

            def open_editor() -> None:
                open_plugin_editor(processor)

            run_live_preview_during_editor(
                instrument=instrument,
                fx_processors=[fx for fx, _path in fx_processors],
                midi_path=preview_midi_path,
                duration_sec=preview_duration,
                engine=engine,
                engine_lock=engine_lock,
                open_editor_fn=open_editor,
            )
        else:
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

            if existing_preset and os.path.isfile(existing_preset):
                apply_plugin_state(processor, existing_preset)

            open_plugin_editor(processor)

        saved_preset_path = os.path.join(generatr_session_dir(), f"{generate_id()}.ddstate")
        save_plugin_state(processor, saved_preset_path)

        return {
            "kind": "plugin",
            "pluginPath": plugin_path,
            "presetPath": saved_preset_path,
            "label": format_plugin_name(plugin_path),
        }
    finally:
        if preview_midi_path:
            with contextlib.suppress(OSError):
                os.remove(preview_midi_path)


def _persist_preset_state(source_path: str, name_hint: str) -> str:
    source = os.path.abspath((source_path or "").strip())
    if not source or not os.path.isfile(source):
        raise ValueError(f"Preset state file not found: {source_path or '(empty)'}")
    dest_dir = get_managed_dir("vst-preset-files")
    os.makedirs(dest_dir, exist_ok=True)
    safe = re.sub(r"[^\w\-]+", "_", (name_hint or "").strip()).strip("_")[:40] or "preset"
    dest = os.path.join(dest_dir, f"{safe}_{generate_id()}.ddstate")
    shutil.copy2(source, dest)
    return dest


def _resolve_drone_selection(selection: dict) -> tuple[str, str, str, str]:
    """Return ``plugin_path``, ``plugin_name``, ``preset_source_path``, ``step_label``."""
    if not isinstance(selection, dict):
        raise ValueError("Invalid plug-in selection.")
    kind = (selection.get("kind") or "").strip().lower()
    step_label = (
        (selection.get("label") or selection.get("name") or "").strip()
    )
    if kind == "plugin":
        plugin_path = (selection.get("pluginPath") or selection.get("plugin_path") or "").strip()
        preset_path = (selection.get("presetPath") or selection.get("preset_path") or "").strip()
        plugin_name = (selection.get("pluginName") or selection.get("plugin_name") or "").strip()
        if not plugin_path:
            raise ValueError("Plug-in selection is missing pluginPath.")
        if not step_label:
            step_label = format_plugin_name(plugin_path)
        return plugin_path, plugin_name, preset_path, step_label
    if kind == "faust":
        from dronmakr.audio.faust_library import faust_path_for_id

        faust_id = (selection.get("faustId") or selection.get("faust_id") or "").strip()
        if not faust_id:
            raise ValueError("Faust selection is missing faustId.")
        plugin_path = faust_path_for_id(faust_id)
        if not step_label:
            step_label = (selection.get("label") or selection.get("name") or faust_id).strip()
        return plugin_path, "Faust", "", step_label or faust_id
    if kind == "patch":
        plugin_path = (selection.get("pluginPath") or selection.get("plugin_path") or "").strip()
        preset_path = (selection.get("presetPath") or selection.get("preset_path") or "").strip()
        plugin_name = (selection.get("pluginName") or selection.get("plugin_name") or "").strip()
        if not step_label:
            step_label = (selection.get("name") or "").strip()
        if plugin_path and preset_path:
            return plugin_path, plugin_name, preset_path, step_label or format_plugin_name(plugin_path)
        patch_name = (selection.get("name") or "").strip()
        if not patch_name:
            raise ValueError("Saved patch selection is missing a name.")
        patch = next((p for p in load_presets_json() if p.get("name") == patch_name), None)
        if patch is None:
            raise ValueError(f"No saved patch named '{patch_name}'.")
        if patch.get("type") == "effect_chain":
            raise ValueError(
                f"Patch '{patch_name}' is an FX chain — save individual FX slots instead."
            )
        plugin_path = (patch.get("plugin_path") or "").strip()
        preset_path = (patch.get("preset_path") or "").strip()
        plugin_name = (patch.get("plugin_name") or "").strip()
        if not plugin_path:
            raise ValueError(f"Patch '{patch_name}' has no plug-in path configured.")
        if not step_label:
            step_label = patch_name
        return plugin_path, plugin_name, preset_path, step_label
    raise ValueError("Unsupported selection kind — choose a plug-in or saved patch first.")


def _build_saved_effect_step(selection: dict, *, name_hint: str) -> dict:
    plugin_path, plugin_name, preset_source, step_label = _resolve_drone_selection(selection)
    if not _plugin_path_exists(plugin_path):
        raise ValueError(f"Plug-in not found: {plugin_path}")
    preset_path = _persist_preset_state(preset_source, name_hint)
    return {
        "id": generate_id(),
        "name": step_label or format_plugin_name(plugin_path),
        "plugin_path": plugin_path,
        "plugin_name": plugin_name,
        "preset_path": preset_path,
    }


def save_drone_preset(
    *,
    role: str,
    name: str,
    instrument_selection: dict | None = None,
    fx_slots: list | None = None,
) -> dict:
    """Persist instrument or FX selection(s) into ``presets.json``."""
    role = (role or "instrument").strip().lower()
    preset_name = (name or "").strip()
    if not preset_name:
        raise ValueError("Preset name is required.")
    if name_exists(preset_name):
        raise ValueError(f"A preset named '{preset_name}' already exists.")

    if role == "instrument":
        if not isinstance(instrument_selection, dict):
            raise ValueError("Choose an instrument plug-in or saved patch before saving.")
        plugin_path, plugin_name, preset_source, _ = _resolve_drone_selection(instrument_selection)
        if not _plugin_path_exists(plugin_path):
            raise ValueError(f"Plug-in not found: {plugin_path}")
        from dronmakr.audio.faust_library import is_faust_instrument_path

        if is_faust_instrument_path(plugin_path):
            preset_path = ""
        else:
            preset_path = _persist_preset_state(preset_source, preset_name)
        preset_data = {
            "id": generate_id(),
            "name": preset_name,
            "type": "instrument",
            "plugin_path": plugin_path,
            "plugin_name": plugin_name,
            "preset_path": preset_path,
        }
    elif role == "effect":
        filled = [slot for slot in (fx_slots or []) if slot]
        if not filled:
            raise ValueError("Add at least one FX slot before saving.")
        if len(filled) == 1:
            step = _build_saved_effect_step(filled[0], name_hint=preset_name)
            preset_data = {
                "id": generate_id(),
                "name": preset_name,
                "type": "effect",
                "plugin_path": step["plugin_path"],
                "plugin_name": step["plugin_name"],
                "preset_path": step["preset_path"],
            }
        else:
            effects = [
                _build_saved_effect_step(slot, name_hint=f"{preset_name}_{idx + 1}")
                for idx, slot in enumerate(filled)
            ]
            preset_data = {
                "id": generate_id(),
                "name": preset_name,
                "type": "effect_chain",
                "effects": effects,
            }
    else:
        raise ValueError("role must be instrument or effect")

    data = load_presets_json()
    data.append(preset_data)
    save_presets_json(data)
    return _patch_summary(preset_data)
