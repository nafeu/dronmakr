"""Instrument pool selection for multi-iteration drone generation."""

from __future__ import annotations

import random
from typing import Any

from dronmakr.audio.faust_library import faust_path_for_id, list_faust_instruments


def instrument_item_key(item: dict[str, Any]) -> str:
    kind = (item.get("kind") or "").strip().lower()
    if kind == "faust":
        faust_id = (item.get("faustId") or item.get("faust_id") or "").strip()
        return f"faust:{faust_id}"
    if kind == "patch":
        name = (item.get("name") or "").strip()
        return f"patch:{name}"
    if kind == "plugin":
        path = (item.get("pluginPath") or item.get("plugin_path") or "").strip()
        return f"plugin:{path}"
    return f"unknown:{kind}"


def pool_item_to_instrument_selection(item: dict[str, Any]) -> dict[str, Any]:
    """Convert a pool member to a single-instrument selection for rendering."""
    kind = (item.get("kind") or "").strip().lower()
    if kind == "faust":
        faust_id = (item.get("faustId") or item.get("faust_id") or "").strip()
        if not faust_id:
            raise ValueError("Faust instrument selection is missing faustId.")
        label = (item.get("label") or item.get("name") or "").strip()
        if not label:
            match = next((entry for entry in list_faust_instruments() if entry["id"] == faust_id), None)
            label = match["label"] if match else faust_id
        return {
            "kind": "faust",
            "faustId": faust_id,
            "pluginPath": faust_path_for_id(faust_id),
            "label": label,
        }
    if kind == "patch":
        name = (item.get("name") or "").strip()
        if not name:
            raise ValueError("Patch instrument selection is missing name.")
        from dronmakr.apps.generatr_plugins import get_drone_patch_detail
        from dronmakr.audio.faust_library import is_faust_instrument_path

        patch = get_drone_patch_detail(name)
        plugin_path = (patch.get("pluginPath") or "").strip()
        preset_path = (patch.get("presetPath") or "").strip()
        if is_faust_instrument_path(plugin_path):
            from dronmakr.audio.faust_library import faust_id_from_path

            faust_id = faust_id_from_path(plugin_path)
            return {
                "kind": "faust",
                "faustId": faust_id,
                "pluginPath": plugin_path,
                "label": patch.get("name") or name,
            }
        return {
            "kind": "plugin",
            "pluginPath": plugin_path,
            "presetPath": preset_path,
            "label": patch.get("name") or name,
            "pluginName": patch.get("pluginName") or "",
        }
    if kind == "plugin":
        plugin_path = (item.get("pluginPath") or item.get("plugin_path") or "").strip()
        if not plugin_path:
            raise ValueError("Instrument plug-in selection is missing pluginPath.")
        return {
            "kind": "plugin",
            "pluginPath": plugin_path,
            "presetPath": item.get("presetPath") or item.get("preset_path") or "",
            "label": item.get("label") or item.get("name") or "",
            "pluginName": item.get("pluginName") or item.get("plugin_name") or "",
        }
    raise ValueError(f"Unsupported instrument pool item kind: {kind or '(empty)'}")


def pick_pool_item(
    items: list[dict[str, Any]],
    mode: str,
    iteration_index: int,
    last_key: str | None = None,
) -> dict[str, Any]:
    if not items:
        raise ValueError("Instrument pool is empty.")
    if len(items) == 1:
        return items[0]
    normalized_mode = (mode or "random").strip().lower()
    if normalized_mode == "round_robin":
        return items[iteration_index % len(items)]
    if normalized_mode == "random":
        return random.choice(items)
    if normalized_mode == "random_unique":
        if len(items) == 2 and last_key:
            others = [item for item in items if instrument_item_key(item) != last_key]
            return random.choice(others if others else items)
        if last_key:
            candidates = [item for item in items if instrument_item_key(item) != last_key]
            if candidates:
                return random.choice(candidates)
        return random.choice(items)
    return random.choice(items)


def resolve_effective_instrument_selection(
    instrument_selection: dict[str, Any] | None,
    *,
    iteration_index: int = 0,
    last_key: str | None = None,
) -> dict[str, Any] | None:
    """Expand a pool selection to a single instrument for one render pass."""
    if not isinstance(instrument_selection, dict):
        return instrument_selection
    if (instrument_selection.get("kind") or "").strip().lower() != "pool":
        return instrument_selection
    items = instrument_selection.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Instrument pool is empty.")
    mode = instrument_selection.get("mode") or "random"
    picked = pick_pool_item(items, mode, iteration_index, last_key)
    return pool_item_to_instrument_selection(picked)
