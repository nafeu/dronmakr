"""Static chord/scale catalog picklists parsed from ``resources/chord-scale-data.json``."""

from __future__ import annotations

import json
from typing import TypedDict

from dronmakr.core.bundle_paths import bundled_asset_path


class ChordScalePicklists(TypedDict):
    roots: list[str]
    tags: list[str]
    chartNames: list[str]


class ChordScaleEntry(TypedDict, total=False):
    id: str
    root: str
    name: str
    notes: list[str]
    tags: list[str]
    type: str


_picklists_cache: ChordScalePicklists | None = None
_catalog_cache: list[ChordScaleEntry] | None = None


def _ci_sort(vals: list[str]) -> list[str]:
    return sorted(vals, key=lambda s: (s.casefold(), s))


def get_chord_scale_picklists(path: str | None = None) -> ChordScalePicklists:
    """Load once, returning sorted unique roots, tags, and chart ``name`` values."""
    global _picklists_cache
    if _picklists_cache is not None:
        return _picklists_cache

    empty: ChordScalePicklists = {"roots": [], "tags": [], "chartNames": []}
    data_path = path or str(bundled_asset_path("resources", "chord-scale-data.json"))
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        _picklists_cache = empty
        return empty

    roots: set[str] = set()
    tags: set[str] = set()
    chart_names: set[str] = set()

    if not isinstance(raw, list):
        _picklists_cache = empty
        return empty

    for item in raw:
        if not isinstance(item, dict):
            continue
        r = item.get("root")
        if isinstance(r, str):
            stripped = r.strip()
            if stripped:
                roots.add(stripped)

        nm = item.get("name")
        if isinstance(nm, str):
            nms = nm.strip()
            if nms:
                chart_names.add(nms)

        tl = item.get("tags") or []
        if isinstance(tl, list):
            for t in tl:
                if isinstance(t, str):
                    ts = t.strip()
                    if ts:
                        tags.add(ts)

    _picklists_cache = {
        "roots": _ci_sort(list(roots)),
        "tags": _ci_sort(list(tags)),
        "chartNames": _ci_sort(list(chart_names)),
    }
    return _picklists_cache


def get_chord_scale_catalog(path: str | None = None) -> list[ChordScaleEntry]:
    """Load the full chord/scale catalog once."""
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache

    data_path = path or str(bundled_asset_path("resources", "chord-scale-data.json"))
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        _catalog_cache = []
        return _catalog_cache

    if not isinstance(raw, list):
        _catalog_cache = []
        return _catalog_cache

    entries: list[ChordScaleEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        root = item.get("root")
        name = item.get("name")
        notes = item.get("notes")
        if not isinstance(root, str) or not isinstance(name, str) or not isinstance(notes, list):
            continue
        if not root.strip() or not name.strip() or not notes:
            continue
        entry: ChordScaleEntry = {
            "id": str(item.get("id") or "").strip(),
            "root": root.strip(),
            "name": name.strip(),
            "notes": [str(n).strip() for n in notes if str(n).strip()],
            "type": str(item.get("type") or "").strip().lower(),
            "tags": [],
        }
        tags_raw = item.get("tags") or []
        if isinstance(tags_raw, list):
            entry["tags"] = [str(t).strip() for t in tags_raw if isinstance(t, str) and str(t).strip()]
        entries.append(entry)

    _catalog_cache = entries
    return _catalog_cache


def warm_chord_scale_picklists() -> None:
    """Eager-load picklists when the server or app initializes."""
    get_chord_scale_picklists()
