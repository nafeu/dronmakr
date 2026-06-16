"""Static chord/scale catalog picklists parsed from ``resources/chord-scale-data.json``."""

from __future__ import annotations

import json
import os
from typing import TypedDict


class ChordScalePicklists(TypedDict):
    roots: list[str]
    tags: list[str]
    chartNames: list[str]


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_CHORD_SCALE_JSON = os.path.join(_REPO_ROOT, "resources", "chord-scale-data.json")

_picklists_cache: ChordScalePicklists | None = None


def _ci_sort(vals: list[str]) -> list[str]:
    return sorted(vals, key=lambda s: (s.casefold(), s))


def get_chord_scale_picklists(path: str | None = None) -> ChordScalePicklists:
    """Load once, returning sorted unique roots, tags, and chart ``name`` values."""
    global _picklists_cache
    if _picklists_cache is not None:
        return _picklists_cache

    empty: ChordScalePicklists = {"roots": [], "tags": [], "chartNames": []}
    data_path = path or _CHORD_SCALE_JSON
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


def warm_chord_scale_picklists() -> None:
    """Eager-load picklists when the server or app initializes."""
    get_chord_scale_picklists()
