"""Chart pool selection for multi-iteration drone generation."""

from __future__ import annotations

from typing import Any

from dronmakr.apps.drone_instrument_pool import pick_pool_item


def chart_entry_key(entry: dict[str, Any]) -> str:
    chart_id = (entry.get("id") or "").strip()
    if chart_id:
        return f"id:{chart_id}"
    root = (entry.get("root") or "").strip()
    name = (entry.get("name") or "").strip()
    return f"chart:{root}:{name}"


def parse_drone_chart_selection(raw: object) -> dict[str, Any] | None:
    """Parse ``chartSelection`` payload: ``{ mode, charts: [...] }``."""
    if not isinstance(raw, dict):
        return None
    charts_raw = raw.get("charts")
    if not isinstance(charts_raw, list) or not charts_raw:
        return None
    mode = (raw.get("mode") or "random").strip().lower()
    if mode not in ("round_robin", "random", "random_unique"):
        raise ValueError("Chart selection mode must be round_robin, random, or random_unique.")

    charts: list[dict[str, Any]] = []
    for item in charts_raw:
        if not isinstance(item, dict):
            continue
        root = (item.get("root") or "").strip()
        name = (item.get("name") or "").strip()
        notes = item.get("notes")
        if not root or not name or not isinstance(notes, list) or not notes:
            continue
        charts.append(
            {
                "id": (item.get("id") or "").strip(),
                "root": root,
                "name": name,
                "notes": [str(n).strip() for n in notes if str(n).strip()],
                "type": (item.get("type") or "").strip().lower(),
                "tags": item.get("tags") if isinstance(item.get("tags"), list) else [],
            }
        )
    if not charts:
        raise ValueError("Chart selection must include at least one valid chart.")
    return {"mode": mode, "charts": charts}


def pick_chart_entry(
    chart_selection: dict[str, Any],
    iteration_index: int,
    last_key: str | None = None,
) -> dict[str, Any]:
    items = chart_selection.get("charts") or []
    mode = chart_selection.get("mode") or "random"
    return pick_pool_item(items, mode, iteration_index, last_key)
