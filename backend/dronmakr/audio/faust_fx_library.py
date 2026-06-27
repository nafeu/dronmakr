"""Built-in Faust effect library for drone FX chain slots."""

from __future__ import annotations

import os
from typing import Any

from dronmakr.core.bundle_paths import bundled_asset_path

FAUST_FX_PREFIX = "faustfx:"

_FAUST_FX_CATEGORIES: tuple[dict[str, str], ...] = (
    {"id": "tone", "label": "Tone & Filter"},
    {"id": "space", "label": "Space & Time"},
    {"id": "reverb", "label": "Reverb & Wash"},
    {"id": "texture", "label": "Texture & Motion"},
    {"id": "stereo", "label": "Stereo & Width"},
)

_FAUST_EFFECTS: tuple[dict[str, str], ...] = (
    {
        "id": "warm_saturation",
        "category": "tone",
        "label": "Warm Saturation",
        "description": "Soft tanh saturation for smoky harmonic bloom.",
        "file": "fx/warm_saturation.dsp",
    },
    {
        "id": "dark_lowpass",
        "category": "tone",
        "label": "Dark Lowpass",
        "description": "Moody resonant low-pass for distant, muffled tones.",
        "file": "fx/dark_lowpass.dsp",
    },
    {
        "id": "rumble_tunnel",
        "category": "tone",
        "label": "Rumble Tunnel",
        "description": "Resonant band-pass tunnel for sub-heavy drone focus.",
        "file": "fx/rumble_tunnel.dsp",
    },
    {
        "id": "ghost_echo",
        "category": "space",
        "label": "Ghost Echo",
        "description": "Long feedback delay with a darkened tail.",
        "file": "fx/ghost_echo.dsp",
    },
    {
        "id": "diffuse_cloud",
        "category": "space",
        "label": "Diffuse Cloud",
        "description": "All-pass diffusion smear for foggy ambience.",
        "file": "fx/diffuse_cloud.dsp",
    },
    {
        "id": "shimmer_haze",
        "category": "space",
        "label": "Shimmer Haze",
        "description": "Detuned delay blend for spectral haze.",
        "file": "fx/shimmer_haze.dsp",
    },
    {
        "id": "cathedral_wash",
        "category": "reverb",
        "label": "Cathedral Wash",
        "description": "Bright hall reverb with a long, airy decay tail.",
        "file": "fx/cathedral_wash.dsp",
    },
    {
        "id": "void_wash",
        "category": "reverb",
        "label": "Void Wash",
        "description": "Vast dark reverb that fades into near-infinite space.",
        "file": "fx/void_wash.dsp",
    },
    {
        "id": "plate_wash",
        "category": "reverb",
        "label": "Plate Wash",
        "description": "Dense plate-style reverb with a bright, smooth long tail.",
        "file": "fx/plate_wash.dsp",
    },
    {
        "id": "cave_wash",
        "category": "reverb",
        "label": "Cave Wash",
        "description": "Dark cavern reverb with a warm, muffled long tail.",
        "file": "fx/cave_wash.dsp",
    },
    {
        "id": "cloud_wash",
        "category": "reverb",
        "label": "Cloud Wash",
        "description": "Spectral greyhole wash with floating cloud-like diffusion.",
        "file": "fx/cloud_wash.dsp",
    },
    {
        "id": "abyss_wash",
        "category": "reverb",
        "label": "Abyss Wash",
        "description": "Sub-heavy deep-space reverb with a slow, bottomless fade.",
        "file": "fx/abyss_wash.dsp",
    },
    {
        "id": "tank_wash",
        "category": "reverb",
        "label": "Tank Wash",
        "description": "Custom diffusion tank with cascading delay smear.",
        "file": "fx/tank_wash.dsp",
    },
    {
        "id": "tape_wobble",
        "category": "texture",
        "label": "Tape Wobble",
        "description": "Modulated micro-delay for unstable tape drift.",
        "file": "fx/tape_wobble.dsp",
    },
    {
        "id": "bit_mist",
        "category": "texture",
        "label": "Bit Mist",
        "description": "Lo-fi bit reduction with a dusty digital grain.",
        "file": "fx/bit_mist.dsp",
    },
    {
        "id": "freeze_grain",
        "category": "texture",
        "label": "Freeze Grain",
        "description": "Comb-filter grain cluster for metallic shimmer.",
        "file": "fx/freeze_grain.dsp",
    },
    {
        "id": "stereo_haze",
        "category": "stereo",
        "label": "Stereo Haze",
        "description": "Modulated chorus drift for drifting, wide ambience.",
        "file": "fx/stereo_haze.dsp",
    },
)


def is_faust_fx_path(path: str) -> bool:
    return (path or "").strip().lower().startswith(FAUST_FX_PREFIX)


def faust_fx_id_from_path(path: str) -> str:
    value = (path or "").strip()
    if not is_faust_fx_path(value):
        return ""
    return value[len(FAUST_FX_PREFIX) :].strip()


def faust_fx_path_for_id(effect_id: str) -> str:
    return f"{FAUST_FX_PREFIX}{effect_id}"


def _effect_summary(entry: dict[str, str]) -> dict[str, str]:
    return {
        "id": entry["id"],
        "label": entry["label"],
        "description": entry["description"],
        "category": entry["category"],
    }


def list_faust_effects() -> list[dict[str, str]]:
    """Flat catalog entries for the FX picker library column."""
    return [_effect_summary(entry) for entry in _FAUST_EFFECTS]


def list_faust_fx_categories() -> list[dict]:
    """Catalog grouped by effect type for the picker UI."""
    by_category: dict[str, list[dict[str, str]]] = {cat["id"]: [] for cat in _FAUST_FX_CATEGORIES}
    for entry in _FAUST_EFFECTS:
        by_category.setdefault(entry["category"], []).append(_effect_summary(entry))
    grouped: list[dict] = []
    for category in _FAUST_FX_CATEGORIES:
        effects = by_category.get(category["id"], [])
        if not effects:
            continue
        grouped.append(
            {
                "id": category["id"],
                "label": category["label"],
                "effects": effects,
            }
        )
    return grouped


def _catalog_entry(effect_id: str) -> dict[str, str]:
    needle = (effect_id or "").strip()
    for entry in _FAUST_EFFECTS:
        if entry["id"] == needle:
            return entry
    raise ValueError(f"Unknown Faust library effect: {effect_id or '(empty)'}")


def resolve_faust_fx_dsp_path(effect_id: str) -> str:
    """Absolute path to a bundled ``.dsp`` file."""
    entry = _catalog_entry(effect_id)
    path = bundled_asset_path("faust", entry["file"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Faust effect file not found: {path}")
    return str(path)


def faust_fx_exists(effect_id: str) -> bool:
    try:
        resolve_faust_fx_dsp_path(effect_id)
    except (ValueError, FileNotFoundError):
        return False
    return True


def faust_fx_path_exists(path: str) -> bool:
    if not is_faust_fx_path(path):
        return False
    return faust_fx_exists(faust_fx_id_from_path(path))


def load_faust_effect(engine: Any, effect_id: str, *, name: str | None = None) -> Any:
    """Compile and return a stereo (2 in / 2 out) Faust audio effect processor."""
    from dronmakr.audio.audio_host import _unique_processor_name

    dsp_path = resolve_faust_fx_dsp_path(effect_id)
    proc_name = name or _unique_processor_name("faustfx")
    processor = engine.make_faust_processor(proc_name)
    processor.set_dsp(os.path.abspath(dsp_path))
    processor.compile()
    return processor
