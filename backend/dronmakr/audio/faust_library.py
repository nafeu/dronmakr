"""Built-in Faust instrument library for drone generation."""

from __future__ import annotations

import os
from typing import Any

from dronmakr.core.bundle_paths import bundled_asset_path

FAUST_INSTRUMENT_PREFIX = "faust:"
FAUST_POLYPHONY_VOICES = 16

_FAUST_CATEGORIES: tuple[dict[str, str], ...] = (
    {"id": "oscillators", "label": "Oscillators"},
    {"id": "subtractive", "label": "Subtractive"},
    {"id": "pads", "label": "Pads & Strings"},
    {"id": "texture", "label": "Noise & Texture"},
    {"id": "plucks", "label": "Plucks & Mallets"},
    {"id": "keys", "label": "Keys & Organs"},
    {"id": "metallic", "label": "Metallic & FM"},
    {"id": "drones", "label": "Drones & Beds"},
)

_FAUST_INSTRUMENTS: tuple[dict[str, str], ...] = (
    # Oscillators
    {"id": "sine_osc", "category": "oscillators", "label": "Sine Oscillator", "description": "Warm sine wave", "file": "sine_osc.dsp"},
    {"id": "triangle_osc", "category": "oscillators", "label": "Triangle Oscillator", "description": "Soft triangle wave", "file": "triangle_osc.dsp"},
    {"id": "saw_osc", "category": "oscillators", "label": "Saw Oscillator", "description": "Bright sawtooth wave", "file": "saw_osc.dsp"},
    {"id": "square_osc", "category": "oscillators", "label": "Square Oscillator", "description": "Hollow square wave", "file": "square_osc.dsp"},
    {"id": "pulse_osc", "category": "oscillators", "label": "Pulse Oscillator", "description": "Variable-width pulse wave", "file": "pulse_osc.dsp"},
    {"id": "dual_saw", "category": "oscillators", "label": "Dual Saw", "description": "Detuned saw pair", "file": "dual_saw.dsp"},
    {"id": "sub_stack", "category": "oscillators", "label": "Sub Stack", "description": "Saw with sub-octave layer", "file": "sub_stack.dsp"},
    {"id": "pwm_shimmer", "category": "oscillators", "label": "PWM Shimmer", "description": "Pulse width modulation", "file": "pwm_shimmer.dsp"},
    # Subtractive
    {"id": "lowpass_saw", "category": "subtractive", "label": "Lowpass Saw", "description": "Resonant filtered saw", "file": "lowpass_saw.dsp"},
    {"id": "acid_squawk", "category": "subtractive", "label": "Acid Squawk", "description": "Squelchy resonant square", "file": "acid_squawk.dsp"},
    {"id": "warm_triangle", "category": "subtractive", "label": "Warm Triangle", "description": "Gentle low-pass triangle", "file": "warm_triangle.dsp"},
    # Pads & Strings
    {"id": "glass_pad", "category": "pads", "label": "Glass Pad", "description": "Bright airy pad", "file": "glass_pad.dsp"},
    {"id": "choir_cloud", "category": "pads", "label": "Choir Cloud", "description": "Detuned vocal cloud", "file": "choir_cloud.dsp"},
    {"id": "velvet_pad", "category": "pads", "label": "Velvet Pad", "description": "Dark long-release pad", "file": "velvet_pad.dsp"},
    {"id": "ghost_strings", "category": "pads", "label": "Ghost Strings", "description": "Ethereal bowed strings", "file": "ghost_strings.dsp"},
    {"id": "lunar_pad", "category": "pads", "label": "Lunar Pad", "description": "Drifting filtered pad", "file": "lunar_pad.dsp"},
    # Noise & Texture
    {"id": "pink_wind", "category": "texture", "label": "Pink Wind", "description": "Breathy band-pass noise", "file": "pink_wind.dsp"},
    {"id": "static_drift", "category": "texture", "label": "Static Drift", "description": "Wandering crackle texture", "file": "static_drift.dsp"},
    {"id": "dust_bed", "category": "texture", "label": "Dust Bed", "description": "Low rumbling noise bed", "file": "dust_bed.dsp"},
    # Plucks & Mallets
    {"id": "soft_pluck", "category": "plucks", "label": "Soft Pluck", "description": "Gentle decaying pluck", "file": "soft_pluck.dsp"},
    {"id": "mallet_fm", "category": "plucks", "label": "Mallet FM", "description": "Percussive FM mallet", "file": "mallet_fm.dsp"},
    {"id": "harpsichord_tick", "category": "plucks", "label": "Harpsichord Tick", "description": "Bright percussive tick", "file": "harpsichord_tick.dsp"},
    # Keys & Organs
    {"id": "drawbar_organ", "category": "keys", "label": "Drawbar Organ", "description": "Additive organ tone", "file": "drawbar_organ.dsp"},
    {"id": "reed_wind", "category": "keys", "label": "Reed Wind", "description": "Nasal reed tone", "file": "reed_wind.dsp"},
    {"id": "fm_bell", "category": "keys", "label": "FM Bell", "description": "Bright bell partials", "file": "fm_bell.dsp"},
    # Metallic & FM
    {"id": "clang_stack", "category": "metallic", "label": "Clang Stack", "description": "Inharmonic metallic clang", "file": "clang_stack.dsp"},
    {"id": "crystal_fm", "category": "metallic", "label": "Crystal FM", "description": "Glassy high-ratio FM", "file": "crystal_fm.dsp"},
    {"id": "bronze_reson", "category": "metallic", "label": "Bronze Reson", "description": "Resonant bronze ring", "file": "bronze_reson.dsp"},
    # Drones & Beds
    {"id": "abyss_rumble", "category": "drones", "label": "Abyss Rumble", "description": "Sub-heavy undertow", "file": "abyss_rumble.dsp"},
    {"id": "pulse_drift", "category": "drones", "label": "Pulse Drift", "description": "Slowly pulsing square", "file": "pulse_drift.dsp"},
    {"id": "solar_haze", "category": "drones", "label": "Solar Haze", "description": "Wide pad with noise halo", "file": "solar_haze.dsp"},
    {"id": "ocean_floor", "category": "drones", "label": "Ocean Floor", "description": "Deep sine with noise swell", "file": "ocean_floor.dsp"},
)


def is_faust_instrument_path(path: str) -> bool:
    return (path or "").strip().lower().startswith(FAUST_INSTRUMENT_PREFIX)


def faust_id_from_path(path: str) -> str:
    value = (path or "").strip()
    if not is_faust_instrument_path(value):
        return ""
    return value[len(FAUST_INSTRUMENT_PREFIX) :].strip()


def faust_path_for_id(instrument_id: str) -> str:
    return f"{FAUST_INSTRUMENT_PREFIX}{instrument_id}"


def _instrument_summary(entry: dict[str, str]) -> dict[str, str]:
    return {
        "id": entry["id"],
        "label": entry["label"],
        "description": entry["description"],
        "category": entry["category"],
    }


def list_faust_instruments() -> list[dict[str, str]]:
    """Flat catalog entries for the instrument picker library column."""
    return [_instrument_summary(entry) for entry in _FAUST_INSTRUMENTS]


def list_faust_library_categories() -> list[dict]:
    """Catalog grouped by patch type for the picker UI."""
    by_category: dict[str, list[dict[str, str]]] = {cat["id"]: [] for cat in _FAUST_CATEGORIES}
    for entry in _FAUST_INSTRUMENTS:
        by_category.setdefault(entry["category"], []).append(_instrument_summary(entry))
    grouped: list[dict] = []
    for category in _FAUST_CATEGORIES:
        instruments = by_category.get(category["id"], [])
        if not instruments:
            continue
        grouped.append(
            {
                "id": category["id"],
                "label": category["label"],
                "instruments": instruments,
            }
        )
    return grouped


def _catalog_entry(instrument_id: str) -> dict[str, str]:
    needle = (instrument_id or "").strip()
    for entry in _FAUST_INSTRUMENTS:
        if entry["id"] == needle:
            return entry
    raise ValueError(f"Unknown Faust library instrument: {instrument_id or '(empty)'}")


def resolve_faust_dsp_path(instrument_id: str) -> str:
    """Absolute path to a bundled ``.dsp`` file."""
    entry = _catalog_entry(instrument_id)
    path = bundled_asset_path("faust", entry["file"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Faust instrument file not found: {path}")
    return str(path)


def faust_instrument_exists(instrument_id: str) -> bool:
    try:
        resolve_faust_dsp_path(instrument_id)
    except (ValueError, FileNotFoundError):
        return False
    return True


def faust_instrument_path_exists(path: str) -> bool:
    if not is_faust_instrument_path(path):
        return False
    return faust_instrument_exists(faust_id_from_path(path))


def load_faust_instrument(engine: Any, instrument_id: str, *, name: str | None = None) -> Any:
    """Compile and return a polyphonic Faust processor for ``instrument_id``."""
    from dronmakr.audio.audio_host import _unique_processor_name

    dsp_path = resolve_faust_dsp_path(instrument_id)
    proc_name = name or _unique_processor_name("faust")
    processor = engine.make_faust_processor(proc_name)
    processor.num_voices = FAUST_POLYPHONY_VOICES
    processor.set_dsp(os.path.abspath(dsp_path))
    processor.compile()
    return processor
