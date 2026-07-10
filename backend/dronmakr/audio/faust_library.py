"""Built-in Faust instrument library for drone generation."""

from __future__ import annotations

import os
from typing import Any

from dronmakr.core.bundle_paths import bundled_asset_path

FAUST_INSTRUMENT_PREFIX = "faust:"
FAUST_POLYPHONY_VOICES = 16
FAUST_INSTRUMENT_INPUT_CHANNELS = 0
FAUST_INSTRUMENT_OUTPUT_CHANNELS = 2

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
    # Oscillators (18)
    {"id": "buzz_stack", "category": "oscillators", "label": "Buzz Stack", "description": "Aggressive saw and square blend.", "file": "buzz_stack.dsp"},
    {"id": "crystal_osc", "category": "oscillators", "label": "Crystal Osc", "description": "Light FM shimmer on a sine core.", "file": "crystal_osc.dsp"},
    {"id": "dual_saw", "category": "oscillators", "label": "Dual Saw", "description": "Detuned saw pair for wide analog-style leads.", "file": "dual_saw.dsp"},
    {"id": "golden_ratio", "category": "oscillators", "label": "Golden Ratio", "description": "Partial stack tuned to golden-ratio intervals.", "file": "golden_ratio.dsp"},
    {"id": "harmonic_stack", "category": "oscillators", "label": "Harmonic Stack", "description": "Rich additive harmonic stack for warm tones.", "file": "harmonic_stack.dsp"},
    {"id": "hollow_form", "category": "oscillators", "label": "Hollow Form", "description": "Odd-harmonic square-ish tone with space.", "file": "hollow_form.dsp"},
    {"id": "iron_saw", "category": "oscillators", "label": "Iron Saw", "description": "Hard clipped saw stack for industrial tones.", "file": "iron_saw.dsp"},
    {"id": "octave_fifth", "category": "oscillators", "label": "Octave Fifth", "description": "Root, octave, and fifth blend for power chords.", "file": "octave_fifth.dsp"},
    {"id": "phase_weave", "category": "oscillators", "label": "Phase Weave", "description": "Interleaved triangle layers with motion.", "file": "phase_weave.dsp"},
    {"id": "pulse_osc", "category": "oscillators", "label": "Pulse Oscillator", "description": "Variable-width pulse wave with a short ADSR envelope.", "file": "pulse_osc.dsp"},
    {"id": "pwm_shimmer", "category": "oscillators", "label": "PWM Shimmer", "description": "Pulse wave with slow width modulation.", "file": "pwm_shimmer.dsp"},
    {"id": "saw_osc", "category": "oscillators", "label": "Saw Oscillator", "description": "Bright sawtooth wave with a short ADSR envelope.", "file": "saw_osc.dsp"},
    {"id": "sine_osc", "category": "oscillators", "label": "Sine Oscillator", "description": "Warm sine wave with a short ADSR envelope.", "file": "sine_osc.dsp"},
    {"id": "square_osc", "category": "oscillators", "label": "Square Oscillator", "description": "Hollow square wave with a short ADSR envelope.", "file": "square_osc.dsp"},
    {"id": "sub_stack", "category": "oscillators", "label": "Sub Stack", "description": "Saw layered with a sub-octave for heavy low-end.", "file": "sub_stack.dsp"},
    {"id": "supersaw", "category": "oscillators", "label": "Supersaw", "description": "Triple detuned saws for a massive unison lead.", "file": "supersaw.dsp"},
    {"id": "triangle_osc", "category": "oscillators", "label": "Triangle Oscillator", "description": "Soft triangle wave with a short ADSR envelope.", "file": "triangle_osc.dsp"},
    {"id": "wire_tone", "category": "oscillators", "label": "Wire Tone", "description": "Thin, tense high square with bite.", "file": "wire_tone.dsp"},
    # Subtractive (12)
    {"id": "acid_squawk", "category": "subtractive", "label": "Acid Squawk", "description": "Snappy resonant square with a squelchy filter.", "file": "acid_squawk.dsp"},
    {"id": "banana_bass", "category": "subtractive", "label": "Banana Bass", "description": "Rubbery resonant bass squelch.", "file": "banana_bass.dsp"},
    {"id": "cigarette_filter", "category": "subtractive", "label": "Cigarette Filter", "description": "Smoky low-pass triangle murmur.", "file": "cigarette_filter.dsp"},
    {"id": "dark_ladder", "category": "subtractive", "label": "Dark Ladder", "description": "Deep resonant ladder filter on a square.", "file": "dark_ladder.dsp"},
    {"id": "filter_arc", "category": "subtractive", "label": "Filter Arc", "description": "Sweeping resonant triangle with bite.", "file": "filter_arc.dsp"},
    {"id": "formant_box", "category": "subtractive", "label": "Formant Box", "description": "Formant-like band-pass vocal filter.", "file": "formant_box.dsp"},
    {"id": "lowpass_saw", "category": "subtractive", "label": "Lowpass Saw", "description": "Resonant low-pass filtered saw for mellow tones.", "file": "lowpass_saw.dsp"},
    {"id": "notch_scrape", "category": "subtractive", "label": "Notch Scrape", "description": "Band-pass saw scrape with a vocal edge.", "file": "notch_scrape.dsp"},
    {"id": "rust_gate", "category": "subtractive", "label": "Rust Gate", "description": "Gritty filtered square with a rusty edge.", "file": "rust_gate.dsp"},
    {"id": "talkbox_saw", "category": "subtractive", "label": "Talkbox Saw", "description": "Mid-focused resonant band on a bright saw.", "file": "talkbox_saw.dsp"},
    {"id": "velvet_ladder", "category": "subtractive", "label": "Velvet Ladder", "description": "Smooth resonant low-pass on a soft saw.", "file": "velvet_ladder.dsp"},
    {"id": "warm_triangle", "category": "subtractive", "label": "Warm Triangle", "description": "Soft triangle through a gentle low-pass.", "file": "warm_triangle.dsp"},
    # Pads & Strings (14)
    {"id": "afterglow_pad", "category": "pads", "label": "Afterglow Pad", "description": "Warm post-sunset glow with long tail.", "file": "afterglow_pad.dsp"},
    {"id": "aurora_pad", "category": "pads", "label": "Aurora Pad", "description": "Bright triangle and saw aurora blend.", "file": "aurora_pad.dsp"},
    {"id": "cathedral_pad", "category": "pads", "label": "Cathedral Pad", "description": "Organ-like sustained cathedral wash.", "file": "cathedral_pad.dsp"},
    {"id": "choir_cloud", "category": "pads", "label": "Choir Cloud", "description": "Stacked detuned sines for a vocal cloud.", "file": "choir_cloud.dsp"},
    {"id": "ghost_strings", "category": "pads", "label": "Ghost Strings", "description": "Ethereal bowed-string style detuned saws.", "file": "ghost_strings.dsp"},
    {"id": "glass_pad", "category": "pads", "label": "Glass Pad", "description": "Bright, airy pad with a slow bloom.", "file": "glass_pad.dsp"},
    {"id": "lunar_pad", "category": "pads", "label": "Lunar Pad", "description": "Slow-evolving pad with a drifting filter.", "file": "lunar_pad.dsp"},
    {"id": "memory_pad", "category": "pads", "label": "Memory Pad", "description": "Nostalgic pad with a very slow bloom.", "file": "memory_pad.dsp"},
    {"id": "mist_strings", "category": "pads", "label": "Mist Strings", "description": "Distant filtered string ensemble.", "file": "mist_strings.dsp"},
    {"id": "nebula_pad", "category": "pads", "label": "Nebula Pad", "description": "Slow-blooming detuned space pad.", "file": "nebula_pad.dsp"},
    {"id": "prism_pad", "category": "pads", "label": "Prism Pad", "description": "Shimmering harmonic prism pad.", "file": "prism_pad.dsp"},
    {"id": "silk_winds", "category": "pads", "label": "Silk Winds", "description": "Airy detuned sine choir in motion.", "file": "silk_winds.dsp"},
    {"id": "velvet_pad", "category": "pads", "label": "Velvet Pad", "description": "Dark filtered saw pad with a long tail.", "file": "velvet_pad.dsp"},
    {"id": "winter_hall", "category": "pads", "label": "Winter Hall", "description": "Cold spacious hall pad with air.", "file": "winter_hall.dsp"},
    # Noise & Texture (12)
    {"id": "ash_rain", "category": "texture", "label": "Ash Rain", "description": "Granular rain-like noise particles.", "file": "ash_rain.dsp"},
    {"id": "dust_bed", "category": "texture", "label": "Dust Bed", "description": "Low rumbling noise bed for texture layers.", "file": "dust_bed.dsp"},
    {"id": "ember_field", "category": "texture", "label": "Ember Field", "description": "Crackling ember field with mid sparkle.", "file": "ember_field.dsp"},
    {"id": "fog_bank", "category": "texture", "label": "Fog Bank", "description": "Wide soft noise fog with slow motion.", "file": "fog_bank.dsp"},
    {"id": "granular_cloud", "category": "texture", "label": "Granular Cloud", "description": "Tremolo noise cloud with drifting motion.", "file": "granular_cloud.dsp"},
    {"id": "pink_wind", "category": "texture", "label": "Pink Wind", "description": "Breathy band-pass noise for airy motion.", "file": "pink_wind.dsp"},
    {"id": "radio_static", "category": "texture", "label": "Radio Static", "description": "Tuned static bursts through a band-pass.", "file": "radio_static.dsp"},
    {"id": "sandstorm", "category": "texture", "label": "Sandstorm", "description": "Dry granular sandstorm texture.", "file": "sandstorm.dsp"},
    {"id": "static_drift", "category": "texture", "label": "Static Drift", "description": "Crackly noise with a wandering band-pass.", "file": "static_drift.dsp"},
    {"id": "steam_hiss", "category": "texture", "label": "Steam Hiss", "description": "Airy steam hiss through a moving band-pass.", "file": "steam_hiss.dsp"},
    {"id": "thunder_roll", "category": "texture", "label": "Thunder Roll", "description": "Low rolling noise swell with weight.", "file": "thunder_roll.dsp"},
    {"id": "vinyl_crackle", "category": "texture", "label": "Vinyl Crackle", "description": "Lo-fi crackle and hiss texture.", "file": "vinyl_crackle.dsp"},
    # Plucks & Mallets (12)
    {"id": "bamboo_hit", "category": "plucks", "label": "Bamboo Hit", "description": "Dry bamboo knock with a woody tone.", "file": "bamboo_hit.dsp"},
    {"id": "glass_pluck", "category": "plucks", "label": "Glass Pluck", "description": "Brittle glassy pluck with fast decay.", "file": "glass_pluck.dsp"},
    {"id": "guitar_mute", "category": "plucks", "label": "Guitar Mute", "description": "Muted guitar pluck with short body.", "file": "guitar_mute.dsp"},
    {"id": "hammer_dulcimer", "category": "plucks", "label": "Hammer Dulcimer", "description": "Hammered string shimmer with harmonics.", "file": "hammer_dulcimer.dsp"},
    {"id": "harp_glide", "category": "plucks", "label": "Harp Glide", "description": "Longer harp-like pluck with shimmer.", "file": "harp_glide.dsp"},
    {"id": "harpsichord_tick", "category": "plucks", "label": "Harpsichord Tick", "description": "Bright, percussive square tick with fast decay.", "file": "harpsichord_tick.dsp"},
    {"id": "kalimba_twinkle", "category": "plucks", "label": "Kalimba Twinkle", "description": "Bright kalimba-like FM twinkle.", "file": "kalimba_twinkle.dsp"},
    {"id": "mallet_fm", "category": "plucks", "label": "Mallet FM", "description": "Percussive two-operator FM mallet.", "file": "mallet_fm.dsp"},
    {"id": "marimba_bloom", "category": "plucks", "label": "Marimba Bloom", "description": "Wooden marimba bloom with FM body.", "file": "marimba_bloom.dsp"},
    {"id": "soft_pluck", "category": "plucks", "label": "Soft Pluck", "description": "Short decaying sine pluck for gentle motion.", "file": "soft_pluck.dsp"},
    {"id": "thumb_piano", "category": "plucks", "label": "Thumb Piano", "description": "Warm thumb piano pluck with body.", "file": "thumb_piano.dsp"},
    {"id": "tick_tock", "category": "plucks", "label": "Tick Tock", "description": "Tiny percussive tick for rhythmic motion.", "file": "tick_tock.dsp"},
    # Keys & Organs (12)
    {"id": "accordion_reed", "category": "keys", "label": "Accordion Reed", "description": "Wheezy accordion reed with detune.", "file": "accordion_reed.dsp"},
    {"id": "celeste", "category": "keys", "label": "Celeste", "description": "Delicate celeste with a high shimmer.", "file": "celeste.dsp"},
    {"id": "church_bell", "category": "keys", "label": "Church Bell", "description": "Large church bell strike with long tail.", "file": "church_bell.dsp"},
    {"id": "clavinet_bite", "category": "keys", "label": "Clavinet Bite", "description": "Snappy clavinet-like filtered square.", "file": "clavinet_bite.dsp"},
    {"id": "drawbar_organ", "category": "keys", "label": "Drawbar Organ", "description": "Classic additive organ tone with drawbar harmonics.", "file": "drawbar_organ.dsp"},
    {"id": "fm_bell", "category": "keys", "label": "FM Bell", "description": "Bright bell partials from frequency modulation.", "file": "fm_bell.dsp"},
    {"id": "harpsichord_lute", "category": "keys", "label": "Harpsichord Lute", "description": "Plucked lute-harpsichord hybrid tone.", "file": "harpsichord_lute.dsp"},
    {"id": "music_box", "category": "keys", "label": "Music Box", "description": "Gentle music box FM tone.", "file": "music_box.dsp"},
    {"id": "pipe_organ", "category": "keys", "label": "Pipe Organ", "description": "Pipe organ tone with strong fundamentals.", "file": "pipe_organ.dsp"},
    {"id": "reed_wind", "category": "keys", "label": "Reed Wind", "description": "Nasal reed tone from pulse and triangle mix.", "file": "reed_wind.dsp"},
    {"id": "toy_piano", "category": "keys", "label": "Toy Piano", "description": "Simple bright toy piano tone.", "file": "toy_piano.dsp"},
    {"id": "wurlitzer_bite", "category": "keys", "label": "Wurlitzer Bite", "description": "Electric piano bite with a mid focus.", "file": "wurlitzer_bite.dsp"},
    # Metallic & FM (12)
    {"id": "anvil_clang", "category": "metallic", "label": "Anvil Clang", "description": "Heavy anvil clang with dense partials.", "file": "anvil_clang.dsp"},
    {"id": "bronze_reson", "category": "metallic", "label": "Bronze Reson", "description": "Resonant FM tone with a bronze-like ring.", "file": "bronze_reson.dsp"},
    {"id": "clang_stack", "category": "metallic", "label": "Clang Stack", "description": "Inharmonic partial stack for metallic clangs.", "file": "clang_stack.dsp"},
    {"id": "copper_bowl", "category": "metallic", "label": "Copper Bowl", "description": "Singing copper bowl with slow decay.", "file": "copper_bowl.dsp"},
    {"id": "crystal_fm", "category": "metallic", "label": "Crystal FM", "description": "Glassy high-ratio FM with a sparkling decay.", "file": "crystal_fm.dsp"},
    {"id": "gong_strike", "category": "metallic", "label": "Gong Strike", "description": "Large gong with a long metallic bloom.", "file": "gong_strike.dsp"},
    {"id": "ring_mod_ghost", "category": "metallic", "label": "Ring Mod Ghost", "description": "Haunted ring-mod-like partial clash.", "file": "ring_mod_ghost.dsp"},
    {"id": "shard_glass", "category": "metallic", "label": "Shard Glass", "description": "Sharp glass shard partial burst.", "file": "shard_glass.dsp"},
    {"id": "silver_fm", "category": "metallic", "label": "Silver FM", "description": "Silvery FM tone with a bright index.", "file": "silver_fm.dsp"},
    {"id": "steel_drum", "category": "metallic", "label": "Steel Drum", "description": "Caribbean steel drum FM strike.", "file": "steel_drum.dsp"},
    {"id": "tin_hat", "category": "metallic", "label": "Tin Hat", "description": "Bright tinny inharmonic stack.", "file": "tin_hat.dsp"},
    {"id": "wire_resonance", "category": "metallic", "label": "Wire Resonance", "description": "Taut wire resonance with harmonic tension.", "file": "wire_resonance.dsp"},
    # Drones & Beds (13)
    {"id": "abyss_rumble", "category": "drones", "label": "Abyss Rumble", "description": "Sub-heavy drone with filtered noise undertow.", "file": "abyss_rumble.dsp"},
    {"id": "aurora_drone", "category": "drones", "label": "Aurora Drone", "description": "Northern lights drone with slow shimmer.", "file": "aurora_drone.dsp"},
    {"id": "cave_pool", "category": "drones", "label": "Cave Pool", "description": "Dark cave pool with dripping noise hints.", "file": "cave_pool.dsp"},
    {"id": "deep_current", "category": "drones", "label": "Deep Current", "description": "Underwater current with sub motion.", "file": "deep_current.dsp"},
    {"id": "ether_void", "category": "drones", "label": "Ether Void", "description": "Minimal long sine void with vast space.", "file": "ether_void.dsp"},
    {"id": "horizon_hum", "category": "drones", "label": "Horizon Hum", "description": "Distant dual-sub horizon hum.", "file": "horizon_hum.dsp"},
    {"id": "magma_flow", "category": "drones", "label": "Magma Flow", "description": "Hot magma drone with slow filter motion.", "file": "magma_flow.dsp"},
    {"id": "night_train", "category": "drones", "label": "Night Train", "description": "Rhythmic pulsing drone like distant rails.", "file": "night_train.dsp"},
    {"id": "ocean_floor", "category": "drones", "label": "Ocean Floor", "description": "Deep sine undertone with gentle noise swell.", "file": "ocean_floor.dsp"},
    {"id": "pulse_drift", "category": "drones", "label": "Pulse Drift", "description": "Slowly pulsing filtered square drone.", "file": "pulse_drift.dsp"},
    {"id": "sleep_signal", "category": "drones", "label": "Sleep Signal", "description": "Hypnotic low pulse bed for deep drones.", "file": "sleep_signal.dsp"},
    {"id": "solar_haze", "category": "drones", "label": "Solar Haze", "description": "Wide detuned pad with a soft noise halo.", "file": "solar_haze.dsp"},
    {"id": "tectonic", "category": "drones", "label": "Tectonic", "description": "Slow sub tectonic rumble with weight.", "file": "tectonic.dsp"},
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


def faust_instrument_preset_for_id(instrument_id: str) -> dict[str, str]:
    """Preset-shaped dict for generate_drone_sample when using a built-in Faust instrument."""
    entry = _catalog_entry(instrument_id)
    return {
        "plugin_path": faust_path_for_id(entry["id"]),
        "preset_path": "",
        "name": entry.get("label") or entry["id"],
        "plugin_name": "Faust",
    }


def pick_random_faust_instrument_preset() -> dict[str, str]:
    """Pick a random shipped Faust instrument in preset dict form."""
    import random

    entry = random.choice(_FAUST_INSTRUMENTS)
    return faust_instrument_preset_for_id(entry["id"])


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


def _assert_faust_instrument_io(processor: Any, instrument_id: str) -> None:
    inputs = int(processor.get_num_input_channels())
    outputs = int(processor.get_num_output_channels())
    if inputs != FAUST_INSTRUMENT_INPUT_CHANNELS or outputs != FAUST_INSTRUMENT_OUTPUT_CHANNELS:
        raise ValueError(
            f"Faust instrument '{instrument_id}' exposes {inputs} input(s) and {outputs} output(s); "
            f"expected {FAUST_INSTRUMENT_INPUT_CHANNELS} in / {FAUST_INSTRUMENT_OUTPUT_CHANNELS} out "
            f"(stereo synth with freq/gain/gate and `process = ... <: _, _;`)."
        )
    if int(processor.num_voices) <= 0:
        raise ValueError(
            f"Faust instrument '{instrument_id}' has polyphony disabled (num_voices={processor.num_voices})."
        )


def load_faust_instrument(engine: Any, instrument_id: str, *, name: str | None = None) -> Any:
    """Compile and return a polyphonic Faust processor for ``instrument_id``."""
    from dronmakr.audio.audio_host import _unique_processor_name

    dsp_path = resolve_faust_dsp_path(instrument_id)
    proc_name = name or _unique_processor_name("faust")
    processor = engine.make_faust_processor(proc_name)
    processor.set_dsp(os.path.abspath(dsp_path))
    processor.num_voices = FAUST_POLYPHONY_VOICES
    processor.compile()
    _assert_faust_instrument_io(processor, instrument_id)
    return processor
