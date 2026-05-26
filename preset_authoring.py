"""Pedalboard preset authoring shared by Patchcraftr GUI and CLI helpers."""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

from mido import Message
import pedalboard
from pedalboard import Pedalboard
from pedalboard.io import AudioFile

from generate_midi import SUPPORTED_PATTERNS_INFO
from settings import get_setting
from utils import (
    MAGENTA,
    RESET,
    generate_id,
    PRESETS_DIR,
    PRESETS_PATH,
    TEMP_DIR,
    with_patchcraftr_prompt as with_prompt,
)

PREVIEW_NUM_BARS = 1
PREVIEW_SAMPLE_RATE = 44100
PREVIEW_TEMPO_BPM = 120
PREVIEW_TIME_SIGNATURE = "4/4"

_REPO_ROOT = Path(__file__).resolve().parent


def preview_sample_wav_path() -> str:
    p = _REPO_ROOT / "resources" / "CDEFGABC.wav"
    return str(p)


PREVIEW_TEMP_PATH = os.path.join(TEMP_DIR, "preset_preview.wav")

MAX_CHAIN_SLOTS = 5


class PluginVariantRequired(Exception):
    """Multi-factory VST3 bundle — caller must choose ``variant_name`` and reload."""

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


def calculate_beat_info(tempo_bpm: str | float, time_signature: str) -> tuple[int, float]:
    beats_per_bar, _beat_type = map(int, time_signature.split("/"))
    beat_duration_s = 60 / float(tempo_bpm)
    return beats_per_bar, beat_duration_s


def calculate_audio_length(tempo_bpm: str | float, time_signature: str, num_bars: int) -> float:
    beats_per_bar, beat_duration_s = calculate_beat_info(tempo_bpm, time_signature)
    total_beats = beats_per_bar * num_bars
    return total_beats * beat_duration_s


def list_installed_plugins(plugin_dirs: list[str], custom_plugins: list[str]) -> list[str]:
    paths = list(custom_plugins)
    for plugin_dir in plugin_dirs:
        plugin_dir = plugin_dir.strip()
        if os.path.exists(plugin_dir):
            paths.extend(glob.glob(os.path.join(plugin_dir, "*.vst3")))
            # macOS: Pedalboard loads VST3 and Audio Unit only — not legacy VST2 (.vst bundles).
            if sys.platform != "darwin":
                paths.extend(glob.glob(os.path.join(plugin_dir, "*.vst")))
            paths.extend(glob.glob(os.path.join(plugin_dir, "*.dll")))
            paths.extend(glob.glob(os.path.join(plugin_dir, "*.so")))
            paths.extend(glob.glob(os.path.join(plugin_dir, "*.component")))
    return paths


def macos_pedalboard_rejects_vst2_path(plugin_path: str) -> bool:
    """True if ``plugin_path`` is a legacy macOS VST2 bundle path (``.vst`` but not ``.vst3``)."""
    if sys.platform != "darwin":
        return False
    base = plugin_path.rstrip("/")
    low = base.lower()
    # Guard against odd names ending in ".vstsomething"
    return low.endswith(".vst") and not low.endswith(".vst3")


def format_plugin_name(plugin_path: str) -> str:
    return (
        os.path.basename(plugin_path)
        .replace(".vst3", "")
        .replace(".vst", "")
        .replace(".component", " (AU)")
    )


def generate_preview_midi():
    audio_length_s = calculate_audio_length(
        PREVIEW_TEMPO_BPM, PREVIEW_TIME_SIGNATURE, PREVIEW_NUM_BARS
    )
    scale_notes = [60, 62, 64, 65, 67, 69, 71, 72]
    midi_messages = []
    for index, note in enumerate(scale_notes):
        note_length_s = audio_length_s / len(scale_notes)
        midi_messages.append(
            Message("note_on", note=note, time=(index * note_length_s))
        )
        midi_messages.append(
            Message("note_off", note=note, time=((index + 1) * note_length_s))
        )
    empty_bar_length = calculate_audio_length(
        PREVIEW_TEMPO_BPM, PREVIEW_TIME_SIGNATURE, 1
    )
    audio_length_s += empty_bar_length
    return midi_messages, audio_length_s


def _parse_variants_from_load_error(error_message: str) -> list[str]:
    return [
        line.strip().strip('"')
        for line in error_message.split("\n")
        if line.startswith('\t"')
    ]


def load_pedalboard_plugin(plugin_path: str, plugin_name: str | None = None):
    """Return ``(plugin, effective_plugin_name)`` or raise ``PluginVariantRequired``."""
    if macos_pedalboard_rejects_vst2_path(plugin_path):
        raise ValueError(
            "On macOS, dronmakr uses Spotify Pedalboard, which only loads VST3 (`.vst3`) and "
            "Audio Unit (`.component`) plug-ins — not legacy VST2 bundles (`.vst`) such as:\n\n"
            f"  {plugin_path}\n\n"
            "Use Vital’s VST3 under Library/Audio/Plug-Ins/VST3/, or install the AU under "
            "Library/Audio/Plug-Ins/Components/ and point PLUGIN_PATHS there. "
            "You can remove `.../Plug-Ins/VST` from PLUGIN_PATHS in Settings to hide incompatible entries."
        ) from None
    try:
        if plugin_name:
            return pedalboard.load_plugin(plugin_path, plugin_name=plugin_name), plugin_name
        return pedalboard.load_plugin(plugin_path), ""
    except ValueError as e:
        error_message = str(e)
        if "contains" in error_message and "To open a specific plugin" in error_message:
            variants = _parse_variants_from_load_error(error_message)
            if not variants:
                raise
            raise PluginVariantRequired(plugin_path, variants) from e
        raise


def serialize_plugin_preset_bytes(plugin) -> bytes:
    """Stable binary snapshot compatible with ``apply_vstpreset_bytes_to_plugin``."""
    blob = plugin.preset_data if hasattr(plugin, "preset_data") else plugin.raw_state
    return bytes(blob)


def apply_vstpreset_bytes_to_plugin(plugin, preset_bytes: bytes) -> None:
    if hasattr(plugin, "preset_data"):
        plugin.preset_data = preset_bytes
    else:
        plugin.raw_state = preset_bytes


def reload_pedalboard_plugin_preserving_state(
    plugin_path: str, plugin_name_hint: str | None, old_plugin
):
    """Fresh ``load_plugin`` instance + prior preset blob.

    Some VST3 instruments stop responding to background MIDI previews after ``show_editor``
    is closed once; re-instantiating the processor restores normal behaviour.
    """
    blob = serialize_plugin_preset_bytes(old_plugin)
    hint = plugin_name_hint.strip() if (plugin_name_hint and plugin_name_hint.strip()) else None
    plug, resolved_name = load_pedalboard_plugin(plugin_path, hint)
    effective_name = (
        plugin_name_hint.strip()
        if (plugin_name_hint and plugin_name_hint.strip())
        else resolved_name
    )
    apply_vstpreset_bytes_to_plugin(plug, blob)
    return plug, effective_name


def write_plugin_state_to_vstpreset(preset_path: str, plugin) -> None:
    os.makedirs(os.path.dirname(preset_path) or ".", exist_ok=True)
    with open(preset_path, "wb") as f:
        if hasattr(plugin, "preset_data"):
            f.write(plugin.preset_data)
        else:
            f.write(plugin.raw_state)


def preview_plugin(plugin, effect_chain_tuples: list):
    """``effect_chain_tuples`` shape: ``(path, name, plugin_instance, meta_tuple)``."""
    num_channels = 2
    sample = preview_sample_wav_path()
    if not os.path.isfile(sample):
        raise FileNotFoundError(f"Missing preview WAV: {sample}")

    if plugin.is_instrument:
        midi_messages, audio_length_s = generate_preview_midi()
        pre_fx_signal = plugin(
            midi_messages,
            duration=audio_length_s,
            sample_rate=PREVIEW_SAMPLE_RATE,
            num_channels=num_channels,
            buffer_size=8192,
            reset=False,
        )
    elif plugin.is_effect:
        with AudioFile(sample, "r") as f:
            audio_length_s = f.frames / f.samplerate
            pre_fx_signal = f.read(f.frames)
        fx_chain = Pedalboard([fx[2] for fx in effect_chain_tuples])
        pre_fx_signal = fx_chain(pre_fx_signal, PREVIEW_SAMPLE_RATE, reset=False)
    else:
        return

    try:
        with AudioFile(PREVIEW_TEMP_PATH, "w", PREVIEW_SAMPLE_RATE, num_channels) as f:
            f.write(pre_fx_signal)
        if sys.platform == "darwin":
            subprocess.run(["afplay", PREVIEW_TEMP_PATH], check=False)
    finally:
        if os.path.isfile(PREVIEW_TEMP_PATH):
            try:
                os.remove(PREVIEW_TEMP_PATH)
            except OSError:
                pass


def load_presets_json() -> list:
    if not os.path.exists(PRESETS_PATH):
        return []
    with open(PRESETS_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_presets_json(presets_data: list) -> None:
    os.makedirs(os.path.dirname(PRESETS_PATH) or ".", exist_ok=True)
    with open(PRESETS_PATH, "w", encoding="utf-8") as f:
        json.dump(presets_data, f, indent=4)


def name_exists(
    name: str,
    *,
    exclude_uid: str | None = None,
    types: tuple[str, ...] | None = None,
) -> bool:
    if not os.path.exists(PRESETS_PATH):
        return False
    with open(PRESETS_PATH, "r", encoding="utf-8") as f:
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
    if not os.path.exists(PRESETS_PATH):
        print("No presets found.")
        return

    try:
        with open(PRESETS_PATH, "r", encoding="utf-8") as f:
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
    if plugin.is_effect and plugin_name not in assert_instrument:
        return True
    return False
