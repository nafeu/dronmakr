"""DawDreamer preset authoring shared by Generate Samples and CLI helpers."""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone

import dronmakr.audio.audio_host as audio_host
from dronmakr.audio.audio_host import (
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
from dronmakr.generate.generate_midi import SUPPORTED_PATTERNS_INFO
from dronmakr.core.paths import get_managed_file
from dronmakr.core.settings import get_setting
from dronmakr.core.utils import (
    MAGENTA,
    RESET,
    generate_id,
    PRESETS_DIR,
    resolve_presets_index_path,
    TEMP_DIR,
    with_main_prompt as with_prompt,
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


def plugin_settings_tuple() -> tuple[list[str], list[str]]:
    plugin_paths = get_setting("PLUGIN_PATHS", "").split(",")
    ignore_plugins = [x.strip() for x in get_setting("IGNORE_PLUGINS", "").split(",") if x.strip()]
    return plugin_paths, ignore_plugins


INSTRUMENT_ALLOW_LIST_KEY = "INSTRUMENT_ALLOW_LIST"
FX_ALLOW_LIST_KEY = "FX_ALLOW_LIST"


def _allowed_plugins_setting_key(role: str) -> str:
    role_norm = (role or "").strip().lower()
    if role_norm == "effect":
        return FX_ALLOW_LIST_KEY
    return INSTRUMENT_ALLOW_LIST_KEY


def get_allowed_plugin_labels(role: str) -> list[str]:
    key = _allowed_plugins_setting_key(role)
    raw = get_setting(key, "")
    if not isinstance(raw, str):
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def list_all_installed_plugin_entries() -> list[dict]:
    """Return every VST3/AU plug-in discovered under PLUGIN_PATHS (no role scan)."""
    plugin_paths, _ = plugin_settings_tuple()
    if not plugin_paths or plugin_paths == [""]:
        return []

    seen_paths: set[str] = set()
    entries: list[dict] = []
    for path in list_installed_plugins(plugin_paths):
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path) or abs_path in seen_paths:
            continue
        seen_paths.add(abs_path)
        label = format_plugin_name(abs_path)
        entries.append({"label": label, "path": abs_path, "displayLabel": label})
    entries.sort(key=lambda item: (item.get("label") or "").lower())
    return entries


def assert_plugin_paths_configured() -> None:
    plugin_paths, _ = plugin_settings_tuple()
    if not plugin_paths or plugin_paths == [""]:
        raise PresetAuthoringConfigError(
            "No VST paths in settings (PLUGIN_PATHS). Set them in Settings."
        )
    discovered = list_installed_plugins(plugin_paths)
    if not discovered:
        raise PresetAuthoringConfigError(
            "No plug-ins discovered under PLUGIN_PATHS."
        )


def build_plugin_label_map() -> dict[str, str]:
    plugin_paths, _ = plugin_settings_tuple()
    if not plugin_paths or plugin_paths == [""]:
        return {}
    return {
        entry["label"]: entry["path"]
        for entry in list_all_installed_plugin_entries()
    }


def plugin_label_is_ignored(label: str, ignore_plugins: list[str] | None = None) -> bool:
    ignores = ignore_plugins
    if ignores is None:
        _, ignores = plugin_settings_tuple()
    return bool(_ignore_tokens_matching_label(label, {token for token in ignores if token}))


def _ignore_tokens_matching_label(label: str, ignores: set[str]) -> set[str]:
    """Return ignore-list tokens that hide ``label`` (mirrors matching used on save)."""
    core = _plugin_label_core(label)
    matched: set[str] = set()
    for token in ignores:
        if not token:
            continue
        token_core = _plugin_label_core(token)
        if token == label or token_core == core:
            matched.add(token)
            continue
        if token in label or label in token:
            matched.add(token)
    return matched


_EFFECT_NAME_TOKENS = (
    " compressor",
    " comp",
    " reverb",
    " delay",
    " eq",
    " equalizer",
    " limiter",
    " gate",
    " de-ess",
    " saturator",
    " distort",
    " chorus",
    " flanger",
    " phaser",
    " tremolo",
    " filter",
    " utility",
    " analyzer",
    " meter",
    " maximizer",
    " imager",
    " exciter",
    " convolution",
    " shimmer",
    " tape",
    " grain",
    " pedal",
    " ozone",
    " transient",
    " widener",
    " auto-tune",
    " autotune",
    " vocoder",
    " enhancer",
    " balancer",
    " clipp",
)


def _plugin_label_core(label: str) -> str:
    text = (label or "").strip()
    return re.sub(r"\s+\(AU\)$", "", text, flags=re.IGNORECASE).strip()


def label_suggests_fx(label: str) -> bool:
    """True when the plug-in name explicitly marks an FX variant (e.g. Reaktor 6 FX, FM8 FX)."""
    core = _plugin_label_core(label).lower().strip()
    if core.endswith(" fx") or core.endswith("-fx") or core.endswith("_fx"):
        return True
    if " fx " in f" {core} ":
        return True
    # Bundles named FM8FX, Reaktor6FX, etc.
    if len(core) > 2 and core.endswith("fx") and not core.endswith(" fx"):
        return True
    return False


def plugin_base_name(label: str) -> str:
    """Strip FX suffixes so instrument/FX pairs sort and group together."""
    core = _plugin_label_core(label)
    low = core.lower().strip()
    for suffix in (" fx", "-fx", "_fx"):
        if low.endswith(suffix):
            return core[: -len(suffix)].strip()
    if len(low) > 2 and low.endswith("fx") and not low.endswith(" fx"):
        return core[:-2].strip()
    return core.strip()


def plugin_pair_key(name: str) -> str:
    """Normalized key for matching instrument/FX pairs (Reaktor 6, FM8, etc.)."""
    return re.sub(r"[\s._-]+", "", plugin_base_name(name).lower())


def _installed_plugin_labels() -> list[str]:
    plugin_paths, _ = plugin_settings_tuple()
    if not plugin_paths or plugin_paths == [""]:
        return []
    return [
        format_plugin_name(path)
        for path in list_installed_plugins(plugin_paths)
        if os.path.exists(path)
    ]


def _correct_plugin_role_from_label(
    label: str,
    role: str,
    *,
    sibling_labels: list[str] | None = None,
) -> str:
    """Apply label/pairing rules on top of cached or DawDreamer-derived roles."""
    if label_suggests_fx(label):
        return "effect"
    siblings = sibling_labels if sibling_labels is not None else _installed_plugin_labels()
    base = plugin_base_name(label).lower()
    if base and any(
        plugin_base_name(other).lower() == base and label_suggests_fx(other)
        for other in siblings
        if other != label
    ):
        return "instrument"
    heuristic = infer_plugin_role_from_label(label)
    if heuristic == "instrument" and role == "effect":
        return "instrument"
    return role


def reconcile_plugin_role(label: str, heuristic_role: str, processor_role: str) -> str:
    """Prefer explicit FX naming over DawDreamer I/O alone (some instruments expose inputs)."""
    if label_suggests_fx(label):
        return "effect"
    if heuristic_role == "instrument" and processor_role == "effect":
        return "instrument"
    if heuristic_role == "effect" and processor_role == "instrument":
        return "instrument"
    if processor_role in ("instrument", "effect"):
        return processor_role
    if heuristic_role in ("instrument", "effect"):
        return heuristic_role
    return "unknown"


def infer_plugin_role_from_label(label: str) -> str:
    """Fast instrument/effect guess from plug-in display name (no DawDreamer load)."""
    if label_suggests_fx(label):
        return "effect"

    trimmed = _plugin_label_core(label)
    low = trimmed.lower()

    spaced = f" {low} "
    for token in _EFFECT_NAME_TOKENS:
        if token in spaced:
            return "effect"

    return "instrument"


def _plugin_entry_sort_key(entry: dict) -> tuple:
    base = plugin_base_name(entry.get("label") or "").lower()
    role_rank = 0 if entry.get("role") == "instrument" else 1
    return (base, role_rank, (entry.get("label") or "").lower())


def enrich_plugin_entries(entries: list[dict]) -> list[dict]:
    """Add displayLabel and sort instrument/FX pairs together (e.g. Reaktor 6 + Reaktor 6 FX)."""
    bases: dict[str, set[str]] = {}
    for entry in entries:
        base = plugin_base_name(entry.get("label") or "").lower()
        role = entry.get("role")
        if base and role in ("instrument", "effect"):
            bases.setdefault(base, set()).add(role)

    enriched: list[dict] = []
    for entry in entries:
        item = dict(entry)
        label = item.get("label") or ""
        base = plugin_base_name(label).lower()
        paired = len(bases.get(base, set())) > 1
        role = item.get("role")
        if paired and role == "instrument":
            item["displayLabel"] = f"{label} (Instrument)"
        elif paired and role == "effect":
            item["displayLabel"] = label if label_suggests_fx(label) else f"{label} (FX)"
        else:
            item["displayLabel"] = label
        enriched.append(item)

    enriched.sort(key=_plugin_entry_sort_key)
    return enriched


def assert_plugin_role_for_slot(plugin_path: str, expected_role: str) -> None:
    """Plug-in slot role is chosen by the user; no DawDreamer classification check."""
    del plugin_path, expected_role


def _classification_cache_path() -> str:
    return get_managed_file("config", "plugin-scan-cache.json")


def _migrate_legacy_classification_cache() -> None:
    new_path = _classification_cache_path()
    if os.path.isfile(new_path):
        return
    legacy_path = os.path.join(TEMP_DIR, "plugin-classifications.json")
    if not os.path.isfile(legacy_path):
        return
    try:
        os.makedirs(os.path.dirname(new_path) or ".", exist_ok=True)
        shutil.copy2(legacy_path, new_path)
    except OSError:
        pass


def _load_classification_cache() -> dict:
    _migrate_legacy_classification_cache()
    path = _classification_cache_path()
    if not os.path.isfile(path):
        return {"plugins": {}, "complete": False, "version": 1}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"plugins": {}, "complete": False, "version": 1}
    if not isinstance(data, dict):
        return {"plugins": {}, "complete": False, "version": 1}
    plugins = data.get("plugins")
    if not isinstance(plugins, dict):
        data["plugins"] = {}
    if "complete" not in data:
        data["complete"] = False
    if "version" not in data:
        data["version"] = 1
    return data


def _installed_plugin_paths() -> list[str]:
    plugin_paths, _ = plugin_settings_tuple()
    if not plugin_paths or plugin_paths == [""]:
        return []
    return [
        path
        for path in list_installed_plugins(plugin_paths)
        if os.path.exists(path)
    ]


def get_plugin_scan_cache_info() -> dict:
    """Return plug-in list metadata for the UI (filesystem scan only)."""
    installed = list_all_installed_plugin_entries()
    count = len(installed)
    return {
        "complete": True,
        "scannedAt": "",
        "verifiedCount": count,
        "installedCount": count,
        "cachePath": "",
    }


def plugin_scan_cache_needs_initial_scan() -> bool:
    return False


def _save_classification_cache(data: dict) -> None:
    os.makedirs(os.path.dirname(_classification_cache_path()) or ".", exist_ok=True)
    with open(_classification_cache_path(), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _plugin_mtime(plugin_path: str) -> float:
    try:
        return float(os.path.getmtime(plugin_path))
    except OSError:
        return 0.0


def _scan_progress_path() -> str:
    return os.path.join(TEMP_DIR, "plugin-classification-progress.json")


def write_plugin_scan_progress(
    *,
    status: str,
    done: int = 0,
    total: int = 0,
    label: str = "",
) -> None:
    os.makedirs(TEMP_DIR, exist_ok=True)
    payload = {
        "status": status,
        "done": max(0, int(done)),
        "total": max(0, int(total)),
        "label": label or "",
    }
    with open(_scan_progress_path(), "w", encoding="utf-8") as f:
        json.dump(payload, f)


def read_plugin_scan_progress() -> dict:
    path = _scan_progress_path()
    if not os.path.isfile(path):
        return {"status": "idle", "done": 0, "total": 0, "label": ""}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"status": "idle", "done": 0, "total": 0, "label": ""}
    if not isinstance(data, dict):
        return {"status": "idle", "done": 0, "total": 0, "label": ""}
    return {
        "status": str(data.get("status") or "idle"),
        "done": max(0, int(data.get("done") or 0)),
        "total": max(0, int(data.get("total") or 0)),
        "label": str(data.get("label") or ""),
    }


def resolve_plugin_role(
    plugin_path: str,
    cache: dict | None = None,
    *,
    allow_slow_load: bool = False,
) -> str:
    """Resolve instrument/effect role using cache, name heuristics, optional DawDreamer load."""
    raw_path = (plugin_path or "").strip()
    from dronmakr.audio.faust_library import (
        faust_id_from_path,
        faust_instrument_exists,
        is_faust_instrument_path,
    )

    if is_faust_instrument_path(raw_path):
        return "instrument" if faust_instrument_exists(faust_id_from_path(raw_path)) else "unknown"

    plugin_path = os.path.abspath(raw_path)
    if not plugin_path or not os.path.exists(plugin_path):
        return "unknown"

    label = format_plugin_name(plugin_path)
    heuristic_role = infer_plugin_role_from_label(label)
    sibling_labels = _installed_plugin_labels()

    cache_data = cache if cache is not None else _load_classification_cache()
    plugins_cache = cache_data.setdefault("plugins", {})
    mtime = _plugin_mtime(plugin_path)
    cached = plugins_cache.get(plugin_path)
    if isinstance(cached, dict) and cached.get("mtime") == mtime:
        cached_role = cached.get("role")
        if cached.get("verified") and cached_role in ("instrument", "effect"):
            return _correct_plugin_role_from_label(
                label, str(cached_role), sibling_labels=sibling_labels
            )
        if cached_role in ("instrument", "effect") and not allow_slow_load:
            return _correct_plugin_role_from_label(
                label, str(cached_role), sibling_labels=sibling_labels
            )

    if not allow_slow_load:
        return _correct_plugin_role_from_label(
            label, heuristic_role, sibling_labels=sibling_labels
        )

    role = heuristic_role
    try:
        engine = create_engine()
        processor = _load_plugin_on_engine(engine, plugin_path)
        processor_role = "instrument" if plugin_is_instrument(processor) else "effect"
        role = reconcile_plugin_role(label, heuristic_role, processor_role)
    except Exception:
        role = heuristic_role

    role = _correct_plugin_role_from_label(label, role, sibling_labels=sibling_labels)

    plugins_cache[plugin_path] = {
        "mtime": mtime,
        "role": role,
        "label": label,
        "verified": True,
    }
    if cache is None:
        _save_classification_cache(cache_data)
    return role


def classify_plugin_path(plugin_path: str, *, cache: dict | None = None) -> str:
    """Return ``instrument``, ``effect``, or ``unknown`` for an installed plug-in path."""
    return resolve_plugin_role(plugin_path, cache, allow_slow_load=True)


def scan_plugin_classifications(*, force: bool = False) -> dict:
    """No-op: plug-in lists come from PLUGIN_PATHS only."""
    del force
    installed = len(list_all_installed_plugin_entries())
    write_plugin_scan_progress(status="done", done=installed, total=installed)
    return {"plugins": {}, "complete": True, "version": 1}


def list_installed_plugin_entries(*, role: str | None = None, respect_ignore: bool = True) -> list[dict]:
    """Return installed plug-ins, optionally filtered by per-role allow list."""
    del respect_ignore  # legacy name; allow-list filtering replaces IGNORE_PLUGINS
    entries = list_all_installed_plugin_entries()
    role_norm = (role or "").strip().lower()
    if role_norm not in ("instrument", "effect"):
        return entries
    allowed = get_allowed_plugin_labels(role_norm)
    if not allowed:
        return entries
    allowed_set = set(allowed)
    return [entry for entry in entries if entry.get("label") in allowed_set]


def save_allowed_plugins_for_role(role: str, allowed_labels: list[str]) -> str:
    """Persist the allow list for instrument or FX pickers."""
    from dronmakr.core.settings import load_settings, save_settings

    role_norm = (role or "").strip().lower()
    if role_norm not in ("instrument", "effect"):
        raise ValueError("role must be instrument or effect")

    key = _allowed_plugins_setting_key(role_norm)
    labels = sorted({str(label).strip() for label in allowed_labels if str(label).strip()})
    new_value = ",".join(labels)
    settings = load_settings()
    settings[key] = new_value
    settings["IGNORE_PLUGINS"] = ""
    save_settings(settings)
    return new_value


def list_installed_plugins(plugin_dirs: list[str]) -> list[str]:
    paths: list[str] = []
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


def _plugin_file_stem(plugin_path: str) -> str:
    base = os.path.basename((plugin_path or "").rstrip("/"))
    for ext in (".vst3", ".component", ".vst", ".dll", ".so"):
        if base.lower().endswith(ext):
            return base[: -len(ext)]
    return base


def find_fx_sibling_plugin_path(plugin_path: str, candidates: list[str] | None = None) -> str | None:
    """Return the FX variant path for an instrument sibling, if installed."""
    abs_path = os.path.abspath((plugin_path or "").strip())
    if not abs_path:
        return None

    label = format_plugin_name(abs_path)
    if label_suggests_fx(label):
        return abs_path

    if candidates is None:
        plugin_paths, _ = plugin_settings_tuple()
        candidates = [
            path
            for path in list_installed_plugins(plugin_paths)
            if os.path.exists(path)
        ]

    source_key = plugin_pair_key(label)
    source_stem = _plugin_file_stem(abs_path)
    stem_key = plugin_pair_key(source_stem)

    for candidate in candidates:
        candidate_abs = os.path.abspath(candidate)
        if candidate_abs == abs_path:
            continue
        candidate_label = format_plugin_name(candidate_abs)
        if not label_suggests_fx(candidate_label):
            continue
        if plugin_pair_key(candidate_label) == source_key:
            return candidate_abs
        if plugin_pair_key(_plugin_file_stem(candidate_abs)) == stem_key:
            return candidate_abs

    for suffix in (" FX", "FX", "-FX", "_FX"):
        target_stem = f"{source_stem}{suffix}"
        for candidate in candidates:
            if _plugin_file_stem(candidate).lower() == target_stem.lower():
                return os.path.abspath(candidate)
    return None


def resolve_fx_plugin_path(plugin_path: str) -> str:
    """Prefer an explicit FX sibling when an instrument plug-in was chosen for an FX slot."""
    from dronmakr.audio.faust_fx_library import is_faust_fx_path
    from dronmakr.audio.faust_library import is_faust_instrument_path

    raw = (plugin_path or "").strip()
    if is_faust_fx_path(raw):
        return raw
    if is_faust_instrument_path(raw):
        raise ValueError(
            f"“{raw}” is a built-in Faust instrument, not an FX. "
            "Use a faustfx: effect from the FX library instead."
        )

    abs_path = os.path.abspath(raw)
    if not abs_path:
        return abs_path

    label = format_plugin_name(abs_path)
    if label_suggests_fx(label):
        return abs_path

    sibling = find_fx_sibling_plugin_path(abs_path)
    if sibling:
        return sibling

    base = plugin_base_name(label)
    try:
        from dronmakr.audio.audio_host import create_engine, load_plugin, plugin_is_instrument

        engine = create_engine()
        processor = load_plugin(engine, abs_path)
        if plugin_is_instrument(processor):
            hint = f" Use the FX variant (for example “{base} FX”)." if base else ""
            raise ValueError(f"“{label}” is an instrument plug-in, not an FX.{hint}")
    except ValueError:
        raise
    except Exception:
        pass

    return abs_path


def format_plugin_name(plugin_path: str) -> str:
    from dronmakr.audio.faust_library import (
        faust_id_from_path,
        is_faust_instrument_path,
        list_faust_instruments,
    )

    if is_faust_instrument_path(plugin_path):
        faust_id = faust_id_from_path(plugin_path)
        match = next((entry for entry in list_faust_instruments() if entry["id"] == faust_id), None)
        return match["label"] if match else faust_id
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


def delete_preset_by_name(name: str) -> bool:
    preset_name = (name or "").strip()
    if not preset_name:
        return False
    for preset in load_presets_json():
        if preset.get("name") == preset_name:
            return delete_preset_by_id(preset.get("id") or "")
    return False
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
                "No instruments — add one in Generate Samples or edit config/presets.json in your dronmakr files folder."
            )
        )
        return

    if len(effects_index) < 1:
        print(
            with_prompt(
                "No saved effects — edit config/presets.json in your dronmakr files folder."
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


def slot_allowed_as_chain_effect(plugin) -> bool:
    """FX chain accepts plug-ins that report as effects."""
    return plugin_is_effect(plugin)


# Backward-compatible aliases during migration
load_pedalboard_plugin = load_plugin
reload_pedalboard_plugin_preserving_state = reload_plugin_preserving_state
macos_pedalboard_rejects_vst2_path = macos_legacy_vst2_path
