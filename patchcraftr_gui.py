"""Desktop Patchcraftr — Tk GUI for authoring drone presets (Pedalboard)."""

from __future__ import annotations

import os
import queue
import sys
import threading

try:
    import tkinter as tk
except ModuleNotFoundError as exc:
    _missing = getattr(exc, "name", "") or ""
    if _missing in ("_tkinter", "tkinter"):
        _maj, _min = sys.version_info.major, sys.version_info.minor
        print(
            "patchcraftr requires Tkinter (_tkinter).\n\n"
            "If you are running the packaged desktop app, rebuild with the current "
            "desktop.spec (it bundles Tcl/Tk). If this still fails, file an issue with your OS and build target.\n\n"
            "If you are running from source on macOS with Homebrew Python, install bindings for your version, e.g.:\n"
            f"  brew install python-tk@{_maj}.{_min}\n\n"
            "Or use a python.org installer (includes Tk). Homebrew’s python@X formula does not ship Tk by default.\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    raise
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Any

from pedalboard import Pedalboard
from patchcraftr_live_monitor import (
    DEFAULT_FX_PREVIEW_SOURCE,
    DEFAULT_MIDI_PREVIEW_STYLE,
    FX_PREVIEW_SOURCES,
    MIDI_PREVIEW_STYLES,
    PatchcraftrLiveMonitor,
)
from preset_authoring import (
    MAX_CHAIN_SLOTS,
    PluginVariantRequired,
    PresetAuthoringConfigError,
    apply_vstpreset_bytes_to_plugin,
    assert_plugin_paths_configured,
    build_plugin_label_map,
    effect_chain_tuple_to_json_effects,
    ensure_authoring_dirs,
    format_plugin_name,
    load_pedalboard_plugin,
    load_presets_json,
    name_exists,
    plugin_settings_tuple,
    reload_pedalboard_plugin_preserving_state,
    serialize_plugin_preset_bytes,
    slot_allowed_as_chain_effect,
    upsert_preset_entry,
    write_plugin_state_to_vstpreset,
)
from settings import ensure_settings
from utils import PRESETS_DIR, generate_id


# Mirrors templates/_app_css_root.html (keep in sync with web UI CSS variables).
WEB_UI_THEME: dict[str, str] = {
    "secondary": "#000000",
    "primary": "#ff9505",
    "theme_a": "#353531",
    "theme_b": "#016fb9",
    "theme_c": "#ec4e20",
    "theme_d": "#FFFFFF",
    "theme_e": "#2e8b57",
}


@dataclass
class FxSlotState:
    plugin_path: str
    plugin_name: str
    plugin: Any
    effect_display_name: str
    effect_uid: str | None = None
    preset_path: str | None = None


class PatchcraftrApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("dronmakr · patchcraftr")
        self.geometry("760x780")
        self.minsize(680, 700)

        self._configure_patchcraftr_theme()

        _, self._assert_instrument, _, _ = plugin_settings_tuple()

        self._plugin_labels_sorted: list[str] = []
        self._plugin_map: dict[str, str] = {}

        self._inst_plugin_path = ""
        self._inst_plugin_name = ""
        self._inst_plugin: Any = None

        self._inst_editor_was_shown_before = False
        self._editor_midi_style_ref = [DEFAULT_MIDI_PREVIEW_STYLE]
        self._midi_preview_style = tk.StringVar(value=DEFAULT_MIDI_PREVIEW_STYLE)
        self._editor_fx_source_ref = [DEFAULT_FX_PREVIEW_SOURCE]
        self._fx_preview_source = tk.StringVar(value=DEFAULT_FX_PREVIEW_SOURCE)

        self._fx_slots: list[FxSlotState | None] = [None] * MAX_CHAIN_SLOTS

        self._slot_labels: list[ttk.Label] = []

        self._inst_name = tk.StringVar()
        self._chain_name = tk.StringVar()

        self._editor_active = False

        self._msg_q: queue.Queue[tuple[str, str]] = queue.Queue()

        self._build_ui()
        self._refresh_plugin_map()
        self.after(150, self._pump_messages)

        try:
            assert_plugin_paths_configured()
        except PresetAuthoringConfigError as e:
            messagebox.showerror("patchcraftr", str(e), parent=self)

        ensure_authoring_dirs()

    def _configure_patchcraftr_theme(self) -> None:
        """Approximate templates/_app_css_root.html using ttk (clam/alt) + themed tk dialogs."""
        C = WEB_UI_THEME
        self._theme_colors = C
        secondary = C["secondary"]
        primary = C["primary"]
        theme_a = C["theme_a"]
        theme_b = C["theme_b"]
        fg = C["theme_d"]
        accent_green = C["theme_e"]
        inner_entry = "#2f2f2b"

        style = ttk.Style(self)
        for name in ("clam", "alt", "default"):
            try:
                style.theme_use(name)
                break
            except tk.TclError:
                continue

        self.configure(background=secondary)

        style.configure(".", background=secondary, foreground=fg)
        style.configure("Chrome.TFrame", background=secondary)
        style.configure(
            "AccentTitle.TLabel",
            background=secondary,
            foreground=primary,
            font=("Helvetica", 16, "bold"),
        )
        style.configure(
            "Hint.TLabel",
            background=secondary,
            foreground="#a8a8a8",
            font=("Helvetica", 10),
        )

        style.configure("TFrame", background=theme_a)
        style.configure(
            "TLabelframe",
            background=theme_a,
            foreground=primary,
            labelforeground=primary,
            borderwidth=2,
            relief="solid",
        )
        style.configure(
            "TLabelframe.Label",
            background=theme_a,
            foreground=primary,
            font=("Helvetica", 10, "bold"),
        )
        style.configure("TLabel", background=theme_a, foreground=fg)

        style.configure(
            "TEntry",
            fieldbackground=inner_entry,
            foreground=fg,
            bordercolor=theme_b,
            insertcolor=fg,
            lightcolor=theme_b,
            darkcolor=theme_a,
        )
        style.map(
            "TEntry",
            fieldbackground=[("readonly", inner_entry)],
            foreground=[("disabled", "#707070")],
        )

        style.configure(
            "TButton",
            background=theme_a,
            foreground=fg,
            bordercolor=theme_b,
            focusthickness=2,
            focuscolor=accent_green,
            lightcolor=theme_a,
            darkcolor=theme_a,
            padding=(10, 4),
            font=("Helvetica", 10),
        )
        style.map(
            "TButton",
            background=[("active", theme_b), ("pressed", theme_a), ("disabled", "#2a2a2a")],
            foreground=[("disabled", "#666666"), ("pressed", fg)],
            bordercolor=[("focus", accent_green), ("!focus", theme_b)],
        )

        style.configure(
            "Accent.TButton",
            background=primary,
            foreground=secondary,
            bordercolor=primary,
            focuscolor=accent_green,
            lightcolor=primary,
            darkcolor=primary,
            padding=(14, 6),
            font=("Helvetica", 11, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#ffa533"), ("pressed", primary), ("disabled", "#404040")],
            foreground=[("disabled", "#888888"), ("pressed", secondary)],
            bordercolor=[("focus", accent_green)],
        )

        style.configure(
            "TRadiobutton",
            background=theme_a,
            foreground=fg,
            focuscolor=theme_a,
            font=("Helvetica", 10),
        )
        style.map(
            "TRadiobutton",
            background=[
                ("active", theme_a),
                ("disabled", theme_a),
                ("selected", theme_a),
            ],
            foreground=[("disabled", "#707070"), ("focus", fg)],
            indicatorcolor=[("selected", primary), ("alternate", fg), ("!selected", inner_entry)],
            indicatorforeground=[("selected", secondary)],
        )

    def _theme_dialog_shell(self, w: tk.Widget, *, outer: bool = False) -> None:
        bg = WEB_UI_THEME["secondary"] if outer else WEB_UI_THEME["theme_a"]
        try:
            w.configure(background=bg)
        except tk.TclError:
            pass

    def _theme_listbox(self, lb: tk.Listbox) -> None:
        C = WEB_UI_THEME
        lb.configure(
            background="#2f2f2b",
            foreground=C["theme_d"],
            selectbackground=C["primary"],
            selectforeground=C["secondary"],
            activestyle="none",
            highlightthickness=1,
            highlightbackground=C["theme_b"],
            highlightcolor=C["primary"],
            borderwidth=0,
            relief="flat",
            font=("Helvetica", 11),
        )

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=10, style="Chrome.TFrame")
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="patchcraftr", style="AccentTitle.TLabel").pack(
            anchor="w", pady=(0, 8)
        )
        hint = (
            "Build an instrument or FX chain to use in dronmakr UI or `generate-drone` CLI"
        )
        ttk.Label(
            main, text=hint, style="Hint.TLabel", wraplength=720, justify="left"
        ).pack(
            anchor="w", pady=(0, 12)
        )

        inst_fr = ttk.LabelFrame(main, text="Instrument", padding=10)
        inst_fr.pack(fill=tk.X, pady=(0, 8))

        r1 = ttk.Frame(inst_fr)
        r1.pack(fill=tk.X)
        ttk.Label(r1, text="Plug-in", width=12).pack(side=tk.LEFT, anchor="nw")
        col = ttk.Frame(r1)
        col.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._inst_plugin_lbl = ttk.Label(col, text="(none)")
        self._inst_plugin_lbl.pack(anchor="w")
        bf_inst = ttk.Frame(col)
        bf_inst.pack(anchor="w", pady=(6, 0))
        ttk.Button(bf_inst, text="Choose instrument", command=self._inst_choose_plugin).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(bf_inst, text="Edit instrument", command=self._inst_edit_plugin).pack(
            side=tk.LEFT
        )

        prev_lf = ttk.LabelFrame(inst_fr, text="Preview settings", padding=(6, 4))
        prev_lf.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(prev_lf, text="Instrument MIDI preview", font=("Helvetica", 9)).pack(
            anchor="w"
        )
        pc = ttk.Frame(prev_lf)
        pc.pack(fill=tk.X)
        for ix, (sid, label) in enumerate(MIDI_PREVIEW_STYLES):
            ttk.Radiobutton(
                pc,
                text=label,
                value=sid,
                variable=self._midi_preview_style,
                command=self._sync_editor_midi_style_ref,
            ).grid(row=ix // 2, column=ix % 2, sticky="w", padx=(0, 12), pady=2)

        ttk.Label(
            prev_lf,
            text="FX-only dry signal (used when no instrument is loaded)",
            font=("Helvetica", 9),
        ).pack(anchor="w", pady=(10, 0))
        fx_pc = ttk.Frame(prev_lf)
        fx_pc.pack(fill=tk.X)
        for jx, (sid, label) in enumerate(FX_PREVIEW_SOURCES):
            ttk.Radiobutton(
                fx_pc,
                text=label,
                value=sid,
                variable=self._fx_preview_source,
                command=self._sync_editor_fx_dry_source_ref,
            ).grid(row=jx // 2, column=jx % 2, sticky="w", padx=(0, 12), pady=2)

        fx_fr = ttk.LabelFrame(
            main, text=f"FX chain ({MAX_CHAIN_SLOTS})", padding=10
        )
        fx_fr.pack(fill=tk.X, pady=(0, 8))

        for i in range(MAX_CHAIN_SLOTS):
            row = ttk.Frame(fx_fr)
            row.pack(fill=tk.X, pady=4)
            ttk.Label(row, text=f"{i + 1}.", width=3).pack(side=tk.LEFT, anchor="n", pady=2)
            lb = ttk.Label(row, text="(empty)", width=50)
            lb.pack(side=tk.LEFT, padx=(4, 8), anchor="w")
            self._slot_labels.append(lb)
            ttk.Button(row, text="Select plugin", width=10, command=lambda ix=i: self._fx_set_slot(ix)).pack(
                side=tk.LEFT, padx=2
            )
            ttk.Button(row, text="Open plugin", width=10, command=lambda ix=i: self._fx_edit_slot(ix)).pack(
                side=tk.LEFT, padx=2
            )
            ttk.Button(row, text="Clear", width=8, command=lambda ix=i: self._fx_clear_slot(ix)).pack(
                side=tk.LEFT, padx=2
            )

        nm_fr = ttk.LabelFrame(main, text="Patch Name", padding=10)
        nm_fr.pack(fill=tk.X, pady=(0, 8))

        nrow = ttk.Frame(nm_fr)
        nrow.pack(fill=tk.X)
        ttk.Label(
            nrow,
            text="Instrument name",
            width=28,
            justify="left",
        ).grid(row=0, column=0, sticky="nw")
        ttk.Entry(nrow, textvariable=self._inst_name, width=48).grid(
            row=0, column=1, sticky="ew", padx=(6, 0)
        )
        nrow.columnconfigure(1, weight=1)

        nrow2 = ttk.Frame(nm_fr)
        nrow2.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(
            nrow2,
            text="Effect name",
            width=28,
            justify="left",
        ).grid(row=0, column=0, sticky="nw")
        ttk.Entry(nrow2, textvariable=self._chain_name, width=48).grid(
            row=0, column=1, sticky="ew", padx=(6, 0)
        )
        nrow2.columnconfigure(1, weight=1)

        ttk.Button(
            main,
            text="Save patch",
            style="Accent.TButton",
            command=self._save_patch,
        ).pack(anchor="e", pady=(12, 0))

    def _slot_label_widget(self, idx: int) -> ttk.Label:
        return self._slot_labels[idx]

    def _pump_messages(self) -> None:
        try:
            while True:
                kind, text = self._msg_q.get_nowait()
                if kind == "error":
                    messagebox.showerror("patchcraftr", text, parent=self)
                elif kind == "info":
                    messagebox.showinfo("patchcraftr", text, parent=self)
        except queue.Empty:
            pass
        self.after(150, self._pump_messages)

    def _sync_editor_midi_style_ref(self) -> None:
        self._editor_midi_style_ref[0] = self._midi_preview_style.get()

    def _sync_editor_fx_dry_source_ref(self) -> None:
        self._editor_fx_source_ref[0] = self._fx_preview_source.get()

    def _enqueue_msg(self, kind: str, text: str) -> None:
        self._msg_q.put((kind, text))

    def _refresh_plugin_map(self) -> None:
        self._plugin_map = build_plugin_label_map()
        self._plugin_labels_sorted = sorted(self._plugin_map.keys(), key=str.lower)

    def _refresh_slot_labels(self) -> None:
        for i in range(MAX_CHAIN_SLOTS):
            st = self._fx_slots[i]
            lb = self._slot_label_widget(i)
            if st is None:
                lb.configure(text="(empty)")
            else:
                lb.configure(text=st.effect_display_name or format_plugin_name(st.plugin_path))

    def _reset_patch_form(self) -> None:
        self._inst_plugin_path = ""
        self._inst_plugin_name = ""
        self._inst_plugin = None
        self._inst_editor_was_shown_before = False
        self._inst_plugin_lbl.configure(text="(none)")
        self._inst_name.set("")
        self._chain_name.set("")
        self._fx_slots = [None] * MAX_CHAIN_SLOTS
        self._midi_preview_style.set(DEFAULT_MIDI_PREVIEW_STYLE)
        self._sync_editor_midi_style_ref()
        self._fx_preview_source.set(DEFAULT_FX_PREVIEW_SOURCE)
        self._sync_editor_fx_dry_source_ref()
        self._refresh_slot_labels()

    def _resolve_variant(self, path: str, variants: list[str]) -> str | None:
        dlg = tk.Toplevel(self)
        self._theme_dialog_shell(dlg)
        dlg.title("Choose plug-in variant")
        dlg.transient(self)
        dlg.grab_set()
        chosen: list[str | None] = [None]

        lb = tk.Listbox(dlg, height=min(12, len(variants)), width=48)
        self._theme_listbox(lb)
        for v in variants:
            lb.insert(tk.END, v)
        lb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        def ok() -> None:
            sel = lb.curselection()
            if sel:
                chosen[0] = variants[sel[0]]
            dlg.destroy()

        bf = ttk.Frame(dlg)
        bf.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(bf, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT, padx=8)
        ttk.Button(bf, text="OK", command=ok).pack(side=tk.RIGHT)

        dlg.wait_window()
        return chosen[0]

    def _load_plugin_with_variants(
        self, path: str, hint_name: str | None = None
    ) -> tuple[Any, str] | None:
        cur_hint = hint_name
        while True:
            try:
                return load_pedalboard_plugin(path, cur_hint or None)
            except PluginVariantRequired as e:
                pick = self._resolve_variant(e.plugin_path, e.variants)
                if pick is None:
                    return None
                cur_hint = pick

    def _pick_plugin_path(self, title: str) -> str | None:
        if not self._plugin_labels_sorted:
            messagebox.showwarning(
                "patchcraftr",
                "No plug-ins found. Set PLUGIN_PATHS in Settings.",
                parent=self,
            )
            return None

        dlg = tk.Toplevel(self)
        self._theme_dialog_shell(dlg)
        dlg.title(title)
        dlg.transient(self)
        dlg.geometry("420x420")
        dlg.grab_set()

        filt = tk.StringVar()
        ttk.Entry(dlg, textvariable=filt).pack(fill=tk.X, padx=8, pady=(8, 4))

        lb = tk.Listbox(dlg, width=52, height=18)
        self._theme_listbox(lb)
        lb.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        labels = list(self._plugin_labels_sorted)

        def refill(filter_txt: str = "") -> None:
            lb.delete(0, tk.END)
            ft = filter_txt.strip().lower()
            for lab in labels:
                if ft in lab.lower():
                    lb.insert(tk.END, lab)

        refill()

        def on_filter(*_a):
            refill(filt.get())

        filt.trace_add("write", on_filter)

        out: list[str | None] = [None]

        def ok() -> None:
            sel = lb.curselection()
            if sel:
                lab = lb.get(sel[0])
                out[0] = self._plugin_map.get(lab)
            dlg.destroy()

        bf = ttk.Frame(dlg)
        bf.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(bf, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT, padx=8)
        ttk.Button(bf, text="OK", command=ok).pack(side=tk.RIGHT)

        dlg.wait_window()
        return out[0]

    def _saved_effect_presets_rows(self) -> list[dict]:
        return [p for p in load_presets_json() if p.get("type") == "effect"]

    def _pick_saved_effect_preset(self, slot_idx: int) -> dict | None:
        rows = self._saved_effect_presets_rows()
        if not rows:
            messagebox.showinfo(
                "patchcraftr",
                "No saved single-FX presets yet. Save one with one FX slot, or use Browse plug-ins.",
                parent=self,
            )
            return None

        dlg = tk.Toplevel(self)
        self._theme_dialog_shell(dlg)
        dlg.title(f"Saved effect for slot {slot_idx + 1}")
        dlg.transient(self)
        dlg.grab_set()

        lb = tk.Listbox(dlg, width=54, height=14)
        self._theme_listbox(lb)
        for r in sorted(rows, key=lambda x: str(x.get("name", "")).lower()):
            lb.insert(tk.END, r["name"])
        lb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        picked: list[dict | None] = [None]

        def ok() -> None:
            sel = lb.curselection()
            if sel:
                name = lb.get(sel[0])
                for r in rows:
                    if r.get("name") == name:
                        picked[0] = r
                        break
            dlg.destroy()

        bf = ttk.Frame(dlg)
        bf.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(bf, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT, padx=8)
        ttk.Button(bf, text="OK", command=ok).pack(side=tk.RIGHT)
        dlg.wait_window()
        return picked[0]

    def _fx_set_slot(self, idx: int) -> None:
        dlg = tk.Toplevel(self)
        self._theme_dialog_shell(dlg)
        dlg.title(f"FX slot {idx + 1}")
        dlg.transient(self)
        dlg.grab_set()
        chosen: list[str | None] = [None]

        def pick_saved() -> None:
            chosen[0] = "saved"
            dlg.destroy()

        def pick_browse() -> None:
            chosen[0] = "browse"
            dlg.destroy()

        ttk.Label(dlg, text="Load a saved single effect, or browse the system for a plug-in.", wraplength=420).pack(
            padx=12, pady=12
        )
        bf = ttk.Frame(dlg)
        bf.pack(pady=(0, 16))
        ttk.Button(bf, text="Saved effect preset…", command=pick_saved).pack(side=tk.LEFT, padx=8)
        ttk.Button(bf, text="Browse plug-ins…", command=pick_browse).pack(side=tk.LEFT, padx=8)
        ttk.Button(bf, text="Cancel", command=dlg.destroy).pack(side=tk.LEFT, padx=8)

        dlg.wait_window()
        mode = chosen[0]
        if mode == "saved":
            row = self._pick_saved_effect_preset(idx)
            if row:
                self._fx_apply_saved_effect_row(idx, row)
        elif mode == "browse":
            self._fx_apply_browse_path(idx)

    def _fx_apply_saved_effect_row(self, idx: int, row: dict) -> None:
        try:
            loaded = self._load_plugin_with_variants(
                row["plugin_path"], row.get("plugin_name") or None
            )
            if loaded is None:
                return
            plug, pname = loaded
            if not slot_allowed_as_chain_effect(plug, pname, self._assert_instrument):
                messagebox.showwarning(
                    "patchcraftr",
                    "That plug-in cannot be used as an FX chain step with current settings.",
                    parent=self,
                )
                return
            with open(row["preset_path"], "rb") as f:
                apply_vstpreset_bytes_to_plugin(plug, f.read())
            self._fx_slots[idx] = FxSlotState(
                plugin_path=row["plugin_path"],
                plugin_name=(row.get("plugin_name") or "") or (pname or ""),
                plugin=plug,
                effect_display_name=row.get("name", format_plugin_name(row["plugin_path"])),
                effect_uid=row.get("id"),
                preset_path=row.get("preset_path"),
            )
            self._refresh_slot_labels()
        except Exception as e:
            self._enqueue_msg("error", str(e))

    def _fx_apply_browse_path(self, idx: int) -> None:
        path = self._pick_plugin_path(f"FX slot {idx + 1}")
        if not path:
            return
        try:
            loaded = self._load_plugin_with_variants(path)
            if loaded is None:
                return
            plug, pname = loaded
            if not slot_allowed_as_chain_effect(plug, pname, self._assert_instrument):
                messagebox.showwarning(
                    "patchcraftr",
                    "That plug-in is not usable as an FX step with ASSERT_INSTRUMENT settings.",
                    parent=self,
                )
                return
            disp = format_plugin_name(path)
            if pname:
                disp = f"{disp} · {pname}"
            self._fx_slots[idx] = FxSlotState(
                plugin_path=path,
                plugin_name=pname or "",
                plugin=plug,
                effect_display_name=disp[:96],
                effect_uid=None,
                preset_path=None,
            )
            self._refresh_slot_labels()
        except Exception as e:
            self._enqueue_msg("error", str(e))

    def _inst_choose_plugin(self) -> None:
        path = self._pick_plugin_path("Instrument plug-in")
        if not path:
            return
        try:
            loaded = self._load_plugin_with_variants(path)
            if loaded is None:
                return
            plug, pname = loaded
            if not plug.is_instrument:
                messagebox.showwarning(
                    "patchcraftr",
                    "That plug-in is not an instrument. Use FX slots for effects.",
                    parent=self,
                )
                return
            self._inst_plugin_path = path
            self._inst_plugin_name = pname or ""
            self._inst_plugin = plug
            self._inst_editor_was_shown_before = False
            self._inst_plugin_lbl.configure(
                text=f"{format_plugin_name(path)} · {self._inst_plugin_name or 'default'}"
            )
        except Exception as e:
            self._enqueue_msg("error", str(e))

    def _run_plugin_editor(
        self,
        editor_plugin: Any,
        *,
        instrument_editor: bool = False,
        audio_engine: Any | None = None,
    ) -> None:
        if self._editor_active:
            messagebox.showwarning(
                "patchcraftr",
                "A plug-in editor is already open.",
                parent=self,
            )
            return

        monitor_target = editor_plugin if audio_engine is None else audio_engine
        upstream_inst = None
        if audio_engine is not None and self._inst_plugin is not None:
            upstream_inst = self._inst_plugin

        close_evt = threading.Event()
        self._editor_active = True
        self._sync_editor_midi_style_ref()
        self._sync_editor_fx_dry_source_ref()
        monitor = PatchcraftrLiveMonitor(
            monitor_target,
            midi_style_ref=self._editor_midi_style_ref,
            upstream_instrument=upstream_inst,
            fx_preview_source_ref=self._editor_fx_source_ref,
        )
        monitor.start()
        try:
            editor_plugin.show_editor(close_evt)
        except BaseException as e:
            self._enqueue_msg("error", f"Editor error:\n{e}")
        finally:
            monitor.stop()
            self._editor_active = False
            if instrument_editor:
                self._inst_editor_was_shown_before = True

    def _inst_edit_plugin(self) -> None:
        if self._inst_plugin is None:
            messagebox.showwarning("patchcraftr", "Choose an instrument plug-in first.", parent=self)
            return
        if not self._inst_plugin_path:
            messagebox.showwarning(
                "patchcraftr", "Instrument plug-in path is missing; pick again.", parent=self
            )
            return
        if self._inst_editor_was_shown_before:
            old = self._inst_plugin
            blob = serialize_plugin_preset_bytes(old)
            try:
                wp, pname = reload_pedalboard_plugin_preserving_state(
                    self._inst_plugin_path,
                    self._inst_plugin_name or None,
                    old,
                )
            except PluginVariantRequired as e:
                pick = self._resolve_variant(e.plugin_path, e.variants)
                if pick is None:
                    return
                wp, pname = load_pedalboard_plugin(self._inst_plugin_path, pick)
                apply_vstpreset_bytes_to_plugin(wp, blob)
            except Exception as e:
                self._enqueue_msg(
                    "error",
                    "Could not refresh instrument before reopening its editor:\n"
                    f"{e}",
                )
                return
            if not wp.is_instrument:
                messagebox.showwarning(
                    "patchcraftr",
                    "That plug-in is not an instrument.",
                    parent=self,
                )
                return
            self._inst_plugin = wp
            self._inst_plugin_name = pname or ""
            self._inst_plugin_lbl.configure(
                text=f"{format_plugin_name(self._inst_plugin_path)} · {self._inst_plugin_name or 'default'}"
            )

        self._run_plugin_editor(self._inst_plugin, instrument_editor=True)

    def _ordered_fx_chain_plugins(self) -> list[Any]:
        return [s.plugin for s in self._fx_slots if s is not None]

    def _fx_edit_slot(self, idx: int) -> None:
        st = self._fx_slots[idx]
        if st is None:
            messagebox.showwarning("patchcraftr", "Slot is empty.", parent=self)
            return
        chain = self._ordered_fx_chain_plugins()
        audio_engine = Pedalboard(chain)

        self._run_plugin_editor(st.plugin, instrument_editor=False, audio_engine=audio_engine)

    def _fx_clear_slot(self, idx: int) -> None:
        self._fx_slots[idx] = None
        self._refresh_slot_labels()

    def _save_patch(self) -> None:
        has_inst = self._inst_plugin is not None and bool(self._inst_plugin_path)
        tuples: list[tuple[str, str, Any, tuple[str, str, str, str]]] = []
        for st in self._fx_slots:
            if st is None:
                continue
            uid_e = st.effect_uid or generate_id()
            safe_nm = st.effect_display_name.replace("/", "-")[:64]
            preset_path_slot = st.preset_path or os.path.join(
                PRESETS_DIR, f"{safe_nm}_{uid_e}.vstpreset"
            )
            write_plugin_state_to_vstpreset(preset_path_slot, st.plugin)
            meta = (st.effect_display_name, uid_e, preset_path_slot, "")
            tuples.append((st.plugin_path, st.plugin_name, st.plugin, meta))

        has_fx = len(tuples) > 0
        nm_inst = self._inst_name.get().strip()
        nm_fx = self._chain_name.get().strip()

        if not has_inst and not has_fx:
            messagebox.showwarning(
                "patchcraftr",
                "Choose an instrument plug-in and/or fill at least one FX slot.",
                parent=self,
            )
            return
        if has_inst and not nm_inst:
            messagebox.showwarning(
                "patchcraftr", "Instrument name is required when an instrument is selected.", parent=self
            )
            return
        if has_fx and not nm_fx:
            messagebox.showwarning(
                "patchcraftr",
                "Effect / FX chain name is required when any FX slot is used.",
                parent=self,
            )
            return

        if has_inst:
            if name_exists(nm_inst, types=("instrument",)):
                messagebox.showwarning(
                    "patchcraftr", f"An instrument preset named '{nm_inst}' already exists.", parent=self
                )
                return

        if has_fx:
            if name_exists(nm_fx, types=("effect", "effect_chain")):
                messagebox.showwarning(
                    "patchcraftr",
                    f"A saved effect or chain named '{nm_fx}' already exists.",
                    parent=self,
                )
                return

        try:
            if has_inst:
                uid_inst = generate_id()
                preset_path_inst = os.path.join(PRESETS_DIR, f"{nm_inst}_{uid_inst}.vstpreset")
                write_plugin_state_to_vstpreset(preset_path_inst, self._inst_plugin)
                upsert_preset_entry(
                    {
                        "id": uid_inst,
                        "name": nm_inst,
                        "plugin_path": self._inst_plugin_path,
                        "plugin_name": self._inst_plugin_name,
                        "preset_path": preset_path_inst,
                        "type": "instrument",
                    }
                )

            if has_fx:
                uid_fx = generate_id()
                if len(tuples) == 1:
                    path_s, pname_s, _plug_s, meta = tuples[0]
                    preset_path_eff = meta[2]
                    upsert_preset_entry(
                        {
                            "id": uid_fx,
                            "name": nm_fx,
                            "type": "effect",
                            "plugin_path": path_s,
                            "plugin_name": pname_s,
                            "preset_path": preset_path_eff,
                        }
                    )
                else:
                    upsert_preset_entry(
                        {
                            "id": uid_fx,
                            "name": nm_fx,
                            "type": "effect_chain",
                            "effects": effect_chain_tuple_to_json_effects(tuples),
                        }
                    )

        except Exception as e:
            self._enqueue_msg("error", str(e))
            return

        parts = []
        if has_inst:
            parts.append(f"instrument “{nm_inst}”")
        if has_fx:
            parts.append(f"effect “{nm_fx}”" + (" [chain]" if len(tuples) > 1 else " [single FX]"))

        self._enqueue_msg("info", "Saved " + " and ".join(parts) + ".")
        self._reset_patch_form()


def main() -> None:
    ensure_settings()
    ensure_authoring_dirs()
    app = PatchcraftrApp()
    app.mainloop()


if __name__ == "__main__":
    main()
