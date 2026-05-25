"""Desktop Patchcraftr — Tk GUI for authoring drone presets (Pedalboard)."""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager

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
from collections import defaultdict
from dataclasses import dataclass
import tkinter.font as tkfont
from tkinter import messagebox, ttk
from typing import Any, Callable

from pedalboard import Pedalboard
from patchcraftr_live_monitor import (
    DEFAULT_FX_PREVIEW_SOURCE,
    DEFAULT_MIDI_PREVIEW_STYLE,
    FX_PREVIEW_SOURCES,
    MIDI_PREVIEW_STYLES,
    PatchcraftrLiveMonitor,
    SAMPLE_RATE,
    render_preview_clip,
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
    list_installed_plugins,
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
from settings import ensure_settings, load_settings, save_settings
from server_error_logging import ensure_server_error_file_logging
from utils import PRESETS_DIR, generate_id


_LOG = logging.getLogger("dronmakr.patchcraftr")

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
    # After closing show_editor once, some VST3/AUs stop processing preview audio until reloaded.
    editor_opened_before: bool = False


class PatchcraftrApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("dronmakr · patchcraftr")
        self.geometry("1120x900")
        self.minsize(960, 820)

        self._init_mono_font_family()
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
        self._main_thread_jobs: queue.Queue[Callable[[], None]] = queue.Queue()

        try:
            _LOG.info("Patchcraftr UI constructing (Tk main thread)")
        except Exception:
            pass

        self._build_ui()
        self._refresh_plugin_map_apply({})
        self.after(160, self._initial_plugin_catalog_scan)
        self.after(150, self._pump_messages)

        ensure_authoring_dirs()

    def _schedule_ui(self, fn: Callable[[], None]) -> None:
        self._main_thread_jobs.put(fn)

    def _refresh_plugin_map_apply(self, m: dict[str, str]) -> None:
        self._plugin_map = m
        self._plugin_labels_sorted = sorted(m.keys(), key=str.lower)

    def _initial_plugin_catalog_scan(self) -> None:
        """Non-blocking SETTINGS + PLUGIN_PATHS validation and first catalog load."""

        def work() -> None:
            try:
                assert_plugin_paths_configured()
            except PresetAuthoringConfigError as e:
                self._schedule_ui(
                    lambda err=str(e): messagebox.showerror("patchcraftr", err, parent=self)
                )
                return
            t0 = time.time()
            try:
                m = build_plugin_label_map()
            except Exception:
                _LOG.exception("Plug-in catalog scan failed (startup)")
                self._schedule_ui(
                    lambda: self._enqueue_msg(
                        "error",
                        "Plug-in catalog scan failed. See errors.log for details.",
                    )
                )
                return
            dt = time.time() - t0
            self._schedule_ui(lambda mm=m, secs=dt: self._finish_plugin_catalog_scan(mm, secs))

        threading.Thread(target=work, daemon=True).start()

    def _finish_plugin_catalog_scan(self, m: dict[str, str], seconds: float) -> None:
        self._refresh_plugin_map_apply(m)
        _LOG.info(
            "Plug-in picker catalog updated: %s items (scan %.2fs)",
            len(self._plugin_labels_sorted),
            seconds,
        )

    def _refresh_plugin_map(self) -> None:
        """Rebuild plug-in labels on the Tk thread (brief disk scan — avoid during drag)."""
        self._refresh_plugin_map_apply(build_plugin_label_map())

    def _enqueue_msg(self, kind: str, text: str) -> None:
        if kind == "error":
            _LOG.error("Patchcraftr message [%s]: %s", kind, text)
        else:
            _LOG.info("Patchcraftr message [%s]: %s", kind, text)
        self._msg_q.put((kind, text))

    @contextmanager
    def _modal_busy(self, message: str):
        """
        Show a lightweight modal splash so long-running Pedalboard work does not look like dead UI.

        On some hosts ``load_plugin`` can take many seconds while the Tk main thread still must run this work;
        grab + drawing here makes that state obvious and avoids contradictory user input.
        """
        splash = tk.Toplevel(self)
        self._theme_dialog_shell(splash)
        splash.title("patchcraftr")
        splash.transient(self)
        splash.resizable(False, False)
        ttk.Label(splash, text=message, wraplength=460, justify="left").pack(padx=28, pady=28)
        splash.update_idletasks()
        w, h = splash.winfo_reqwidth(), splash.winfo_reqheight()
        self.update_idletasks()
        px = self.winfo_rootx() + max(40, (self.winfo_width() - w) // 2)
        py = self.winfo_rooty() + max(40, (self.winfo_height() - h) // 2)
        splash.geometry(f"+{px}+{py}")
        try:
            splash.grab_set()
        except tk.TclError:
            pass
        splash.update()
        try:
            yield splash
        finally:
            try:
                splash.grab_release()
            except tk.TclError:
                pass
            try:
                splash.destroy()
            except tk.TclError:
                pass

    def _init_mono_font_family(self) -> None:
        try:
            available = {str(f).lower() for f in tkfont.families()}
        except tk.TclError:
            available = set()
        for cand in ("Menlo", "Consolas", "SF Mono", "Monaco", "Courier New", "DejaVu Sans Mono"):
            if cand.lower() in available:
                self._mono_family: str = cand
                return
        self._mono_family = "Courier New"

    def _mono(self, pts: int, *, bold: bool = False):
        """UI monospace font tuple for ttk / tk widgets."""
        if bold:
            return (self._mono_family, pts, "bold")
        return (self._mono_family, pts)

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
            font=self._mono(19, bold=True),
        )
        style.configure(
            "Hint.TLabel",
            background=secondary,
            foreground="#a8a8a8",
            font=self._mono(11),
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
            font=self._mono(12, bold=True),
        )
        style.configure("TLabel", background=theme_a, foreground=fg, font=self._mono(12))

        style.configure(
            "TEntry",
            fieldbackground=inner_entry,
            foreground=fg,
            bordercolor=theme_b,
            insertcolor=fg,
            lightcolor=theme_b,
            darkcolor=theme_a,
            font=self._mono(12),
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
            font=self._mono(11),
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
            font=self._mono(12, bold=True),
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
            font=self._mono(11),
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
            font=self._mono(12),
        )

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=10, style="Chrome.TFrame")
        main.pack(fill=tk.BOTH, expand=True)

        head = ttk.Frame(main, style="Chrome.TFrame")
        head.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(head, text="patchcraftr", style="AccentTitle.TLabel").pack(
            side=tk.LEFT, anchor="w"
        )
        ttk.Button(head, text="+ new patch", command=self._confirm_new_patch).pack(
            side=tk.RIGHT, anchor="e", padx=(12, 0)
        )

        hint = (
            "Build an instrument or FX chain to use in dronmakr UI or `generate-drone` CLI"
        )
        ttk.Label(
            main, text=hint, style="Hint.TLabel", wraplength=1000, justify="left"
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
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(
            bf_inst,
            text="Configure plugin list",
            command=self._open_ignore_plugins_dialog,
        ).pack(side=tk.LEFT)

        fx_fr = ttk.LabelFrame(
            main, text=f"FX chain ({MAX_CHAIN_SLOTS})", padding=10
        )
        fx_fr.pack(fill=tk.X, pady=(0, 8))

        for i in range(MAX_CHAIN_SLOTS):
            row = ttk.Frame(fx_fr)
            row.pack(fill=tk.X, pady=4)
            ttk.Label(row, text=f"{i + 1}.", width=3).pack(side=tk.LEFT, anchor="nw", pady=2)
            lb = ttk.Label(row, text="(empty)")
            lb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 8), anchor="w")
            self._slot_labels.append(lb)
            btn_row = ttk.Frame(row)
            btn_row.pack(side=tk.RIGHT)
            ttk.Button(btn_row, text="Select FX", command=lambda ix=i: self._fx_set_slot(ix)).pack(
                side=tk.LEFT, padx=(0, 4)
            )
            ttk.Button(btn_row, text="Open plugin", command=lambda ix=i: self._fx_edit_slot(ix)).pack(
                side=tk.LEFT, padx=(0, 4)
            )
            ttk.Button(btn_row, text="Clear", command=lambda ix=i: self._fx_clear_slot(ix)).pack(
                side=tk.LEFT, padx=0
            )

        prev_fr = ttk.LabelFrame(main, text="Preview Settings", padding=10)
        prev_fr.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(prev_fr, text="Instrument MIDI preview", font=self._mono(11)).pack(
            anchor="w"
        )
        pc = ttk.Frame(prev_fr)
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
            prev_fr,
            text="FX-only dry signal (used when no instrument is loaded)",
            font=self._mono(11),
        ).pack(anchor="w", pady=(10, 0))
        fx_pc = ttk.Frame(prev_fr)
        fx_pc.pack(fill=tk.X)
        for jx, (sid, label) in enumerate(FX_PREVIEW_SOURCES):
            ttk.Radiobutton(
                fx_pc,
                text=label,
                value=sid,
                variable=self._fx_preview_source,
                command=self._sync_editor_fx_dry_source_ref,
            ).grid(row=jx // 2, column=jx % 2, sticky="w", padx=(0, 12), pady=2)

        ttk.Button(
            prev_fr,
            text="Rendered preview (offline WAV playback)",
            command=self._rendered_preview_fallback,
        ).pack(anchor="w", pady=(10, 4))
        ttk.Label(
            prev_fr,
            text="Uses the MIDI / FX dry choices above. Handy when live monitor audio glitches.",
            style="Hint.TLabel",
            wraplength=1000,
            justify="left",
        ).pack(anchor="w")

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
        while True:
            try:
                job = self._main_thread_jobs.get_nowait()
            except queue.Empty:
                break
            try:
                job()
            except Exception:
                _LOG.exception("Patchcraftr UI job failed")

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

    def _rendered_preview_fallback(self) -> None:
        thr = getattr(self, "_offline_preview_thread", None)
        if thr is not None and thr.is_alive():
            _LOG.info("Rendered preview: already running, ignored click")
            messagebox.showinfo(
                "patchcraftr",
                "A preview is already rendering.",
                parent=self,
            )
            return
        _LOG.info("UI: rendered preview (offline WAV) requested")
        self._sync_editor_midi_style_ref()
        self._sync_editor_fx_dry_source_ref()
        has_inst = self._inst_plugin is not None and bool(self._inst_plugin_path)
        has_fx = any(s is not None for s in self._fx_slots)
        if not has_inst and not has_fx:
            messagebox.showwarning(
                "patchcraftr",
                "Choose an instrument plug-in and/or fill at least one FX slot.",
                parent=self,
            )
            return
        if has_inst:
            if self._inst_plugin is None:
                messagebox.showwarning(
                    "patchcraftr",
                    "Instrument plug-in instance is missing. Choose the instrument again.",
                    parent=self,
                )
                return
            if not str(self._inst_plugin_path).strip():
                messagebox.showwarning(
                    "patchcraftr",
                    "Instrument plug-in path is missing; pick again.",
                    parent=self,
                )
                return

        self._offline_preview_thread = threading.Thread(
            target=self._rendered_preview_worker,
            args=(has_inst,),
            daemon=True,
        )
        _LOG.info("Rendered preview: background thread started (has_inst=%s)", has_inst)
        self._offline_preview_thread.start()

    def _rendered_preview_worker(self, has_inst: bool) -> None:
        path = ""
        try:
            import soundfile as sf  # noqa: WPS433

            chain = self._ordered_fx_chain_plugins()
            if has_inst:
                # Same routing as live preview: synth → every FX slot (including empty chain).
                plugin = Pedalboard(chain)
                upstream = self._inst_plugin
            else:
                plugin = Pedalboard(chain)
                upstream = None

            arr = render_preview_clip(
                plugin,
                duration_sec=5.0,
                midi_style_ref=self._editor_midi_style_ref,
                upstream_instrument=upstream,
                fx_preview_source_ref=self._editor_fx_source_ref,
            )
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            sf.write(path, arr, SAMPLE_RATE, subtype="PCM_16")
            self._play_wav_path(path)
            _LOG.info("Rendered preview: WAV written and playback requested")
        except Exception as e:
            _LOG.exception("Rendered preview worker failed: %s", e)
            self.after(
                0,
                lambda err=str(e): self._enqueue_msg(
                    "error", f"Rendered preview failed:\n{err}"
                ),
            )
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _play_wav_path(self, path: str) -> None:
        if sys.platform == "darwin":
            subprocess.run(["afplay", path], check=False)
        elif sys.platform.startswith("win"):
            env = os.environ.copy()
            env["PATCHCRAFTR_WAV"] = path
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        '$p = $env:PATCHCRAFTR_WAV; '
                        "$s = New-Object System.Media.SoundPlayer($p); "
                        "$s.PlaySync()"
                    ),
                ],
                env=env,
                check=False,
                capture_output=True,
            )
        else:
            r = subprocess.run(["aplay", path], check=False, capture_output=True)
            if r.returncode != 0:
                subprocess.run(["xdg-open", path], check=False)

    def _validate_loaded_plugin_as_instrument(self, plug: Any) -> str | None:
        """Return a user-facing error string if ``plug`` cannot be used as an instrument slot."""
        if getattr(plug, "is_instrument", False):
            return None
        if getattr(plug, "is_effect", False):
            return (
                "This plug-in loads as an effect processor, not as an instrument. "
                'Add it with “Select FX” instead of “Choose instrument”.'
            )
        return (
            "This plug-in does not report an instrument interface. Pick a synth or sampler."
        )

    def _validate_loaded_plugin_as_fx_chain(self, plug: Any, pname: str) -> str | None:
        """Return a user-facing error if ``plug`` cannot be used in an FX chain slot."""
        if slot_allowed_as_chain_effect(plug, pname, self._assert_instrument):
            return None
        pn_st = (pname or "").strip()
        if getattr(plug, "is_effect", False) and pn_st and pn_st in self._assert_instrument:
            return (
                "That plug-in variant is listed under ASSERT_INSTRUMENT in Settings and is treated "
                "as an instrument here. It cannot be placed in an FX slot."
            )
        if getattr(plug, "is_instrument", False) and not getattr(plug, "is_effect", False):
            return (
                "This plug-in reports as an instrument only. "
                'Use “Choose instrument” instead of “Select FX”.'
            )
        if not getattr(plug, "is_effect", False):
            return (
                "This plug-in does not report as an effect processor here. Choose a different FX, "
                "or assign it via “Choose instrument” if it is a synth."
            )
        return "This plug-in cannot be used as an FX chain step with the current Settings."

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
        short = format_plugin_name(path)
        while True:
            try:
                with self._modal_busy(f"Loading plug-in…\n{short}"):
                    plug, pname = load_pedalboard_plugin(path, cur_hint or None)
                pname_s = pname or getattr(plug, "name", "") or ""
                _LOG.info(
                    "Pedalboard load_plugin OK path=%r hint=%s plugin_name=%r is_inst=%s is_effect=%s",
                    path,
                    cur_hint if cur_hint is not None else "(default)",
                    pname_s[:120] if pname_s else "",
                    getattr(plug, "is_instrument", "?"),
                    getattr(plug, "is_effect", "?"),
                )
                return plug, pname
            except PluginVariantRequired as e:
                _LOG.info(
                    "Plug-in asks for variant (path=%r): choose among %s",
                    format_plugin_name(e.plugin_path),
                    ", ".join(e.variants[:8]) + ("…" if len(e.variants) > 8 else ""),
                )
                pick = self._resolve_variant(e.plugin_path, e.variants)
                if pick is None:
                    _LOG.info("User cancelled variant picker for %r", short)
                    return None
                cur_hint = pick
            except Exception:
                _LOG.exception("load_pedalboard_plugin failed (%r)", path)
                raise

    def _pick_plugin_path(
        self,
        title: str,
        *,
        sorted_labels: list[str] | None = None,
        plugin_map: dict[str, str] | None = None,
    ) -> str | None:
        labels_sorted = self._plugin_labels_sorted if sorted_labels is None else sorted_labels
        pmap = self._plugin_map if plugin_map is None else plugin_map
        _LOG.debug("Plug-in picker open: %s (%s titles)", title, len(labels_sorted))
        if not labels_sorted:
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
        dlg.geometry("560x560")
        dlg.grab_set()

        filt = tk.StringVar()
        ttk.Entry(dlg, textvariable=filt).pack(fill=tk.X, padx=8, pady=(8, 4))

        lb = tk.Listbox(dlg, width=52, height=18)
        self._theme_listbox(lb)
        lb.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        labels = list(labels_sorted)

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
                out[0] = pmap.get(lab)
            dlg.destroy()

        bf = ttk.Frame(dlg)
        bf.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(bf, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT, padx=8)
        ttk.Button(bf, text="OK", command=ok).pack(side=tk.RIGHT)

        dlg.wait_window()
        return out[0]

    def _saved_effect_presets_rows(self) -> list[dict]:
        return [p for p in load_presets_json() if p.get("type") == "effect"]

    def _saved_effect_preset_display_rows(self) -> list[tuple[str, dict]]:
        rows = self._saved_effect_presets_rows()
        if not rows:
            return []
        name_counts = defaultdict(int)
        for r in rows:
            nm_raw = str(r.get("name", "") or "").strip()
            canon = nm_raw if nm_raw else "(unnamed)"
            name_counts[canon] += 1
        display_data: list[tuple[str, dict]] = []
        for r in sorted(rows, key=lambda x: str(x.get("name", "")).lower()):
            nm_raw = str(r.get("name", "") or "").strip()
            canon = nm_raw if nm_raw else "(unnamed)"
            disp = canon
            if name_counts[canon] > 1:
                rid = str(r.get("id", ""))[:8]
                disp = f"{disp} [{rid}]"
            display_data.append((disp, r))
        return display_data

    def _fx_set_slot(self, idx: int) -> None:
        _LOG.info("UI: FX slot %s — scanning plug-ins before opening picker", idx + 1)

        def work() -> None:
            try:
                t0 = time.time()
                m = build_plugin_label_map()
                scan_s = time.time() - t0
            except Exception:
                _LOG.exception("FX slot %s: plug-in catalog scan failed", idx + 1)
                self._schedule_ui(
                    lambda: self._enqueue_msg(
                        "error",
                        f"Could not enumerate plug-ins for FX slot {idx + 1}.\nSee errors.log for details.",
                    )
                )
                return

            self._schedule_ui(lambda: self._resume_fx_set_slot_dialog(idx, m, scan_s))

        threading.Thread(target=work, daemon=True).start()

    def _resume_fx_set_slot_dialog(self, idx: int, m: dict[str, str], scan_s: float) -> None:
        self._refresh_plugin_map_apply(m)
        _LOG.info(
            "FX slot %s picker: catalog refreshed (%s plug-ins, scan %.2fs)",
            idx + 1,
            len(self._plugin_labels_sorted),
            scan_s,
        )
        presets_ready = bool(self._saved_effect_presets_rows())
        plugins_ok = bool(self._plugin_labels_sorted)

        if not plugins_ok and not presets_ready:
            messagebox.showwarning(
                "patchcraftr",
                "No plug-ins discovered under PLUGIN_PATHS and no saved single-effect presets.",
                parent=self,
            )
            return

        lbls_ordered = sorted(m.keys(), key=str.lower)

        dlg = tk.Toplevel(self)
        self._theme_dialog_shell(dlg)
        dlg.title(f"FX slot {idx + 1} — plug-ins or saved presets")
        dlg.transient(self)
        dlg.geometry("980x640")
        dlg.minsize(760, 480)
        dlg.grab_set()

        root = ttk.Frame(dlg, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(2, weight=1)

        ttk.Label(root, text="VST/AU plug-ins", font=self._mono(11, bold=True)).grid(
            row=0, column=0, sticky="nw"
        )
        ttk.Label(root, text="Saved single-effect presets", font=self._mono(11, bold=True)).grid(
            row=0, column=1, sticky="nw", padx=(14, 0)
        )

        filt_pl = tk.StringVar()
        filt_pr = tk.StringVar()
        ttk.Entry(root, textvariable=filt_pl).grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(4, 6))
        ttk.Entry(root, textvariable=filt_pr).grid(row=1, column=1, sticky="ew", padx=(14, 0), pady=(4, 6))

        left_sf = ttk.Frame(root)
        left_sf.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        lb_pl_sb = ttk.Scrollbar(left_sf)
        lb_pl = tk.Listbox(
            left_sf, height=24, width=44, yscrollcommand=lb_pl_sb.set, exportselection=False
        )
        lb_pl_sb.config(command=lb_pl.yview)
        self._theme_listbox(lb_pl)
        lb_pl.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lb_pl_sb.pack(side=tk.RIGHT, fill=tk.Y)

        right_sf = ttk.Frame(root)
        right_sf.grid(row=2, column=1, sticky="nsew", padx=(14, 0))
        lb_pr_sb = ttk.Scrollbar(right_sf)
        lb_pr = tk.Listbox(
            right_sf, height=24, width=46, yscrollcommand=lb_pr_sb.set, exportselection=False
        )
        lb_pr_sb.config(command=lb_pr.yview)
        self._theme_listbox(lb_pr)
        lb_pr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lb_pr_sb.pack(side=tk.RIGHT, fill=tk.Y)

        all_plugin_labels = list(lbls_ordered)
        preset_pairs = self._saved_effect_preset_display_rows()
        preset_visible: list[dict] = []

        def refill_plugins(filter_txt: str = "") -> None:
            lb_pl.delete(0, tk.END)
            ft = filter_txt.strip().lower()
            for lab in all_plugin_labels:
                if ft in lab.lower():
                    lb_pl.insert(tk.END, lab)

        def refill_presets(filter_txt: str = "") -> None:
            lb_pr.delete(0, tk.END)
            preset_visible.clear()
            ft = filter_txt.strip().lower()
            for lbl, row in preset_pairs:
                pname = format_plugin_name(str(row.get("plugin_path", "") or ""))
                blob = " ".join(
                    filter(
                        None,
                        [
                            lbl.lower(),
                            str(row.get("name", "")).lower(),
                            pname.lower(),
                            str(row.get("plugin_path", "")).lower(),
                        ],
                    )
                )
                if not ft or ft in blob:
                    lb_pr.insert(tk.END, lbl)
                    preset_visible.append(row)

        refill_plugins()
        refill_presets()

        def on_filt_pl(*_a: Any) -> None:
            refill_plugins(filt_pl.get())

        def on_filt_pr(*_a: Any) -> None:
            refill_presets(filt_pr.get())

        filt_pl.trace_add("write", on_filt_pl)
        filt_pr.trace_add("write", on_filt_pr)

        def on_sel_pl(*_ignored: Any) -> None:
            if lb_pl.curselection():
                lb_pr.selection_clear(0, tk.END)

        def on_sel_pr(*_ignored: Any) -> None:
            if lb_pr.curselection():
                lb_pl.selection_clear(0, tk.END)

        lb_pl.bind("<<ListboxSelect>>", on_sel_pl)
        lb_pr.bind("<<ListboxSelect>>", on_sel_pr)

        bf = ttk.Frame(dlg)
        bf.pack(fill=tk.X, pady=(8, 8), padx=10)

        pmap_snap = m
        browse_path_holder: dict[str, str | None] = {"path": None}
        preset_row_holder: dict[str, Any | None] = {"row": None}

        def ok_pick() -> None:
            psi = lb_pl.curselection()
            pri = lb_pr.curselection()
            if psi and pri:
                messagebox.showinfo(
                    "patchcraftr",
                    "Pick either one plug-in (left) or one saved preset (right).",
                    parent=dlg,
                )
                return
            if psi:
                lab = lb_pl.get(psi[0])
                browse_path_holder["path"] = pmap_snap.get(lab)
            elif pri:
                ix = pri[0]
                if ix < len(preset_visible):
                    preset_row_holder["row"] = preset_visible[ix]
            else:
                messagebox.showwarning("patchcraftr", "Choose a plug-in or a preset.", parent=dlg)
                return
            dlg.destroy()

        ttk.Button(bf, text="OK", command=ok_pick).pack(side=tk.RIGHT, padx=(0, 8))
        ttk.Button(bf, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT)

        dlg.wait_window()

        if browse_path_holder["path"]:
            p = browse_path_holder["path"]
            self.after(80, lambda i=idx, pa=p: self._fx_apply_browse_path(i, pa))
        elif preset_row_holder["row"] is not None:
            prow = preset_row_holder["row"]
            self.after(80, lambda i=idx, r=prow: self._fx_apply_saved_effect_row(i, r))

    def _fx_apply_saved_effect_row(self, idx: int, row: dict, *, open_editor: bool = True) -> None:
        _LOG.info(
            "FX slot %s: load from saved preset row id=%s plugin=%r",
            idx + 1,
            str(row.get("id", ""))[:12],
            format_plugin_name(str(row.get("plugin_path", "") or "")),
        )
        try:
            loaded = self._load_plugin_with_variants(
                row["plugin_path"], row.get("plugin_name") or None
            )
            if loaded is None:
                return
            plug, pname = loaded
            err = self._validate_loaded_plugin_as_fx_chain(plug, pname)
            if err:
                messagebox.showwarning("patchcraftr", err, parent=self)
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
            if open_editor:
                self.after(120, lambda i=idx: self._fx_edit_slot(i))
        except Exception as e:
            self._enqueue_msg("error", str(e))

    def _fx_apply_browse_path(
        self, idx: int, path: str | None = None, *, open_editor: bool = True
    ) -> None:
        if path is None:
            path = self._pick_plugin_path(f"FX slot {idx + 1}")
        if not path:
            return
        _LOG.info("FX slot %s: load browsed plug-in %r", idx + 1, format_plugin_name(path))
        try:
            loaded = self._load_plugin_with_variants(path)
            if loaded is None:
                return
            plug, pname = loaded
            err = self._validate_loaded_plugin_as_fx_chain(plug, pname)
            if err:
                messagebox.showwarning("patchcraftr", err, parent=self)
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
            if open_editor:
                self.after(120, lambda i=idx: self._fx_edit_slot(i))
        except Exception as e:
            self._enqueue_msg("error", str(e))

    def _inst_choose_plugin(self) -> None:
        _LOG.info("UI: Choose instrument clicked")

        def work() -> None:
            try:
                assert_plugin_paths_configured()
            except PresetAuthoringConfigError as e:
                self._schedule_ui(
                    lambda err=str(e): messagebox.showerror("patchcraftr", err, parent=self)
                )
                return
            try:
                t0 = time.time()
                m = build_plugin_label_map()
                scan_s = time.time() - t0
            except Exception:
                _LOG.exception("Choose instrument: catalog scan failed")
                self._schedule_ui(
                    lambda: self._enqueue_msg(
                        "error",
                        "Could not enumerate plug-ins for Pick instrument.\nSee errors.log for details.",
                    )
                )
                return

            self._schedule_ui(
                lambda: self._resume_instrument_choose_after_catalog(m, scan_s),
            )

        threading.Thread(target=work, daemon=True).start()

    def _resume_instrument_choose_after_catalog(self, m: dict[str, str], scan_s: float) -> None:
        self._refresh_plugin_map_apply(m)
        lbls = self._plugin_labels_sorted
        _LOG.info(
            "Choose instrument picker: catalog %s plugs (scan %.2fs)",
            len(lbls),
            scan_s,
        )
        path = self._pick_plugin_path(
            "Instrument plug-in",
            sorted_labels=lbls,
            plugin_map=m,
        )
        if not path:
            _LOG.info("Choose instrument: cancelled or empty picker")
            return
        try:
            _LOG.info(
                "Choose instrument: selected %r (%s)",
                path,
                format_plugin_name(path),
            )
            loaded = self._load_plugin_with_variants(path)
            if loaded is None:
                return
            plug, pname = loaded
            err = self._validate_loaded_plugin_as_instrument(plug)
            if err:
                messagebox.showwarning("patchcraftr", err, parent=self)
                return
            self._inst_plugin_path = path
            self._inst_plugin_name = pname or ""
            self._inst_plugin = plug
            self._inst_editor_was_shown_before = False
            self._inst_plugin_lbl.configure(
                text=f"{format_plugin_name(path)} · {self._inst_plugin_name or 'default'}"
            )
            self.after(80, self._inst_edit_plugin)
        except Exception as e:
            self._enqueue_msg("error", str(e))

    def _run_plugin_editor(
        self,
        editor_plugin: Any,
        *,
        instrument_editor: bool = False,
        audio_engine: Any | None = None,
        fx_slot_index: int | None = None,
    ) -> None:
        if self._editor_active:
            messagebox.showwarning(
                "patchcraftr",
                "A plug-in editor is already open.",
                parent=self,
            )
            return

        audio_engine_effective = audio_engine
        if instrument_editor:
            # Preview always runs instrument MIDI through the full FX chain (empty chain = pass-through).
            audio_engine_effective = Pedalboard(self._ordered_fx_chain_plugins())

        if audio_engine_effective is None:
            monitor_target = editor_plugin
        else:
            monitor_target = audio_engine_effective
        upstream_inst = None
        if audio_engine_effective is not None and self._inst_plugin is not None:
            upstream_inst = self._inst_plugin

        close_evt = threading.Event()
        self._editor_active = True
        _LOG.info(
            "Opening Pedalboard show_editor instrument=%s fx_slot=%s",
            instrument_editor,
            fx_slot_index,
        )
        self._sync_editor_midi_style_ref()
        self._sync_editor_fx_dry_source_ref()
        monitor = PatchcraftrLiveMonitor(
            monitor_target,
            midi_style_ref=self._editor_midi_style_ref,
            upstream_instrument=upstream_inst,
            fx_preview_source_ref=self._editor_fx_source_ref,
        )
        monitor.start()
        _LOG.info(
            "Live preview monitor running (instrument_editor=%s fx_slot=%s)",
            instrument_editor,
            fx_slot_index,
        )
        try:
            editor_plugin.show_editor(close_evt)
        except BaseException as e:
            self._enqueue_msg("error", f"Editor error:\n{e}")
        finally:
            monitor.stop()
            self._editor_active = False
            _LOG.info("Closed Pedalboard editor (instrument=%s slot=%s)", instrument_editor, fx_slot_index)
            if instrument_editor:
                self._inst_editor_was_shown_before = True
            if fx_slot_index is not None:
                reopened = self._fx_slots[fx_slot_index]
                if reopened is not None:
                    reopened.editor_opened_before = True

    def _inst_edit_plugin(self) -> None:
        _LOG.info("UI: Edit instrument clicked")
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
                try:
                    with self._modal_busy(
                        "Refreshing instrument plug-in…\n" + format_plugin_name(self._inst_plugin_path),
                    ):
                        wp, pname = reload_pedalboard_plugin_preserving_state(
                            self._inst_plugin_path,
                            self._inst_plugin_name or None,
                            old,
                        )
                except PluginVariantRequired as e:
                    pick = self._resolve_variant(e.plugin_path, e.variants)
                    if pick is None:
                        return
                    with self._modal_busy(
                        "Loading instrument variant…\n" + format_plugin_name(self._inst_plugin_path),
                    ):
                        wp, pname = load_pedalboard_plugin(self._inst_plugin_path, pick)
                    apply_vstpreset_bytes_to_plugin(wp, blob)
            except Exception as e:
                self._enqueue_msg(
                    "error",
                    "Could not refresh instrument before reopening its editor:\n"
                    f"{e}",
                )
                return
            ierr = self._validate_loaded_plugin_as_instrument(wp)
            if ierr:
                messagebox.showwarning("patchcraftr", ierr, parent=self)
                return
            self._inst_plugin = wp
            self._inst_plugin_name = pname or ""
            self._inst_plugin_lbl.configure(
                text=f"{format_plugin_name(self._inst_plugin_path)} · {self._inst_plugin_name or 'default'}"
            )

        self._run_plugin_editor(self._inst_plugin, instrument_editor=True)

    def _reload_fx_slot_preserving_editor_reopen(self, idx: int) -> bool:
        """Re-instantiate the slot plug-in if its editor has been opened before (VST preview fix)."""
        st = self._fx_slots[idx]
        if st is None or not st.editor_opened_before:
            return True
        old = st.plugin
        blob = serialize_plugin_preset_bytes(old)
        try:
            try:
                with self._modal_busy(
                    "Refreshing FX plug-in…\n" + format_plugin_name(st.plugin_path),
                ):
                    wp, pname = reload_pedalboard_plugin_preserving_state(
                        st.plugin_path,
                        st.plugin_name or None,
                        old,
                    )
            except PluginVariantRequired as e:
                pick = self._resolve_variant(e.plugin_path, e.variants)
                if pick is None:
                    return False
                with self._modal_busy(
                    "Loading FX variant…\n" + format_plugin_name(st.plugin_path),
                ):
                    wp, pname = load_pedalboard_plugin(st.plugin_path, pick)
                apply_vstpreset_bytes_to_plugin(wp, blob)
        except Exception as e:
            self._enqueue_msg(
                "error",
                f"Could not refresh FX plug-in before reopening editor:\n{e}",
            )
            return False

        eff_err = self._validate_loaded_plugin_as_fx_chain(wp, pname or "")
        if eff_err:
            messagebox.showwarning("patchcraftr", eff_err, parent=self)
            return False

        self._fx_slots[idx] = FxSlotState(
            plugin_path=st.plugin_path,
            plugin_name=(pname or st.plugin_name or "").strip(),
            plugin=wp,
            effect_display_name=st.effect_display_name,
            effect_uid=st.effect_uid,
            preset_path=st.preset_path,
            editor_opened_before=st.editor_opened_before,
        )
        self._refresh_slot_labels()
        return True

    def _ordered_fx_chain_plugins(self) -> list[Any]:
        return [s.plugin for s in self._fx_slots if s is not None]

    def _fx_edit_slot(self, idx: int) -> None:
        _LOG.info("UI: Open FX plug-in editor for slot %s", idx + 1)
        st = self._fx_slots[idx]
        if st is None:
            messagebox.showwarning("patchcraftr", "Slot is empty.", parent=self)
            return
        if not self._reload_fx_slot_preserving_editor_reopen(idx):
            return
        st = self._fx_slots[idx]
        assert st is not None
        chain = self._ordered_fx_chain_plugins()
        audio_engine = Pedalboard(chain)

        self._run_plugin_editor(
            st.plugin,
            instrument_editor=False,
            audio_engine=audio_engine,
            fx_slot_index=idx,
        )

    def _fx_clear_slot(self, idx: int) -> None:
        _LOG.info("UI: clear FX slot %s", idx + 1)
        self._fx_slots[idx] = None
        self._refresh_slot_labels()

    def _save_patch(self) -> None:
        _LOG.info("UI: Save patch clicked")

        has_inst = self._inst_plugin is not None and bool(self._inst_plugin_path)

        has_fx = any(s is not None for s in self._fx_slots)

        nm_inst = self._inst_name.get().strip()
        nm_fx = self._chain_name.get().strip()

        if not has_inst and not has_fx:
            messagebox.showwarning(
                "patchcraftr",
                "Choose an instrument plug-in and/or fill at least one FX slot.",
                parent=self,
            )
            return
        if has_inst:
            if self._inst_plugin is None:
                messagebox.showwarning(
                    "patchcraftr",
                    "Instrument plug-in instance is missing. Choose the instrument again.",
                    parent=self,
                )
                return
            if not str(self._inst_plugin_path).strip():
                messagebox.showwarning(
                    "patchcraftr",
                    "Instrument plug-in path is missing; pick again.",
                    parent=self,
                )
                return
            if not nm_inst:
                messagebox.showwarning(
                    "patchcraftr",
                    "Enter a patch name under “Instrument name” (cannot be blank).",
                    parent=self,
                )
                return
        if has_fx and not nm_fx:
            messagebox.showwarning(
                "patchcraftr",
                "Enter a name under “Effect name” — it cannot be blank when saving FX.",
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

        tuples: list[tuple[str, str, Any, tuple[str, str, str, str]]] = []
        try:
            with self._modal_busy("Saving patch…"):
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

                has_fx_saves = len(tuples) > 0

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

                if has_fx_saves:
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
        if tuples:
            parts.append(f"effect “{nm_fx}”" + (" [chain]" if len(tuples) > 1 else " [single FX]"))

        self._enqueue_msg("info", "Saved " + " and ".join(parts) + ".")
        _LOG.info(
            "Save patch completed has_inst=%s fx_slots_written=%s",
            has_inst,
            len(tuples),
        )

        try:
            t0 = time.time()
            self._refresh_plugin_map()
            _LOG.debug("Plug-in map refresh after save %.3fs", time.time() - t0)
        except Exception:
            _LOG.exception("Post-save plug-in map refresh failed")

    @staticmethod
    def _parse_ignore_plugins_setting(raw: str) -> list[str]:
        items = [x.strip() for x in (raw or "").split(",") if x.strip()]
        seen: set[str] = set()
        ordered: list[str] = []
        for x in items:
            if x not in seen:
                seen.add(x)
                ordered.append(x)
        return ordered

    @staticmethod
    def _label_matches_ignore_pattern(label: str, patterns: list[str]) -> bool:
        return any(p and p in label for p in patterns)

    def _open_ignore_plugins_dialog(self) -> None:
        _LOG.info("UI: IGNORE_PLUGINS configure clicked")
        plugin_dirs, _, _, custom_plugins = plugin_settings_tuple()
        if not plugin_dirs or plugin_dirs == [""]:
            messagebox.showwarning(
                "patchcraftr",
                "PLUGIN_PATHS is empty. Configure plug-in scan directories in Settings first.",
                parent=self,
            )
            return

        def work() -> None:
            try:
                t0 = time.time()
                all_paths = list_installed_plugins(plugin_dirs, custom_plugins)
                labels_map: dict[str, str] = {}
                for p in all_paths:
                    lab = format_plugin_name(p)
                    labels_map.setdefault(lab, p)
                scan_s = time.time() - t0
            except Exception:
                _LOG.exception("IGNORE_PLUGINS: list_installed_plugins failed")
                self._schedule_ui(
                    lambda: self._enqueue_msg(
                        "error",
                        "Could not scan plug-ins for IGNORE_PLUGINS dialog.\nSee errors.log.",
                    )
                )
                return
            frozen = dict(labels_map)
            self._schedule_ui(
                lambda: self._present_ignore_plugins_dialog(frozen, scan_s),
            )

        threading.Thread(target=work, daemon=True).start()

    def _present_ignore_plugins_dialog(self, labels_by_display: dict[str, str], scan_s: float) -> None:
        _LOG.info(
            "IGNORE_PLUGINS dialog scan done: %s labels in %.2fs",
            len(labels_by_display),
            scan_s,
        )
        dlg = tk.Toplevel(self)
        self._theme_dialog_shell(dlg)
        dlg.title("Configure plugin list (IGNORE_PLUGINS)")
        dlg.transient(self)
        dlg.geometry("940x620")
        dlg.minsize(720, 480)
        dlg.grab_set()

        intro = (
            "Left: plug-ins offered in selectors. Right: ignore substrings saved to IGNORE_PLUGINS "
            "(same matching as preset authoring: a pattern hides any plug-in whose display name contains it). "
            "Double-click or use the arrows to move items. Save writes settings.json."
        )
        ttk.Label(dlg, text=intro, wraplength=900, justify="left").pack(
            anchor="w", padx=12, pady=(10, 6)
        )

        root = ttk.Frame(dlg, padding=(10, 0))
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=0)
        root.columnconfigure(2, weight=1)
        root.rowconfigure(1, weight=1)

        ttk.Label(root, text="Available plug-ins", font=self._mono(11, bold=True)).grid(
            row=0, column=0, sticky="nw"
        )
        ttk.Label(root, text="Ignored (IGNORE_PLUGINS)", font=self._mono(11, bold=True)).grid(
            row=0, column=2, sticky="nw", padx=(12, 0)
        )

        left_sf = ttk.Frame(root)
        left_sf.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        sb_l = ttk.Scrollbar(left_sf)
        lb_avail = tk.Listbox(
            left_sf, height=24, exportselection=False, yscrollcommand=sb_l.set
        )
        sb_l.config(command=lb_avail.yview)
        self._theme_listbox(lb_avail)
        lb_avail.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_l.pack(side=tk.RIGHT, fill=tk.Y)

        mid = ttk.Frame(root)
        mid.grid(row=1, column=1, padx=8, pady=(4, 0))

        right_sf = ttk.Frame(root)
        right_sf.grid(row=1, column=2, sticky="nsew", padx=(0, 0), pady=(4, 0))
        sb_r = ttk.Scrollbar(right_sf)
        lb_ign = tk.Listbox(
            right_sf, height=24, exportselection=False, yscrollcommand=sb_r.set
        )
        sb_r.config(command=lb_ign.yview)
        self._theme_listbox(lb_ign)
        lb_ign.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_r.pack(side=tk.RIGHT, fill=tk.Y)

        ignored: list[str] = list(
            self._parse_ignore_plugins_setting(load_settings().get("IGNORE_PLUGINS", ""))
        )
        dirty: list[bool] = [False]

        def refresh_lists() -> None:
            lbls = sorted(labels_by_display.keys(), key=str.lower)
            lb_avail.delete(0, tk.END)
            for lab in lbls:
                if not self._label_matches_ignore_pattern(lab, ignored):
                    lb_avail.insert(tk.END, lab)
            lb_ign.delete(0, tk.END)
            for pat in sorted(ignored, key=str.lower):
                lb_ign.insert(tk.END, pat)

        def mark_dirty() -> None:
            dirty[0] = True

        def move_selection_to_ignored(_evt: Any = None) -> None:
            sel = lb_avail.curselection()
            if not sel:
                return
            lab = lb_avail.get(sel[0])
            if lab not in ignored:
                ignored.append(lab)
                mark_dirty()
                refresh_lists()

        def move_selection_to_allowed(_evt: Any = None) -> None:
            sel = lb_ign.curselection()
            if not sel:
                return
            pat = lb_ign.get(sel[0])
            if pat in ignored:
                ignored.remove(pat)
                mark_dirty()
                refresh_lists()

        def ignore_all() -> None:
            merged = sorted(set(ignored) | set(labels_by_display.keys()), key=str.lower)
            ignored.clear()
            ignored.extend(merged)
            mark_dirty()
            refresh_lists()

        def allow_all() -> None:
            if ignored:
                ignored.clear()
                mark_dirty()
                refresh_lists()

        refresh_lists()

        ttk.Button(mid, text="Ignore →", width=14, command=move_selection_to_ignored).pack(
            pady=(0, 6)
        )
        ttk.Button(mid, text="← Allow", width=14, command=move_selection_to_allowed).pack(
            pady=(0, 6)
        )
        ttk.Button(mid, text="Ignore all", width=14, command=ignore_all).pack(pady=(0, 6))
        ttk.Button(mid, text="Allow all", width=14, command=allow_all).pack()

        lb_avail.bind("<Double-Button-1>", move_selection_to_ignored)
        lb_ign.bind("<Double-Button-1>", move_selection_to_allowed)

        bf = ttk.Frame(dlg)
        bf.pack(fill=tk.X, pady=(10, 12), padx=12)

        def save_ignores() -> None:
            settings = load_settings()
            settings["IGNORE_PLUGINS"] = ",".join(ignored)
            save_settings(settings)
            self._refresh_plugin_map()
            dirty[0] = False
            dlg.destroy()
            messagebox.showinfo(
                "patchcraftr",
                "IGNORE_PLUGINS saved. Plug-in pickers will use the updated list.",
                parent=self,
            )

        def cancel_dialog() -> None:
            if dirty[0] and not messagebox.askyesno(
                "patchcraftr",
                "Discard changes to IGNORE_PLUGINS?",
                parent=dlg,
            ):
                return
            dlg.destroy()

        ttk.Button(bf, text="Save", command=save_ignores).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(bf, text="Cancel", command=cancel_dialog).pack(side=tk.RIGHT)
        dlg.protocol("WM_DELETE_WINDOW", cancel_dialog)

    def _confirm_new_patch(self) -> None:
        _LOG.info("UI: + new patch clicked")
        if not messagebox.askyesno(
            "patchcraftr · new patch",
            "Discard the current form and start blank?\n\n"
            "Instrument, FX slots, and patch names will be cleared.",
            parent=self,
        ):
            _LOG.info("New patch: user cancelled")
            return
        _LOG.info("New patch: form reset")
        self._reset_patch_form()


def _raise_patchcraftr_window(root: tk.Tk) -> None:
    """When launched from the tray or another app, bring this Tk window to the foreground."""
    _LOG.debug("Raising Patchcraftr window (pid=%s platform=%s)", os.getpid(), sys.platform)
    if sys.platform == "darwin":
        try:
            from AppKit import NSApplication  # type: ignore[import-not-found]

            NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        except Exception:
            pass
        try:
            pid = os.getpid()
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    (
                        'tell application "System Events" to '
                        f"set frontmost of first process whose unix id is {pid} to true"
                    ),
                ],
                check=False,
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass
    try:
        root.update_idletasks()
        root.deiconify()
        root.lift()
    except tk.TclError:
        pass
    try:
        root.focus_force()
    except tk.TclError:
        pass


def main() -> None:
    ensure_settings()
    ensure_authoring_dirs()
    path = ensure_server_error_file_logging(announce=sys.stderr.isatty())
    app = PatchcraftrApp()
    _LOG.info("Patchcraftr mainloop starting (errors.log=%s)", path)
    app.after(50, lambda: _raise_patchcraftr_window(app))
    app.mainloop()


if __name__ == "__main__":
    main()
