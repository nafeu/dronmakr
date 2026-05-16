"""Low-latency-tolerant live audio preview while a Pedalboard plugin editor is open."""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from typing import Any, Sequence

import numpy as np
from pedalboard import Pedalboard
from scipy import signal

SAMPLE_RATE = 44100
CHANNELS = 2
BLOCK_DUR_SEC = 0.1
# Gaussian noise RMS scale (±1 floats). Values like 1e-4 are effectively silent on most outputs.
FX_PREVIEW_NOISE_LEVEL = 0.045
# Parallel dry from the instrument when monitoring instrument→FX (hear synth if FX chain is quiet).
_UPSTREAM_DRY_BLEND = 0.14

# FX-only dry signal (no instrument) — IDs are stable API.
FX_PREVIEW_GAUSSIAN = "fx_gaussian_noise"
FX_PREVIEW_SCALE_WAV = "fx_scale_wav"
FX_PREVIEW_SOURCES: list[tuple[str, str]] = [
    (FX_PREVIEW_GAUSSIAN, "Gaussian white noise"),
    (FX_PREVIEW_SCALE_WAV, "CDEFGABC scale (WAV)"),
]
DEFAULT_FX_PREVIEW_SOURCE = FX_PREVIEW_GAUSSIAN

PREVIEW_NOTE_VELOCITY = 96
PREVIEW_RETRIGGER_SEC = 2.0

# (style_id, label) for Patchcraftr UI — IDs are stable API.
MIDI_PREVIEW_STYLES: list[tuple[str, str]] = [
    ("sustained_c4", "Sustained middle C"),
    ("octave_cycle", "Single C · 4 octaves"),
    ("c_major_chord", "C major chord"),
    ("c_scale_up", "C major scale ↑"),
]
DEFAULT_MIDI_PREVIEW_STYLE = MIDI_PREVIEW_STYLES[0][0]


def midi_preview_style_ids() -> tuple[str, ...]:
    return tuple(s for s, _ in MIDI_PREVIEW_STYLES)


def fx_preview_source_ids() -> tuple[str, ...]:
    return tuple(s for s, _ in FX_PREVIEW_SOURCES)


def _resolve_fx_preview_wav_path() -> str:
    """Same resource as preset authoring (`resources/CDEFGABC.wav`)."""
    from preset_authoring import preview_sample_wav_path

    return preview_sample_wav_path()


def _load_wav_loop_mono_stereo(path: str) -> tuple[np.ndarray | None, bool]:
    """Return (samples, 2) float32 at ``SAMPLE_RATE``, or (None, False) if missing/invalid."""
    if not os.path.isfile(path):
        return None, False
    try:
        import soundfile as sf  # noqa: WPS433
    except ImportError:
        return None, False
    try:
        data, fs = sf.read(path, dtype="float32", always_2d=True)
    except OSError:
        return None, False
    if data.size == 0:
        return None, False
    if data.shape[1] == 1:
        data = np.hstack((data, data))
    elif data.shape[1] > CHANNELS:
        data = data[:, :CHANNELS]
    if fs != SAMPLE_RATE:
        num_out = max(1, int(round(data.shape[0] * SAMPLE_RATE / fs)))
        data = signal.resample(data, num_out, axis=0).astype(np.float32, copy=False)
    return np.ascontiguousarray(data, dtype=np.float32), True


class PatchcraftrLiveMonitor:
    """
    Runs a background audio thread that drives the same ``plugin`` instance as ``show_editor``.

    Instruments: continuous synthetic MIDI (style comes from ``midi_style_ref`` each block).
    Instrument ``process(..., reset=...)`` always uses ``reset=False`` because Pedalboard raises
    if a plugin that requires main-thread reload is reset from this worker thread.

    Effects: Gaussian noise stimulus through ``plugin`` when there is no ``upstream_instrument``.

    FX editor with an upstream instrument (``upstream_instrument`` set): MIDI through that
    instrument, then through ``plugin`` (typically a ``Pedalboard`` of every FX slot) so edits
    are heard in context of the full chain.
    """

    def __init__(
        self,
        plugin: Any,
        midi_style_ref: list[str] | None = None,
        *,
        upstream_instrument: Any | None = None,
        fx_preview_source_ref: list[str] | None = None,
    ):
        self._plugin = plugin
        self._upstream_instrument = upstream_instrument
        self._midi_style_ref = midi_style_ref if midi_style_ref is not None else [DEFAULT_MIDI_PREVIEW_STYLE]
        self._fx_preview_source_ref = (
            fx_preview_source_ref if fx_preview_source_ref is not None else [DEFAULT_FX_PREVIEW_SOURCE]
        )
        self._stop = threading.Event()
        self._audio: threading.Thread | None = None
        self._stream_gate = threading.Lock()
        self._playback_stream: Any = None
        self._logged_chain_err = False
        self._logged_upstream_synth_err = False
        self._wav_loop: np.ndarray | None = None
        self._wav_pos = 0
        self._wav_load_attempted = False
        # MIDI preview state (instrument only)
        self._last_preview_notes: tuple[int, ...] = ()
        self._style_cached: str = ""
        self._octave_cycle_i = 0
        self._scale_step_i = 0

    def start(self) -> None:
        self._stop.clear()
        self._audio = threading.Thread(target=self._audio_loop, daemon=True)
        self._audio.start()

    def stop(self) -> None:
        self._stop.set()
        stream = None
        with self._stream_gate:
            stream = self._playback_stream
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass
        if self._audio:
            self._audio.join(timeout=8.0)
            self._audio = None

    def _current_style(self) -> str:
        sid = self._midi_style_ref[0]
        if sid not in midi_preview_style_ids():
            return DEFAULT_MIDI_PREVIEW_STYLE
        return sid

    def _current_fx_preview_source(self) -> str:
        sid = self._fx_preview_source_ref[0]
        if sid not in fx_preview_source_ids():
            return DEFAULT_FX_PREVIEW_SOURCE
        return sid

    def _ensure_wav_loop_loaded(self) -> None:
        if self._wav_load_attempted:
            return
        self._wav_load_attempted = True
        path = _resolve_fx_preview_wav_path()
        data, ok = _load_wav_loop_mono_stereo(path)
        if ok and data is not None:
            self._wav_loop = data
        else:
            self._wav_loop = None
            print(
                f"patchcraftr: FX preview WAV missing or unreadable ({path!r}) — "
                "using Gaussian noise for this session.",
                file=sys.stderr,
            )

    def _next_fx_only_dry_block(self, rng: np.random.Generator, frames: int) -> np.ndarray:
        src = self._current_fx_preview_source()
        if src == FX_PREVIEW_SCALE_WAV:
            self._ensure_wav_loop_loaded()
            loop = self._wav_loop
            if loop is not None and loop.shape[0] > 0:
                n = loop.shape[0]
                out = np.empty((frames, CHANNELS), dtype=np.float32)
                filled = 0
                while filled < frames:
                    take = min(frames - filled, n - self._wav_pos)
                    out[filled : filled + take] = loop[self._wav_pos : self._wav_pos + take]
                    self._wav_pos = (self._wav_pos + take) % n
                    filled += take
                return out
        noise = rng.standard_normal((frames, CHANNELS), dtype=np.float32)
        noise *= FX_PREVIEW_NOISE_LEVEL
        return noise

    def _note_offs(self, mido: Any, notes: Sequence[int], t0: float) -> list[Any]:
        return [
            mido.Message("note_off", note=n, velocity=0, channel=0, time=t0 + i * 1e-4)
            for i, n in enumerate(notes)
        ]

    def _note_ons_staggered(self, mido: Any, notes: Sequence[int], t0: float) -> list[Any]:
        return [
            mido.Message(
                "note_on",
                note=n,
                velocity=PREVIEW_NOTE_VELOCITY,
                channel=0,
                time=t0 + i * 1.5e-3,
            )
            for i, n in enumerate(notes)
        ]

    def _target_notes_for_style(self, style: str) -> tuple[int, ...]:
        if style == "sustained_c4":
            return (60,)
        if style == "octave_cycle":
            # Four C naturals from middle C upward
            roots = (60, 72, 84, 96)
            n = roots[self._octave_cycle_i % len(roots)]
            self._octave_cycle_i += 1
            return (n,)
        if style == "c_major_chord":
            return (60, 64, 67)
        if style == "c_scale_up":
            scale = (60, 62, 64, 65, 67, 69, 71, 72)
            n = scale[self._scale_step_i % len(scale)]
            self._scale_step_i += 1
            return (n,)
        return (60,)

    def _preview_midi_for_block(self, idx: int, block_dur: float) -> list[Any]:
        try:
            import mido  # noqa: WPS433
        except ImportError:
            return []

        retrigger_blocks = max(1, int(round(PREVIEW_RETRIGGER_SEC / block_dur)))
        style = self._current_style()

        msgs: list[Any] = []

        if style != self._style_cached:
            msgs.extend(self._note_offs(mido, self._last_preview_notes, 0.0))
            self._last_preview_notes = ()
            self._style_cached = style
            self._octave_cycle_i = 0
            self._scale_step_i = 0

        retrigger_tick = idx % retrigger_blocks == 0
        if not retrigger_tick and not msgs:
            return []

        if retrigger_tick:
            # Clear previous notes (except first block has no previous)
            if self._last_preview_notes:
                t_base = 0.0 if not msgs else 5e-3
                msgs.extend(self._note_offs(mido, self._last_preview_notes, t_base))
            targets = self._target_notes_for_style(style)
            t_on = 1e-3 * max(1, len(msgs))
            msgs.extend(self._note_ons_staggered(mido, targets, t_on))
            self._last_preview_notes = tuple(targets)

        return msgs

    @staticmethod
    def _buffer_size_arg(frames: int) -> int:
        return min(8192, max(512, frames))

    @staticmethod
    def _shape_outputs_to_stream(arr: Any, frames: int) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.column_stack((arr, arr))
        elif arr.ndim == 2 and arr.shape[0] != frames and arr.shape[1] == frames:
            arr = arr.T
        if arr.shape[0] != frames:
            if arr.shape[0] > frames:
                arr = arr[:frames]
            else:
                pad = np.zeros((frames - arr.shape[0], CHANNELS), dtype=np.float32)
                arr = np.vstack((arr, pad))
        if arr.shape[1] < CHANNELS:
            arr = np.column_stack([arr[:, 0]] * CHANNELS)
        elif arr.shape[1] > CHANNELS:
            arr = arr[:, :CHANNELS]
        return arr

    def _synth_upstream_block(self, idx: int, frames: int, block_dur: float, sr: float) -> np.ndarray:
        upstream = self._upstream_instrument
        assert upstream is not None
        bs = self._buffer_size_arg(frames)
        midis = self._preview_midi_for_block(idx, block_dur)
        try:
            # ExternalPlugin/NI Massive and others forbid reset/reload from background threads;
            # live preview always runs off the UI thread (sounddevice worker).
            arr = upstream(
                midis,
                duration=block_dur,
                sample_rate=sr,
                num_channels=CHANNELS,
                buffer_size=bs,
                reset=False,
            )
        except Exception:
            if not self._logged_upstream_synth_err:
                traceback.print_exc()
                self._logged_upstream_synth_err = True
            arr = np.zeros((frames, CHANNELS), dtype=np.float32)
        return self._shape_outputs_to_stream(arr, frames)

    def _pass_through_effects_chain(
        self, idx: int, frames: int, sr: float, inp: np.ndarray
    ) -> np.ndarray:
        plugin = self._plugin
        bs = self._buffer_size_arg(frames)
        inp = np.ascontiguousarray(np.nan_to_num(inp, copy=False), dtype=np.float32)
        # Never pass reset=True from this thread: live and offline preview run off the UI
        # thread, and many AUs report "must be reloaded on the main thread" if reset runs here.
        reset_effects = False

        # When an instrument feeds the chain, step through each ExternalPlugin so latency /
        # PDC behaviour matches offline renders and avoids rare Chain edge-cases with
        # streaming buffers. FX-only preview still uses a single Pedalboard() call.
        if self._upstream_instrument is not None and isinstance(plugin, Pedalboard):
            out = self._shape_outputs_to_stream(inp, frames)
            try:
                for eff in plugin:
                    out = eff(
                        out,
                        sr,
                        buffer_size=bs,
                        reset=reset_effects,
                    )
                    out = self._shape_outputs_to_stream(out, frames)
            except Exception:
                if not self._logged_chain_err:
                    traceback.print_exc()
                    self._logged_chain_err = True
                out = np.zeros((frames, CHANNELS), dtype=np.float32)
            return out

        try:
            out = plugin(
                inp,
                sr,
                buffer_size=bs,
                reset=reset_effects,
            )
        except Exception:
            if not self._logged_chain_err:
                traceback.print_exc()
                self._logged_chain_err = True
            out = np.zeros((frames, CHANNELS), dtype=np.float32)
        return self._shape_outputs_to_stream(out, frames)

    def _audio_loop(self) -> None:
        import sounddevice as sd  # noqa: WPS433

        plugin = self._plugin
        sr = SAMPLE_RATE
        block_dur = BLOCK_DUR_SEC
        frames = max(128, int(round(sr * block_dur)))
        block_dur = frames / sr
        rng = np.random.default_rng()

        stream = None
        idx = 0

        try:
            stream = sd.OutputStream(
                samplerate=sr,
                channels=CHANNELS,
                dtype="float32",
                blocksize=frames,
                latency="high",
            )
            with self._stream_gate:
                self._playback_stream = stream
            stream.start()

            while not self._stop.is_set():
                buf: np.ndarray

                upstream = self._upstream_instrument

                if upstream is not None and (
                    getattr(plugin, "is_effect", False) or isinstance(plugin, Pedalboard)
                ):
                    midi_dry = self._synth_upstream_block(idx, frames, block_dur, sr)
                    wet = self._pass_through_effects_chain(idx, frames, sr, midi_dry)
                    buf = wet + (_UPSTREAM_DRY_BLEND * midi_dry)
                    buf = self._shape_outputs_to_stream(buf, frames)
                    buf = np.clip(buf, -1.0, 1.0)
                elif getattr(plugin, "is_instrument", False) and upstream is None:
                    midis = self._preview_midi_for_block(idx, block_dur)
                    try:
                        arr = plugin(
                            midis,
                            duration=block_dur,
                            sample_rate=sr,
                            num_channels=CHANNELS,
                            buffer_size=self._buffer_size_arg(frames),
                            reset=False,
                        )
                    except Exception:
                        arr = np.zeros((frames, CHANNELS), dtype=np.float32)
                    buf = self._shape_outputs_to_stream(arr, frames)
                elif getattr(plugin, "is_effect", False) or isinstance(plugin, Pedalboard):
                    inp = self._next_fx_only_dry_block(rng, frames)
                    buf = self._pass_through_effects_chain(idx, frames, sr, inp)
                else:
                    time.sleep(block_dur)
                    idx += 1
                    continue

                try:
                    stream.write(np.ascontiguousarray(buf))
                except Exception:
                    pass

                idx += 1
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
                with self._stream_gate:
                    self._playback_stream = None

    def render_offline_clip(self, duration_sec: float) -> np.ndarray:
        """
        Synthesize the same audio path as the live monitor, without sounddevice (for file playback).
        """
        plugin = self._plugin
        sr = SAMPLE_RATE
        block_dur = BLOCK_DUR_SEC
        frames = max(128, int(round(sr * block_dur)))
        block_dur = frames / sr
        rng = np.random.default_rng()
        total_frames = max(frames, int(round(duration_sec * sr)))
        chunks: list[np.ndarray] = []
        idx = 0
        max_blocks = 50000

        while sum(c.shape[0] for c in chunks) < total_frames and idx < max_blocks:
            upstream = self._upstream_instrument

            if upstream is not None and (
                getattr(plugin, "is_effect", False) or isinstance(plugin, Pedalboard)
            ):
                midi_dry = self._synth_upstream_block(idx, frames, block_dur, sr)
                wet = self._pass_through_effects_chain(idx, frames, sr, midi_dry)
                buf = wet + (_UPSTREAM_DRY_BLEND * midi_dry)
                buf = self._shape_outputs_to_stream(buf, frames)
                buf = np.clip(buf, -1.0, 1.0)
            elif getattr(plugin, "is_instrument", False) and upstream is None:
                midis = self._preview_midi_for_block(idx, block_dur)
                try:
                    arr = plugin(
                        midis,
                        duration=block_dur,
                        sample_rate=sr,
                        num_channels=CHANNELS,
                        buffer_size=self._buffer_size_arg(frames),
                        reset=False,
                    )
                except Exception:
                    arr = np.zeros((frames, CHANNELS), dtype=np.float32)
                buf = self._shape_outputs_to_stream(arr, frames)
            elif getattr(plugin, "is_effect", False) or isinstance(plugin, Pedalboard):
                inp = self._next_fx_only_dry_block(rng, frames)
                buf = self._pass_through_effects_chain(idx, frames, sr, inp)
            else:
                buf = np.zeros((frames, CHANNELS), dtype=np.float32)

            chunks.append(np.ascontiguousarray(buf, dtype=np.float32))
            idx += 1

        if not chunks:
            return np.zeros((total_frames, CHANNELS), dtype=np.float32)
        out = np.vstack(chunks)
        if out.shape[0] > total_frames:
            out = out[:total_frames]
        return out


def render_preview_clip(
    plugin: Any,
    *,
    duration_sec: float = 4.0,
    midi_style_ref: list[str] | None = None,
    upstream_instrument: Any | None = None,
    fx_preview_source_ref: list[str] | None = None,
) -> np.ndarray:
    """Offline clip with the same routing as :class:`PatchcraftrLiveMonitor`."""
    mon = PatchcraftrLiveMonitor(
        plugin,
        midi_style_ref=midi_style_ref,
        upstream_instrument=upstream_instrument,
        fx_preview_source_ref=fx_preview_source_ref,
    )
    return mon.render_offline_clip(duration_sec)
