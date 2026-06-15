"""Low-latency-tolerant live audio preview while a DawDreamer plugin editor is open."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import traceback
from typing import Any, Sequence

import numpy as np
from scipy import signal

from audio_host import (
    DawDreamerGraphSession,
    DawDreamerPreviewSession,
    plugin_is_effect,
    plugin_is_instrument,
    samples_channels_to_daw_audio,
    SAMPLE_RATE,
)
from preset_authoring import preview_midi_to_note_tuples, generate_preview_midi

CHANNELS = 2
BLOCK_DUR_SEC = 0.1
FX_PREVIEW_NOISE_LEVEL = 0.045
_UPSTREAM_DRY_BLEND = 0.14

FX_PREVIEW_GAUSSIAN = "fx_gaussian_noise"
FX_PREVIEW_SCALE_WAV = "fx_scale_wav"
FX_PREVIEW_SOURCES: list[tuple[str, str]] = [
    (FX_PREVIEW_GAUSSIAN, "Gaussian white noise"),
    (FX_PREVIEW_SCALE_WAV, "CDEFGABC scale (WAV)"),
]
DEFAULT_FX_PREVIEW_SOURCE = FX_PREVIEW_GAUSSIAN

PREVIEW_NOTE_VELOCITY = 96
PREVIEW_RETRIGGER_SEC = 2.0

MIDI_PREVIEW_STYLES: list[tuple[str, str]] = [
    ("sustained_c4", "Sustained middle C"),
    ("octave_cycle", "Single C · 4 octaves"),
    ("c_major_chord", "C major chord"),
    ("c_scale_up", "C major scale ↑"),
]
DEFAULT_MIDI_PREVIEW_STYLE = MIDI_PREVIEW_STYLES[0][0]

_LOG = logging.getLogger("dronmakr.patchcraftr")


def midi_preview_style_ids() -> tuple[str, ...]:
    return tuple(s for s, _ in MIDI_PREVIEW_STYLES)


def fx_preview_source_ids() -> tuple[str, ...]:
    return tuple(s for s, _ in FX_PREVIEW_SOURCES)


def default_sounddevice_output_index(sd_module: Any) -> int | None:
    def_dev = getattr(sd_module.default, "device", None)
    if def_dev is None:
        return None
    if isinstance(def_dev, int):
        return def_dev if def_dev >= 0 else None
    try:
        if hasattr(def_dev, "__len__") and len(def_dev) > 1:
            out_idx = int(def_dev[1])
            return out_idx if out_idx >= 0 else None
    except (TypeError, ValueError, IndexError):
        return None
    return None


def _resolve_fx_preview_wav_path() -> str:
    from preset_authoring import preview_sample_wav_path

    return preview_sample_wav_path()


def _load_wav_loop_mono_stereo(path: str) -> tuple[np.ndarray | None, bool]:
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
    Background audio thread driving the same DawDreamer processors as ``open_editor``.

    ``graph_session`` holds instrument + FX on one RenderEngine. ``preview_mode``:
    ``instrument_fx`` | ``instrument_only`` | ``fx_chain``.
    """

    def __init__(
        self,
        graph_session: DawDreamerGraphSession,
        *,
        preview_mode: str = "instrument_fx",
        midi_style_ref: list[str] | None = None,
        fx_preview_source_ref: list[str] | None = None,
    ) -> None:
        self._graph = graph_session
        self._preview_mode = preview_mode
        self._midi_style_ref = midi_style_ref if midi_style_ref is not None else [DEFAULT_MIDI_PREVIEW_STYLE]
        self._fx_source_ref = (
            fx_preview_source_ref if fx_preview_source_ref is not None else [DEFAULT_FX_PREVIEW_SOURCE]
        )
        self._preview = DawDreamerPreviewSession(graph_session)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._playback_stream: Any | None = None
        self._stream_gate = threading.Lock()
        self._last_preview_notes: tuple[int, ...] = ()
        self._wav_loop: np.ndarray | None = None
        self._wav_pos = 0
        self._logged_write_err = False
        self._logged_render_err = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        with self._stream_gate:
            self._playback_stream = None

    @staticmethod
    def _shape_outputs_to_stream(arr: np.ndarray, frames: int) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.column_stack((arr, arr))
        elif arr.ndim == 2 and arr.shape[0] != frames and arr.shape[1] == frames:
            pass
        elif arr.ndim == 2 and arr.shape[0] != frames:
            arr = arr.T if arr.shape[1] == frames else arr
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

    def _preview_midi_notes(self, block_index: int, block_dur: float) -> list[tuple[int, int, float, float]]:
        style = self._midi_style_ref[0] if self._midi_style_ref else DEFAULT_MIDI_PREVIEW_STYLE
        if block_index % max(1, int(PREVIEW_RETRIGGER_SEC / block_dur)) != 0 and self._last_preview_notes:
            notes = list(self._last_preview_notes)
            return [(n, PREVIEW_NOTE_VELOCITY, 0.0, block_dur) for n in notes]

        if style == "sustained_c4":
            targets = [60]
        elif style == "octave_cycle":
            octaves = [36, 48, 60, 72]
            targets = [octaves[block_index % len(octaves)]]
        elif style == "c_major_chord":
            targets = [60, 64, 67]
        else:
            scale = [60, 62, 64, 65, 67, 69, 71, 72]
            targets = [scale[block_index % len(scale)]]

        self._last_preview_notes = tuple(targets)
        return [(n, PREVIEW_NOTE_VELOCITY, 0.0, block_dur) for n in targets]

    def _next_fx_only_dry_block(self, rng: np.random.Generator, frames: int) -> np.ndarray:
        src = self._fx_source_ref[0] if self._fx_source_ref else DEFAULT_FX_PREVIEW_SOURCE
        if src == FX_PREVIEW_SCALE_WAV:
            if self._wav_loop is None:
                path = _resolve_fx_preview_wav_path()
                self._wav_loop, ok = _load_wav_loop_mono_stereo(path)
                if not ok or self._wav_loop is None:
                    src = FX_PREVIEW_GAUSSIAN
            if src == FX_PREVIEW_SCALE_WAV and self._wav_loop is not None:
                n = self._wav_loop.shape[0]
                idx = np.arange(self._wav_pos, self._wav_pos + frames) % n
                self._wav_pos = (self._wav_pos + frames) % n
                return self._wav_loop[idx]
        noise = rng.standard_normal((frames, CHANNELS)).astype(np.float32) * FX_PREVIEW_NOISE_LEVEL
        return noise

    def _render_block(
        self, block_index: int, frames: int, block_dur: float, rng: np.random.Generator
    ) -> np.ndarray:
        mode = self._preview_mode
        try:
            if mode == "instrument_fx" and self._graph.instrument is not None:
                notes = self._preview_midi_notes(block_index, block_dur)
                dry = self._preview.render_block(block_dur, midi_notes=notes)
                dry = self._shape_outputs_to_stream(dry, frames)
                wet = dry
                return np.clip(wet + _UPSTREAM_DRY_BLEND * dry, -1.0, 1.0)
            if mode == "instrument_only" and self._graph.instrument is not None:
                notes = self._preview_midi_notes(block_index, block_dur)
                out = self._preview.render_block(block_dur, midi_notes=notes)
                return self._shape_outputs_to_stream(out, frames)
            if mode == "fx_chain":
                dry = self._next_fx_only_dry_block(rng, frames)
                out = self._preview.render_block(block_dur, dry_audio=dry)
                return self._shape_outputs_to_stream(out, frames)
        except Exception:
            if not self._logged_render_err:
                _LOG.exception("Patchcraftr: DawDreamer preview render failed once")
                traceback.print_exc()
                self._logged_render_err = True
        return np.zeros((frames, CHANNELS), dtype=np.float32)

    def _audio_loop(self) -> None:
        try:
            import sounddevice as sd  # noqa: WPS433
        except ImportError:
            _LOG.warning("sounddevice not installed — live preview disabled")
            return

        sr = SAMPLE_RATE
        frames = max(128, int(round(sr * BLOCK_DUR_SEC)))
        block_dur = frames / sr
        rng = np.random.default_rng()

        out_idx = default_sounddevice_output_index(sd)
        out_kwargs: dict[str, Any] = {
            "samplerate": sr,
            "channels": CHANNELS,
            "blocksize": frames,
            "dtype": "float32",
        }
        if out_idx is not None:
            out_kwargs["device"] = out_idx

        stream = None
        try:
            stream = sd.OutputStream(**out_kwargs)
            with self._stream_gate:
                self._playback_stream = stream
            stream.start()
        except Exception:
            _LOG.exception("Patchcraftr could not start sounddevice playback")
            return

        idx = 0
        try:
            while not self._stop.is_set():
                buf = self._render_block(idx, frames, block_dur, rng)
                try:
                    stream.write(np.ascontiguousarray(buf))
                except Exception as ex:
                    if not self._logged_write_err:
                        _LOG.exception("Patchcraftr stream.write failed: %s", ex)
                        self._logged_write_err = True
                idx += 1
        finally:
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

    def render_offline_clip(self, duration_sec: float) -> np.ndarray:
        sr = SAMPLE_RATE
        frames = max(128, int(round(sr * BLOCK_DUR_SEC)))
        block_dur = frames / sr
        rng = np.random.default_rng()
        total_frames = max(frames, int(round(duration_sec * sr)))
        chunks: list[np.ndarray] = []
        idx = 0
        while sum(c.shape[0] for c in chunks) < total_frames and idx < 50000:
            chunks.append(self._render_block(idx, frames, block_dur, rng))
            idx += 1
        if not chunks:
            return np.zeros((total_frames, CHANNELS), dtype=np.float32)
        out = np.vstack(chunks)
        return out[:total_frames]


def render_preview_clip(
    graph_session: DawDreamerGraphSession,
    *,
    duration_sec: float = 4.0,
    preview_mode: str = "instrument_fx",
    midi_style_ref: list[str] | None = None,
    fx_preview_source_ref: list[str] | None = None,
) -> np.ndarray:
    mon = PatchcraftrLiveMonitor(
        graph_session,
        preview_mode=preview_mode,
        midi_style_ref=midi_style_ref,
        fx_preview_source_ref=fx_preview_source_ref,
    )
    return mon.render_offline_clip(duration_sec)
