"""
DawDreamer VST/AU hosting — offline render and preset state.

Import this module before numba-backed ``dsp`` helpers (LLVM init order).
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

import dawdreamer as daw
import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100
BUFFER_SIZE = 512
HEADROOM_GAIN = 0.5  # -6 dB

_LOG = logging.getLogger("dronmakr.audio_host")

# Harmless JUCE stderr noise when DawDreamer loads macOS VST3 bundles (see DBraun/DawDreamer#218).
_KNOWN_DAWDREAMER_STDERR_NOISE = (
    "attempt to map invalid URI",
)


@contextlib.contextmanager
def _filter_dawdreamer_stderr():
    """Drop known-harmless JUCE messages while loading plug-ins on macOS."""
    if sys.platform != "darwin":
        yield
        return

    read_fd, write_fd = os.pipe()
    saved_stderr = os.dup(2)
    reader_done = threading.Event()

    def _reader() -> None:
        buf = b""
        try:
            while True:
                chunk = os.read(read_fd, 8192)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace")
                    if any(noise in text for noise in _KNOWN_DAWDREAMER_STDERR_NOISE):
                        _LOG.debug("DawDreamer stderr (suppressed): %s", text)
                    else:
                        os.write(saved_stderr, line + b"\n")
            if buf:
                text = buf.decode("utf-8", errors="replace")
                if any(noise in text for noise in _KNOWN_DAWDREAMER_STDERR_NOISE):
                    _LOG.debug("DawDreamer stderr (suppressed): %s", text)
                else:
                    os.write(saved_stderr, buf)
        except OSError:
            pass
        finally:
            reader_done.set()

    try:
        os.dup2(write_fd, 2)
        os.close(write_fd)
        threading.Thread(target=_reader, daemon=True).start()
        yield
    finally:
        os.dup2(saved_stderr, 2)
        os.close(saved_stderr)
        try:
            os.close(read_fd)
        except OSError:
            pass
        reader_done.wait(timeout=2.0)


def create_engine(sample_rate: int = SAMPLE_RATE, buffer_size: int = BUFFER_SIZE) -> Any:
    return daw.RenderEngine(sample_rate, buffer_size)


def _unique_processor_name(prefix: str = "proc") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def load_plugin(engine: Any, plugin_path: str, *, name: str | None = None) -> Any:
    """Load a VST3/AU plug-in processor on ``engine``."""
    proc_name = name or _unique_processor_name("plugin")
    with _filter_dawdreamer_stderr():
        return engine.make_plugin_processor(proc_name, os.path.abspath(plugin_path))


def plugin_is_instrument(processor: Any) -> bool:
    try:
        return int(processor.get_num_input_channels()) == 0
    except Exception:
        return False


def plugin_is_effect(processor: Any) -> bool:
    return not plugin_is_instrument(processor)


def save_plugin_state(processor: Any, preset_path: str) -> None:
    os.makedirs(os.path.dirname(preset_path) or ".", exist_ok=True)
    processor.save_state(os.path.abspath(preset_path))


def apply_plugin_state(processor: Any, preset_path: str) -> None:
    if preset_path and os.path.isfile(preset_path):
        processor.load_state(os.path.abspath(preset_path))


def serialize_plugin_preset_bytes(processor: Any) -> bytes:
    fd, path = tempfile.mkstemp(suffix=".ddstate")
    os.close(fd)
    try:
        save_plugin_state(processor, path)
        with open(path, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def apply_vstpreset_bytes_to_plugin(processor: Any, preset_bytes: bytes) -> None:
    fd, path = tempfile.mkstemp(suffix=".ddstate")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(preset_bytes)
        apply_plugin_state(processor, path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def write_plugin_state_to_vstpreset(preset_path: str, processor: Any) -> None:
    save_plugin_state(processor, preset_path)


def open_plugin_editor(processor: Any) -> None:
    processor.open_editor()


def daw_audio_to_samples_channels(audio: np.ndarray) -> np.ndarray:
    """DawDreamer ``(channels, samples)`` → ``(samples, channels)`` float32."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=0)
    if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    if arr.ndim == 1:
        arr = np.column_stack([arr, arr])
    elif arr.shape[1] == 1:
        arr = np.column_stack([arr[:, 0], arr[:, 0]])
    return np.ascontiguousarray(arr, dtype=np.float32)


def samples_channels_to_daw_audio(audio: np.ndarray) -> np.ndarray:
    """``(samples, channels)`` or ``(channels, samples)`` → DawDreamer ``(channels, samples)``."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return np.stack([arr, arr], axis=0)
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    if arr.shape[0] == 1:
        arr = np.vstack([arr, arr])
    return np.ascontiguousarray(arr, dtype=np.float32)


def _build_plugin_graph(
    instrument: Any | None,
    fx_processors: Sequence[Any],
    *,
    playback: Any | None = None,
) -> list[tuple[Any, list[str]]]:
    graph: list[tuple[Any, list[str]]] = []
    tail_name: str | None = None

    if instrument is not None:
        graph.append((instrument, []))
        tail_name = instrument.get_name()
    elif playback is not None:
        graph.append((playback, []))
        tail_name = playback.get_name()

    for fx in fx_processors:
        inputs = [tail_name] if tail_name else []
        graph.append((fx, inputs))
        tail_name = fx.get_name()

    return graph


def _midi_duration_seconds(midi_path: str) -> float:
    import mido  # noqa: PLC0415

    mid = mido.MidiFile(midi_path)
    return float(mid.length)


DRONE_MIDI_TEMPO_BPM = 120
# Small post-roll so the final note-off and FX tails are not clipped at the block boundary.
RENDER_TAIL_SEC = 0.25
# Some VST/AU hosts drop the first offline block unless primed after ``load_state``.
PLUGIN_RENDER_WARMUP_SEC = 0.2
MIN_RENDER_PEAK = 1e-5


def _inject_midi_notes_from_file(instrument: Any, midi_path: str) -> int:
    """Program note events with ``add_midi_note`` instead of ``load_midi``.

    ``load_midi(all_events=True)`` can pull in meta/CC traffic that some hosts mishandle,
    and note timing is clearer when we convert from PrettyMIDI ourselves.
    """
    import pretty_midi  # noqa: PLC0415

    instrument.clear_midi()
    pm = pretty_midi.PrettyMIDI(os.path.abspath(midi_path))
    note_count = 0
    for inst in pm.instruments:
        for note in inst.notes:
            if int(note.velocity) <= 0:
                continue
            start = float(note.start)
            duration = max(1e-4, float(note.end - note.start))
            instrument.add_midi_note(
                int(note.pitch),
                int(note.velocity),
                start,
                duration,
                beats=False,
            )
            note_count += 1
    return note_count


def _prepare_instrument_for_render(
    engine: Any,
    instrument: Any,
    midi_path: str,
) -> int:
    """Load note events after the graph is wired; returns injected note count."""
    engine.set_bpm(float(DRONE_MIDI_TEMPO_BPM))
    return _inject_midi_notes_from_file(instrument, midi_path)


def _render_output_is_usable(audio: np.ndarray) -> bool:
    if audio.size == 0:
        return False
    if not np.isfinite(audio).all():
        return False
    return float(np.max(np.abs(audio))) >= MIN_RENDER_PEAK


def _trim_rendered_audio(
    audio: np.ndarray,
    *,
    duration_sec: float,
    sample_rate: int,
    headroom_gain: float,
) -> np.ndarray:
    out = daw_audio_to_samples_channels(audio)
    target_samples = int(round(float(duration_sec) * sample_rate))
    if out.shape[0] > target_samples:
        out = out[:target_samples]
    if headroom_gain != 1.0:
        out = out * float(headroom_gain)
    return out


def _render_loaded_midi_graph(
    engine: Any,
    graph: list[tuple[Any, list[str]]],
    instrument: Any,
    midi_path: str,
    duration_sec: float,
    *,
    sample_rate: int,
    headroom_gain: float,
) -> np.ndarray:
    engine.load_graph(graph)
    note_count = _prepare_instrument_for_render(engine, instrument, midi_path)
    if note_count <= 0:
        raise ValueError(f"No playable notes found in MIDI file: {midi_path}")
    if PLUGIN_RENDER_WARMUP_SEC > 0:
        time.sleep(PLUGIN_RENDER_WARMUP_SEC)
    engine.render(float(duration_sec) + RENDER_TAIL_SEC)
    return _trim_rendered_audio(
        engine.get_audio(),
        duration_sec=duration_sec,
        sample_rate=sample_rate,
        headroom_gain=headroom_gain,
    )


def render_midi_graph(
    instrument: Any,
    fx_processors: Sequence[Any],
    midi_path: str,
    *,
    duration_sec: float | None = None,
    sample_rate: int = SAMPLE_RATE,
    buffer_size: int = BUFFER_SIZE,
    headroom_gain: float = HEADROOM_GAIN,
    engine: Any | None = None,
) -> np.ndarray:
    """Offline instrument (+ optional FX) render from a MIDI file. Returns (samples, channels)."""
    if isinstance(instrument, str):
        raise TypeError("render_midi_graph expects a loaded PluginProcessor as instrument")
    eng = engine or create_engine(sample_rate, buffer_size)
    inst = instrument
    fx_loaded: list[Any] = list(fx_processors)
    graph = _build_plugin_graph(inst, fx_loaded)
    if not graph:
        raise ValueError("render_midi_graph requires an instrument processor")

    dur = duration_sec if duration_sec is not None else _midi_duration_seconds(midi_path)
    out = _render_loaded_midi_graph(
        eng,
        graph,
        inst,
        midi_path,
        float(dur),
        sample_rate=sample_rate,
        headroom_gain=headroom_gain,
    )
    return out


def render_midi_chain_from_paths(
    instrument_path: str,
    instrument_state_path: str | None,
    fx_specs: Sequence[tuple[str, str | None]],
    midi_path: str,
    *,
    duration_sec: float | None = None,
    sample_rate: int = SAMPLE_RATE,
    buffer_size: int = BUFFER_SIZE,
    headroom_gain: float = HEADROOM_GAIN,
) -> np.ndarray:
    """Load plug-ins by path, apply saved states, render MIDI → (samples, channels)."""
    dur = duration_sec if duration_sec is not None else _midi_duration_seconds(midi_path)

    last_out: np.ndarray | None = None
    for attempt in range(2):
        engine = create_engine(sample_rate, buffer_size)
        inst = load_plugin(engine, instrument_path, name="instrument")
        if instrument_state_path:
            apply_plugin_state(inst, instrument_state_path)

        fx_procs = []
        for idx, (fx_path, fx_state) in enumerate(fx_specs):
            fx = load_plugin(engine, fx_path, name=f"fx_{idx}")
            if fx_state:
                apply_plugin_state(fx, fx_state)
            fx_procs.append(fx)

        graph = _build_plugin_graph(inst, fx_procs)
        last_out = _render_loaded_midi_graph(
            engine,
            graph,
            inst,
            midi_path,
            float(dur),
            sample_rate=sample_rate,
            headroom_gain=headroom_gain,
        )
        if _render_output_is_usable(last_out):
            return last_out
        if attempt == 0:
            _LOG.warning(
                "DawDreamer render looked empty or invalid; retrying once after plugin warmup"
            )
            time.sleep(PLUGIN_RENDER_WARMUP_SEC)

    if last_out is None:
        raise RuntimeError("DawDreamer render produced no audio")
    return last_out


def render_audio_through_fx_chain(
    input_audio: np.ndarray,
    fx_processors: Sequence[Any],
    *,
    sample_rate: int = SAMPLE_RATE,
    buffer_size: int = BUFFER_SIZE,
) -> np.ndarray:
    """Process ``(channels, samples)`` or ``(samples, channels)`` through FX processors."""
    daw_in = samples_channels_to_daw_audio(input_audio)
    duration_sec = daw_in.shape[1] / float(sample_rate)
    engine = create_engine(sample_rate, buffer_size)
    playback = engine.make_playback_processor("playback", daw_in)
    graph = _build_plugin_graph(None, fx_processors, playback=playback)
    engine.load_graph(graph)
    engine.render(duration_sec)
    return daw_audio_to_samples_channels(engine.get_audio())


def render_wav_through_fx_paths(
    wav_path: str,
    fx_specs: Sequence[tuple[str, str | None]],
    *,
    sample_rate: int = SAMPLE_RATE,
    buffer_size: int = BUFFER_SIZE,
) -> tuple[np.ndarray, int]:
    """Load WAV, run through FX chain loaded from paths. Returns (samples, channels), sr."""
    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    if sr != sample_rate:
        raise ValueError(f"Expected sample rate {sample_rate}, got {sr}")
    daw_in = data.T
    duration_sec = daw_in.shape[1] / float(sample_rate)
    engine = create_engine(sample_rate, buffer_size)
    playback = engine.make_playback_processor("playback", daw_in)

    fx_procs: list[Any] = []
    for idx, (fx_path, fx_state) in enumerate(fx_specs):
        fx = load_plugin(engine, fx_path, name=f"fx_{idx}")
        if fx_state:
            apply_plugin_state(fx, fx_state)
        fx_procs.append(fx)

    graph = _build_plugin_graph(None, fx_procs, playback=playback)
    engine.load_graph(graph)
    engine.render(duration_sec)
    return daw_audio_to_samples_channels(engine.get_audio()), sample_rate


def reload_plugin_preserving_state(
    engine: Any,
    plugin_path: str,
    old_processor: Any,
    *,
    name: str | None = None,
) -> Any:
    blob_path = None
    fd, blob_path = tempfile.mkstemp(suffix=".ddstate")
    os.close(fd)
    try:
        save_plugin_state(old_processor, blob_path)
        new_proc = load_plugin(engine, plugin_path, name=name or old_processor.get_name())
        apply_plugin_state(new_proc, blob_path)
        return new_proc
    finally:
        if blob_path:
            try:
                os.unlink(blob_path)
            except OSError:
                pass


@dataclass
class DawDreamerGraphSession:
    """Shared RenderEngine holding instrument + FX processors for Patchcraftr."""

    sample_rate: int = SAMPLE_RATE
    buffer_size: int = BUFFER_SIZE
    engine: Any = field(init=False)
    instrument: Any | None = field(default=None, init=False)
    instrument_path: str = field(default="", init=False)
    fx_processors: list[Any] = field(default_factory=list, init=False)
    fx_paths: list[str] = field(default_factory=list, init=False)
    _engine_lock: threading.RLock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.engine = create_engine(self.sample_rate, self.buffer_size)
        self.fx_processors = []
        self.fx_paths = []
        self._engine_lock = threading.RLock()

    @property
    def engine_lock(self) -> threading.RLock:
        return self._engine_lock

    def clear(self) -> None:
        with self._engine_lock:
            self.instrument = None
            self.instrument_path = ""
            self.fx_processors = []
            self.fx_paths = []
            self.engine = create_engine(self.sample_rate, self.buffer_size)

    def set_instrument(self, plugin_path: str, state_path: str | None = None) -> Any:
        with self._engine_lock:
            self.instrument_path = plugin_path
            self.instrument = load_plugin(self.engine, plugin_path, name="instrument")
            if state_path:
                apply_plugin_state(self.instrument, state_path)
            return self.instrument

    def set_fx_chain(
        self,
        specs: Sequence[tuple[str, str | None]],
    ) -> list[Any]:
        with self._engine_lock:
            self.fx_processors = []
            self.fx_paths = []
            for idx, (path, state_path) in enumerate(specs):
                fx = load_plugin(self.engine, path, name=f"fx_{idx}")
                if state_path:
                    apply_plugin_state(fx, state_path)
                self.fx_processors.append(fx)
                self.fx_paths.append(path)
            return self.fx_processors

    def rebuild_graph(self, playback: Any | None = None) -> None:
        with self._engine_lock:
            graph = _build_plugin_graph(self.instrument, self.fx_processors, playback=playback)
            if graph:
                self.engine.load_graph(graph)
