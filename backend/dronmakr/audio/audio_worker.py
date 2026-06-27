"""
Run DawDreamer / VST work in a dedicated child process.

Desktop tray mode serves Flask from a background thread (see ``DRONMAKR_ASYNC_MODE=threading``).
Many AU/VST hosts require plug-in load and lifecycle on the process main thread; this runner
re-invokes the same app binary with ``--audio-worker`` so work runs on a real main thread.
"""

from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

_WORKER_FLAG = "--audio-worker"
_WORKER_CHILD_ENV = "DRONMAKR_AUDIO_WORKER"
_ASYNC_THREADING_ENV = "DRONMAKR_ASYNC_MODE"
_WORKER_TIMEOUT_S = int(os.environ.get("DRONMAKR_AUDIO_WORKER_TIMEOUT", "3600"))


def _should_delegate_to_worker() -> bool:
    if os.environ.get(_WORKER_CHILD_ENV):
        return False
    if os.environ.get(_ASYNC_THREADING_ENV, "").lower() != "threading":
        return False
    # Desktop/web UI serves Flask from a worker thread; VST/AU load must run on main thread.
    # Standalone CLI invocations (even with threading env set) already run on main thread.
    return threading.current_thread() is not threading.main_thread()


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _worker_argv() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, _WORKER_FLAG]
    backend_entry = _backend_root() / "backend_server.py"
    if not backend_entry.is_file():
        raise RuntimeError(
            "Cannot spawn audio worker: backend_server.py is missing. "
            "Run from the dronmakr repo/install layout."
        )
    return [sys.executable, str(backend_entry.resolve()), _WORKER_FLAG]


_ORIGINAL_BUILTIN_PRINT = builtins.print


def _worker_print_routed_to_stderr(*args, **kwargs):
    if kwargs.get("file") is None:
        kwargs["file"] = sys.stderr
    return _ORIGINAL_BUILTIN_PRINT(*args, **kwargs)


def _patch_print_to_stderr() -> None:
    builtins.print = _worker_print_routed_to_stderr


def invoke_audio_worker(task: str, params: dict) -> dict:
    cmd = _worker_argv()
    env = os.environ.copy()
    env.pop(_ASYNC_THREADING_ENV, None)
    env[_WORKER_CHILD_ENV] = "1"
    payload = json.dumps({"task": task, "params": params}, ensure_ascii=False).encode("utf-8")
    proc = subprocess.run(
        cmd,
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(_backend_root()),
        timeout=_WORKER_TIMEOUT_S,
        check=False,
    )
    err_txt = proc.stderr.decode("utf-8", errors="replace").strip()
    out_txt = proc.stdout.decode("utf-8", errors="replace").strip()
    last_json = None
    if out_txt:
        try:
            last_json = json.loads(out_txt.splitlines()[-1])
        except json.JSONDecodeError:
            last_json = None
    if proc.returncode != 0:
        if isinstance(last_json, dict) and last_json.get("ok"):
            _ORIGINAL_BUILTIN_PRINT(
                f"[audio_worker] child exited {proc.returncode} after success; "
                "treating JSON result as OK (DawDreamer teardown abort)",
                file=sys.stderr,
            )
            return last_json
        if isinstance(last_json, dict) and last_json.get("error"):
            raise RuntimeError(last_json["error"])
        msg = err_txt or out_txt or f"exit {proc.returncode}"
        raise RuntimeError(f"Audio worker failed ({proc.returncode}): {msg}")
    if not out_txt or last_json is None:
        raise RuntimeError("Audio worker produced no valid JSON stdout")
    data = last_json
    if not data.get("ok"):
        raise RuntimeError(data.get("error") or str(data))
    return data


def delegate_generate_drone_sample_if_needed(
    input_path: str,
    output_path: str,
    presets_path: str,
    instrument: str | None,
    effect: str | None,
    render_duration_sec: float | None = None,
    tempo_bpm: float | None = None,
    instrument_selection: dict | None = None,
    fx_slots: list | None = None,
) -> str | None:
    if not _should_delegate_to_worker():
        return None
    data = invoke_audio_worker(
        "generate_drone_sample",
        {
            "input_path": input_path,
            "output_path": output_path,
            "presets_path": presets_path,
            "instrument": instrument,
            "effect": effect,
            "render_duration_sec": render_duration_sec,
            "tempo_bpm": tempo_bpm,
            "instrument_selection": instrument_selection,
            "fx_slots": fx_slots,
        },
    )
    return str(data["output_path"])


def delegate_open_drone_plugin_editor_if_needed(
    plugin_path: str,
    role: str,
    preset_path: str | None = None,
    *,
    editor_preview: dict | None = None,
) -> dict | None:
    if not _should_delegate_to_worker():
        return None
    data = invoke_audio_worker(
        "open_drone_plugin_editor",
        {
            "plugin_path": plugin_path,
            "role": role,
            "preset_path": preset_path,
            "editor_preview": editor_preview,
        },
    )
    return dict(data.get("result") or {})


def delegate_scan_plugin_classifications_if_needed(*, force: bool = False) -> None:
    if not _should_delegate_to_worker():
        from dronmakr.presets.preset_authoring import scan_plugin_classifications

        scan_plugin_classifications(force=force)
        return
    invoke_audio_worker("scan_plugin_classifications", {"force": force})


def spawn_background_plugin_scan(*, force: bool = False) -> None:
    """Start plug-in verification scan without blocking the HTTP/UI thread."""
    import subprocess

    from dronmakr.presets.preset_authoring import read_plugin_scan_progress, write_plugin_scan_progress

    if read_plugin_scan_progress().get("status") == "running":
        return

    write_plugin_scan_progress(status="running", done=0, total=0, label="Starting…")
    cmd = _worker_argv()
    env = os.environ.copy()
    env.pop(_ASYNC_THREADING_ENV, None)
    env[_WORKER_CHILD_ENV] = "1"
    payload = json.dumps({"task": "scan_plugin_classifications", "params": {"force": force}}, ensure_ascii=False)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        cwd=str(_backend_root()),
    )
    assert proc.stdin is not None
    proc.stdin.write(payload.encode("utf-8"))
    proc.stdin.close()


def delegate_apply_effect_if_needed(
    input_path: str,
    effect_chain: str,
    presets_path: str,
) -> bool:
    if not _should_delegate_to_worker():
        return False
    invoke_audio_worker(
        "apply_effect",
        {
            "input_path": input_path,
            "effect_chain": effect_chain,
            "presets_path": presets_path,
        },
    )
    return True


def delegate_apply_plugin_patch_if_needed(
    input_path: str,
    patch_name: str,
    presets_path: str,
) -> bool:
    if not _should_delegate_to_worker():
        return False
    invoke_audio_worker(
        "apply_plugin_patch",
        {
            "input_path": input_path,
            "patch_name": patch_name,
            "presets_path": presets_path,
        },
    )
    return True


def run_stdio_worker() -> None:
    import dronmakr.audio.audio_host as audio_host  # noqa: F401, PLC0415 — DawDreamer before numba

    _patch_print_to_stderr()
    try:
        raw = sys.stdin.read()
        job = json.loads(raw)
        task = job["task"]
        params = job["params"]
        if task == "generate_drone_sample":
            from dronmakr.generate.generate_sample import generate_drone_sample

            out = generate_drone_sample(
                input_path=params["input_path"],
                output_path=params["output_path"],
                presets_path=params["presets_path"],
                instrument=params.get("instrument"),
                effect=params.get("effect"),
                render_duration_sec=params.get("render_duration_sec"),
                tempo_bpm=params.get("tempo_bpm"),
                instrument_selection=params.get("instrument_selection"),
                fx_slots=params.get("fx_slots"),
            )
            result = {"ok": True, "output_path": out}
        elif task == "open_drone_plugin_editor":
            from dronmakr.apps.generatr_plugins import open_drone_plugin_editor_capture

            capture = open_drone_plugin_editor_capture(
                params["plugin_path"],
                params.get("role") or "instrument",
                params.get("preset_path"),
                editor_preview=params.get("editor_preview"),
            )
            result = {"ok": True, "result": capture}
        elif task == "scan_plugin_classifications":
            from dronmakr.presets.preset_authoring import scan_plugin_classifications

            scan_plugin_classifications(force=bool(params.get("force")))
            result = {"ok": True}
        elif task == "apply_effect":
            from dronmakr.generate.generate_sample import apply_effect

            apply_effect(
                params["input_path"],
                params["effect_chain"],
                presets_path=params["presets_path"],
            )
            result = {"ok": True}
        elif task == "apply_plugin_patch":
            from dronmakr.audio.process_sample import apply_plugin_patch_to_sample

            apply_plugin_patch_to_sample(
                params["input_path"],
                params["patch_name"],
                presets_path=params["presets_path"],
            )
            result = {"ok": True}
        else:
            result = {"ok": False, "error": f"unknown task {task!r}"}
        sys.stdout.buffer.write(
            (json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8")
        )
        sys.stdout.buffer.flush()
        # DawDreamer/JUCE plug-ins often SIGABRT during normal interpreter teardown on macOS.
        # Exit immediately after reporting JSON so the parent process can treat the run as success.
        os._exit(0 if result.get("ok") else 1)
    except Exception as e:  # noqa: BLE001
        err_payload = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        try:
            sys.stdout.buffer.write(
                (json.dumps(err_payload, ensure_ascii=False) + "\n").encode("utf-8")
            )
            sys.stdout.buffer.flush()
        except OSError:
            pass
        os._exit(1)


# Backward-compatible aliases
invoke_pedalboard_worker = invoke_audio_worker
ensure_pedalboard_midi_utils = lambda: None
