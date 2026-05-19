"""
Run Pedalboard / VST work in a dedicated child process.

Desktop tray mode serves Flask from a background thread (see ``DRONMAKR_ASYNC_MODE=threading``).
Many AU/VST hosts require plug-in load and lifecycle on the process main thread; this runner
re-invokes the same app binary with ``--pedalboard-worker`` so work runs on a real main thread.

Parent HTTP handlers delegate here only when ``DRONMAKR_ASYNC_MODE`` is ``threading`` and the
call is not already inside the worker (``DRONMAKR_PEDALBOARD_WORKER``).
"""

from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
from pathlib import Path

_WORKER_FLAG = "--pedalboard-worker"
_WORKER_CHILD_ENV = "DRONMAKR_PEDALBOARD_WORKER"
_ASYNC_THREADING_ENV = "DRONMAKR_ASYNC_MODE"
_WORKER_TIMEOUT_S = int(os.environ.get("DRONMAKR_PEDALBOARD_WORKER_TIMEOUT", "3600"))


def _should_delegate_to_worker() -> bool:
    if os.environ.get(_WORKER_CHILD_ENV):
        return False
    return os.environ.get(_ASYNC_THREADING_ENV, "").lower() == "threading"


def _package_root() -> Path:
    """Directory containing package modules (cwd for the worker subprocess)."""
    return Path(__file__).resolve().parent


def _worker_argv() -> list[str]:
    """Command line to spawn the worker entrypoint (frozen exe or dev ``desktop_app.py``)."""
    if getattr(sys, "frozen", False):
        return [sys.executable, _WORKER_FLAG]
    try:
        import desktop_app
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Cannot spawn pedalboard worker: desktop_app is not importable. "
            "Run from the dronmakr repo/install layout."
        ) from e
    return [sys.executable, str(Path(desktop_app.__file__).resolve()), _WORKER_FLAG]


_ORIGINAL_BUILTIN_PRINT = builtins.print


def _worker_print_routed_to_stderr(*args, **kwargs):
    """Module-level shim so repr/pickle never looks for a missing ``._print`` on this module."""
    if kwargs.get("file") is None:
        kwargs["file"] = sys.stderr
    return _ORIGINAL_BUILTIN_PRINT(*args, **kwargs)


def _patch_print_to_stderr() -> None:
    """Keep stdout clean for a single JSON status line (logs go to stderr)."""
    builtins.print = _worker_print_routed_to_stderr


def invoke_pedalboard_worker(task: str, params: dict) -> dict:
    """
    Run one task in a child process. Raises RuntimeError on failure.
    Returns decoded JSON object with at least ``{"ok": true, ...}``.
    """
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
        cwd=str(_package_root()),
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
        if isinstance(last_json, dict) and last_json.get("error"):
            raise RuntimeError(last_json["error"])
        msg = err_txt or out_txt or f"exit {proc.returncode}"
        raise RuntimeError(f"Pedalboard worker failed ({proc.returncode}): {msg}")
    if not out_txt or last_json is None:
        raise RuntimeError("Pedalboard worker produced no valid JSON stdout")
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
) -> str | None:
    """If desktop threading mode, run generation in a child process; else return None."""
    if not _should_delegate_to_worker():
        return None
    data = invoke_pedalboard_worker(
        "generate_drone_sample",
        {
            "input_path": input_path,
            "output_path": output_path,
            "presets_path": presets_path,
            "instrument": instrument,
            "effect": effect,
        },
    )
    return str(data["output_path"])


def delegate_apply_effect_if_needed(
    input_path: str,
    effect_chain: str,
    presets_path: str,
) -> bool:
    """If desktop threading mode, run apply_effect in a child process; else return False."""
    if not _should_delegate_to_worker():
        return False
    invoke_pedalboard_worker(
        "apply_effect",
        {
            "input_path": input_path,
            "effect_chain": effect_chain,
            "presets_path": presets_path,
        },
    )
    return True


def run_stdio_worker() -> None:
    """Entry: read one JSON job from stdin, write one JSON line to stdout."""
    _patch_print_to_stderr()
    try:
        raw = sys.stdin.read()
        job = json.loads(raw)
        task = job["task"]
        params = job["params"]
        if task == "generate_drone_sample":
            from generate_sample import generate_drone_sample

            out = generate_drone_sample(
                input_path=params["input_path"],
                output_path=params["output_path"],
                presets_path=params["presets_path"],
                instrument=params.get("instrument"),
                effect=params.get("effect"),
            )
            result = {"ok": True, "output_path": out}
        elif task == "apply_effect":
            from generate_sample import apply_effect

            apply_effect(
                params["input_path"],
                params["effect_chain"],
                presets_path=params["presets_path"],
            )
            result = {"ok": True}
        else:
            result = {"ok": False, "error": f"unknown task {task!r}"}
        sys.stdout.buffer.write(
            (json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8")
        )
        sys.stdout.buffer.flush()
        if not result.get("ok"):
            sys.exit(1)
    except Exception as e:  # noqa: BLE001
        err_payload = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        try:
            sys.stdout.buffer.write(
                (json.dumps(err_payload, ensure_ascii=False) + "\n").encode("utf-8")
            )
            sys.stdout.buffer.flush()
        except OSError:
            pass
        sys.exit(1)
