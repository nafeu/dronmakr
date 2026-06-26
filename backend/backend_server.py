"""Python sidecar entry: starts the Flask backend for the Tauri desktop shell."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

os.environ.setdefault("DRONMAKR_ASYNC_MODE", "threading")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dronmakr backend server")
    parser.add_argument("--port", type=int, default=3766, help="HTTP listen port")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose logging and Flask debug mode",
    )
    parser.add_argument(
        "--dev-frontend",
        action="store_true",
        help="Serve live Jinja templates from assets/ with auto-reload (development only)",
    )
    parser.add_argument(
        "--smoke-imports",
        action="store_true",
        help="Minimal import check for CI (soundfile + libsndfile)",
    )
    if "--audio-worker" in sys.argv or "--pedalboard-worker" in sys.argv:
        sys.argv = [a for a in sys.argv if a not in ("--audio-worker", "--pedalboard-worker")]
        from dronmakr.audio.audio_worker import run_stdio_worker

        run_stdio_worker()
        raise SystemExit(0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke_imports:
        import soundfile as _sf  # noqa: PLC0415

        print("smoke: soundfile OK", _sf.__libsndfile_version__, flush=True)
        raise SystemExit(0)

    if getattr(sys, "frozen", False):
        from dronmakr.core.server_error_logging import ensure_server_error_file_logging

        ensure_server_error_file_logging(announce=False)

    from dronmakr.server.webui import start_server

    stop = threading.Event()

    def _handle_signal(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    start_server(
        debug=args.debug,
        port=int(args.port),
        host=str(args.host),
        build_sample_cache=not getattr(sys, "frozen", False),
        dev_frontend=args.dev_frontend,
    )
    print(f"[backend] ready on http://{args.host}:{args.port}", flush=True)

    while not stop.wait(0.5):
        pass
    print("[backend] shutting down", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(0) from None
