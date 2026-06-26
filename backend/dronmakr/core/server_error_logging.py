"""
Rotating ``errors.log`` for dronmakr: server/traceback errors plus optional app diagnostics.

The ``dronmakr`` logger family (including ``dronmakr.server``, ``dronmakr.apps``, …)
writes at INFO and above into this file. Unhandled Flask request exceptions attach full
tracebacks via ``register_flask_server_error_signals``.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from flask import Flask
from flask.signals import got_request_exception
from dronmakr.core.settings import get_server_logs_dir

_APP_ERROR_LOG_BASENAME = "errors.log"
_MAX_BYTES = 2 * 1024 * 1024
_BACKUP_COUNT = 4

_file_handler_installed = False
_flask_signal_registered = False

_DRAONMAKR_LOGGER = "dronmakr"

_QUIET_PROBE_REQUEST_RE = re.compile(
    r'"(?:GET|HEAD) (?:'
    + "|".join(re.escape(path) for path in ("/api/health", "/dev/reload-check"))
    + r")(?:\?[^\"]*)? HTTP/[^\"]+\" (\d{3})"
)
_QUIET_ENGINEIO_RE = re.compile(
    r"(?:Received packet PONG|Sending packet PING|Sending packet MESSAGE.*[\"']ping[\"'])",
    re.IGNORECASE,
)


class QuietNoiseLogFilter(logging.Filter):
    """Drop health probes and websocket heartbeat lines from log files."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        match = _QUIET_PROBE_REQUEST_RE.search(msg)
        if match is not None and 200 <= int(match.group(1)) < 400:
            return False
        if _QUIET_ENGINEIO_RE.search(msg):
            return False
        return True


def errors_log_path() -> Path:
    """Resolved path for the rotating ``errors.log`` file."""
    return get_server_logs_dir() / _APP_ERROR_LOG_BASENAME


def server_error_log_path() -> Path:
    """Resolved path — alias for backwards compatibility (:func:`errors_log_path`)."""
    return errors_log_path()


def ensure_server_error_file_logging(*, announce: bool = False) -> str:
    """
    Attach a rotating file handler at INFO+ to logger ``dronmakr`` (idempotent).

    Returns absolute path of ``errors.log``. When ``announce`` is True, prints one
    line (with ANSI prompt when ``utils.with_main_prompt`` is available).

    Intended to run when the unified web/desktop server starts.
    """
    global _file_handler_installed

    logs_dir = get_server_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = (logs_dir / _APP_ERROR_LOG_BASENAME).resolve()
    abs_path = str(path)

    if _file_handler_installed:
        if announce:
            _announce_log_path(abs_path)
        return abs_path

    _file_handler_installed = True
    fh = RotatingFileHandler(
        path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
        delay=False,
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    fh.addFilter(QuietNoiseLogFilter())

    app_log = logging.getLogger(_DRAONMAKR_LOGGER)
    app_log.setLevel(logging.INFO)
    app_log.addHandler(fh)
    # Child loggers like dronmakr.server propagate here;
    # do not also send everything on the Python root logger to stderr via this handler.
    app_log.propagate = False

    if announce:
        _announce_log_path(abs_path)
    return abs_path


def mirror_errors_log_to(*logger_names: str) -> None:
    """Attach the rotating errors.log handler to additional loggers (werkzeug, etc.)."""
    ensure_server_error_file_logging()
    app_log = logging.getLogger(_DRAONMAKR_LOGGER)
    file_handler = next(
        (h for h in app_log.handlers if isinstance(h, RotatingFileHandler)),
        None,
    )
    if file_handler is None:
        return
    for name in logger_names:
        logger = logging.getLogger(name)
        if file_handler not in logger.handlers:
            logger.addHandler(file_handler)
        if logger.level > logging.INFO:
            logger.setLevel(logging.INFO)


def log_server_session_start(version: str) -> None:
    """Write a session-start line so errors.log shows recent activity."""
    path = ensure_server_error_file_logging()
    logging.getLogger(_DRAONMAKR_LOGGER).info(
        "dronmakr %s server session started (log: %s)", version, path
    )


def _announce_log_path(abs_path: str) -> None:
    try:
        from dronmakr.core.utils import with_main_prompt as _wp

        print(_wp(f"Error log (errors.log): {abs_path}"), flush=True)
    except Exception:
        print(f"[dronmakr] Error log (errors.log): {abs_path}", flush=True)


def register_flask_server_error_signals(app: Flask) -> None:
    """
    Extra context for traceback logs (endpoint + full path) alongside Werkzeug/Flask defaults.
    """
    global _flask_signal_registered
    if _flask_signal_registered:
        return
    _flask_signal_registered = True

    logger = logging.getLogger("dronmakr.server")

    @got_request_exception.connect_via(app)
    def _log_unhandled(sender: Flask, **kwargs: object) -> None:
        exception = kwargs.get("exception")
        try:
            from flask import has_request_context, request

            if has_request_context():
                qs = getattr(request, "query_string", b"") or b""
                qsuffix = ""
                if isinstance(qs, (bytes, bytearray)) and qs:
                    try:
                        qsuffix = "?" + qs.decode("utf-8", errors="replace")
                    except Exception:
                        qsuffix = "?…"
                ctx = f"{request.method} {request.path}{qsuffix}"
            else:
                ctx = "(outside request)"
        except Exception:
            ctx = "(request context unavailable)"
        if isinstance(exception, BaseException):
            logger.error(
                "Unhandled exception for %s",
                ctx,
                exc_info=(type(exception), exception, exception.__traceback__),
            )
        else:
            logger.error("Unhandled exception for %s", ctx, exc_info=True)


def reveal_server_error_log_for_user() -> Path:
    """Create log file/dir if missing; reveal ``errors.log`` in the OS file manager."""
    p = Path(ensure_server_error_file_logging())
    if not p.is_file():
        p.touch(exist_ok=True)
    plat = sys.platform
    if plat == "darwin":
        subprocess.Popen(["open", "-R", str(p)], close_fds=True)
    elif plat == "win32":
        subprocess.Popen(["explorer", "/select," + str(p)], close_fds=False)
    else:
        subprocess.Popen(["xdg-open", str(p.parent)], close_fds=True)
    return p
