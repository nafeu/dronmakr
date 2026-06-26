"""
Unified Flask web UI for dronmakr: serves auditionr and beatbuildr on a single server.
"""

import os

os.environ.setdefault("DRONMAKR_ASYNC_MODE", "threading")

import logging
import re
import sys
import threading
import time
import urllib.error
import urllib.request

from flask import Flask, Response, jsonify, redirect, request, send_from_directory, url_for
from flask_socketio import SocketIO

from dronmakr.core.bundle_paths import get_frontend_dist_dir, get_static_dir
from dronmakr.core.utils import get_version, with_main_prompt as with_prompt

# Import registration and helpers from sub-apps
from dronmakr.apps.auditionr import register_auditionr
from dronmakr.apps.beatbuildr import (
    register_beatbuildr,
    ensure_beat_patterns,
    ensure_drum_kits,
    ensure_all_sample_caches,
)
from dronmakr.core.native_folder_picker import pick_folder_subprocess
from dronmakr.core.settings import (
    DRUM_PATH_KEYS,
    DEFAULT_DRUM_PATH_PRESET_NAME,
    ensure_settings,
    get_active_drum_path_preset_name,
    get_drum_path_presets,
    get_setting,
    get_files_root,
    has_configured_drum_paths,
    has_configured_files_root,
    has_configured_plugin_paths,
    load_settings,
    set_files_root,
    save_settings,
    set_active_drum_path_preset,
    ensure_managed_files_root,
    ensure_folysplitr_drum_path_preset,
)

# Helpers for unified socket connect
from dronmakr.core.utils import (
    SAVED_DIR,
    export_collections_package,
    get_auditionr_folder_counts,
    get_latest_exports,
    get_presets,
    get_collections_files,
    sanitize_export_wavs,
    trash_selected_saved_samples,
    validate_saved_paths_for_package,
)
from dronmakr.generate.generate_midi import get_patterns
from dronmakr.core.config_validation import validate_server_config_names
from dronmakr.processing.processing_actions import get_processing_actions_payload
from dronmakr.apps.folysplitr import ensure_recordings_dir, ensure_splits_dirs, register_folysplitr

from dronmakr.core.server_error_logging import (
    ensure_server_error_file_logging,
    log_server_session_start,
    mirror_errors_log_to,
    register_flask_server_error_signals,
)
from dronmakr.server.dev_frontend import enable_dev_frontend, get_dev_frontend


app = Flask(__name__, static_folder=str(get_static_dir()))
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

DEBUG_WEBSOCKETS = False
_FRONTEND_DIST = get_frontend_dist_dir()


def _serve_page(filename: str):
    dev = get_dev_frontend()
    if dev is not None:
        try:
            html = dev.render_page(filename)
        except KeyError:
            return jsonify({"error": f"Unknown page: {filename}"}), 404
        except Exception as exc:
            app.logger.exception("dev-frontend render failed for %s", filename)
            return jsonify({"error": f"Template render failed: {exc}"}), 500
        return Response(html, mimetype="text/html; charset=utf-8")

    if not (_FRONTEND_DIST / filename).is_file():
        return jsonify({"error": f"Missing built page: {filename}. Run scripts/build_frontend.py."}), 503
    return send_from_directory(str(_FRONTEND_DIST), filename)


@app.route("/dev/reload-check")
def dev_reload_check():
    """Poll endpoint used by the dev auto-reload script injected into HTML pages."""
    dev = get_dev_frontend()
    if dev is None:
        return jsonify({"error": "Dev frontend mode is not enabled"}), 404
    return jsonify({"version": dev.current_version()})


@socketio.on("connect")
def handle_connect(auth=None):
    """Unified connect: send auditionr exports/configs; beatbuildr requests its kit over the socket."""
    if DEBUG_WEBSOCKETS:
        print("Client connected via WebSocket")
    socketio.emit(
        "configs",
        {
            "presets": get_presets(),
            "patterns": get_patterns(),
            "processingActions": get_processing_actions_payload(),
        },
    )
    quarantined = sanitize_export_wavs()
    socketio.emit(
        "exports",
        {
            "files": get_latest_exports(),
            "quarantined": quarantined,
        },
    )
    socketio.emit("folder_counts", get_auditionr_folder_counts())


@app.route("/")
def index():
    """Landing page with app links."""
    if not has_configured_files_root():
        return redirect(url_for("onboarding_page"))
    return _serve_page("index.html")


@app.route("/auditionr")
def auditionr_page():
    """Auditionr single-page app."""
    if not has_configured_files_root():
        return redirect(url_for("onboarding_page"))
    return _serve_page("auditionr.html")


@app.route("/beatbuildr")
def beatbuildr_page():
    """Beatbuildr single-page app."""
    if not has_configured_files_root():
        return redirect(url_for("onboarding_page"))
    return _serve_page("beatbuildr.html")


@app.route("/settings")
def settings_page():
    """Settings page for editing config/settings.json values."""
    return _serve_page("settings.html")


@app.route("/onboarding")
def onboarding_page():
    """First-run onboarding to select the dronmakr files root."""
    return _serve_page("onboarding.html")


@app.route("/about")
def about_page():
    """Desktop-friendly about / credits."""
    return _serve_page("about.html")


@app.route("/collections")
def collections_page():
    """Collections view: saved folder as waveform grid with filters and packaging sidebar."""
    if not has_configured_files_root():
        return redirect(url_for("onboarding_page"))
    return _serve_page("collections.html")


@app.route("/folysplitr")
def folysplitr_page():
    """Folysplitr recorder + split workflow page."""
    if not has_configured_files_root():
        return redirect(url_for("onboarding_page"))
    return _serve_page("folysplitr.html")


@app.route("/api/settings/config-status")
def api_settings_config_status():
    """Return whether drum and plugin path settings are usable."""
    return jsonify(
        {
            "drumPathsConfigured": has_configured_drum_paths(),
            "pluginPathsConfigured": has_configured_plugin_paths(),
        }
    )


@app.route("/api/settings/plugin-path-defaults")
def api_settings_plugin_path_defaults():
    """Return OS-recommended PLUGIN_PATHS (comma-separated)."""
    from dronmakr.presets.plugin_default_paths import default_plugin_paths_csv

    return jsonify({"pluginPaths": default_plugin_paths_csv()})


@app.route("/api/diagnostics")
def api_diagnostics():
    """Paths and environment hints for troubleshooting."""
    import sys

    from dronmakr.core.settings import SETTINGS_PATH

    log_path = ensure_server_error_file_logging()
    return jsonify(
        {
            "errorsLogPath": str(log_path),
            "settingsPath": SETTINGS_PATH,
            "frozen": bool(getattr(sys, "frozen", False)),
        }
    )


@app.route("/api/auditionr/exports")
def api_auditionr_exports():
    """HTTP fallback for auditionr queue bootstrap when the socket is slow."""
    quarantined = sanitize_export_wavs()
    return jsonify({"files": get_latest_exports(), "quarantined": quarantined})


@app.route("/api/collections/saved")
def api_collections_saved():
    """Return saved/ plus splits/**/*.wav for collections (name, path, type)."""
    return jsonify({"files": get_collections_files()})


@app.route("/api/collections/package-selection", methods=["POST"])
def api_collections_package_selection():
    """
    Validate a list of /saved/... or /splits/... paths for packaging (must exist on disk).
    Used by the collections packaging UI and future export flows.
    """
    data = request.get_json(silent=True) or {}
    paths = data.get("paths")
    if paths is None:
        return jsonify({"ok": False, "error": "Missing paths"}), 400
    if not isinstance(paths, list):
        return jsonify({"ok": False, "error": "paths must be a list"}), 400
    valid, invalid = validate_saved_paths_for_package(paths)
    return jsonify(
        {"ok": True, "items": valid, "invalidPaths": invalid, "count": len(valid)}
    )


@app.route("/api/collections/export-package", methods=["POST"])
def api_collections_export_package():
    """Copy selected saved samples into packages/{author}_-_{package}/ with new names."""
    data = request.get_json(silent=True) or {}
    paths = data.get("paths")
    if not isinstance(paths, list):
        return jsonify({"ok": False, "error": "paths must be a list"}), 400
    result = export_collections_package(
        paths_in_order=paths,
        package_name=(data.get("packageName") or "").strip(),
        author_name=(data.get("authorName") or "").strip(),
        include_generated=bool(data.get("includeGeneratedName")),
        include_style=bool(data.get("includeStyle")),
        trash_on_save=bool(data.get("trashOnSave")),
        package_layout=data.get("packageLayout") or data.get("package_layout"),
    )
    if not result.get("ok"):
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/collections/trash-selected", methods=["POST"])
def api_collections_trash_selected():
    """Move selected saved samples to trash/."""
    data = request.get_json(silent=True) or {}
    paths = data.get("paths")
    if not isinstance(paths, list):
        return jsonify({"ok": False, "error": "paths must be a list"}), 400
    result = trash_selected_saved_samples(paths)
    if not result.get("ok"):
        return jsonify(result), 400
    return jsonify(result)


def _serve_saved_file(filename):
    """Serve a file from the saved/ directory."""
    return send_from_directory(SAVED_DIR, filename)


app.add_url_rule(
    "/saved/<path:filename>",
    "serve_saved_file",
    _serve_saved_file,
)


@app.route("/api/settings", methods=["GET"])
def api_settings_get():
    """Return current settings as JSON."""
    settings = load_settings()
    settings["FILES_ROOT"] = get_files_root(settings=settings, allow_default=False)
    return jsonify(settings)


@app.route("/api/settings", methods=["POST"])
def api_settings_save():
    """Save settings from JSON body."""
    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON"}), 400
    settings = load_settings()
    drum_presets = get_drum_path_presets(settings)
    active_preset = get_active_drum_path_preset_name(settings)
    active_paths = dict(drum_presets.get(active_preset, {}))
    for k, v in data.items():
        if isinstance(v, str):
            if k in DRUM_PATH_KEYS:
                active_paths[k] = v
            elif k == "FILES_ROOT":
                try:
                    settings["FILES_ROOT"] = set_files_root(v)
                except ValueError as e:
                    return jsonify({"error": str(e)}), 400
            else:
                settings[k] = v
    drum_presets[active_preset] = active_paths
    settings["DRUM_PATH_PRESETS"] = drum_presets
    settings["ACTIVE_DRUM_PATH_PRESET"] = active_preset
    save_settings(settings)
    return jsonify({"ok": True})


@app.route("/api/settings/files-root", methods=["POST"])
def api_settings_files_root():
    """Persist files root and ensure required directories exist."""
    data = request.get_json(silent=True) or {}
    root = data.get("path")
    if not isinstance(root, str):
        return jsonify({"ok": False, "error": "path must be a string"}), 400
    try:
        resolved = set_files_root(root)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except OSError as e:
        return jsonify({"ok": False, "error": f"Could not create folders: {e}"}), 400
    return jsonify({"ok": True, "path": resolved})


@app.route("/api/drum-path-presets", methods=["GET"])
def api_drum_path_presets_get():
    """Return drum path preset names and the active preset."""
    settings = load_settings()
    presets = get_drum_path_presets(settings)
    active = get_active_drum_path_preset_name(settings)
    return jsonify(
        {
            "activePreset": active,
            "defaultPreset": DEFAULT_DRUM_PATH_PRESET_NAME,
            "presets": presets,
            "drumPathKeys": DRUM_PATH_KEYS,
        }
    )


@app.route("/api/drum-path-presets", methods=["POST"])
def api_drum_path_presets_create():
    """Create a new drum path preset."""
    data = request.get_json() or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON"}), 400
    name = (data.get("name") or "").strip()
    clone_from_active = bool(data.get("cloneFromActive", True))
    if not name:
        return jsonify({"error": "Preset name is required"}), 400

    settings = load_settings()
    presets = get_drum_path_presets(settings)
    if name in presets:
        return jsonify({"error": f'Preset "{name}" already exists'}), 400

    if clone_from_active:
        active = get_active_drum_path_preset_name(settings)
        src = presets.get(active, {})
        presets[name] = {key: src.get(key, "") for key in DRUM_PATH_KEYS}
    else:
        presets[name] = {key: "" for key in DRUM_PATH_KEYS}

    settings["DRUM_PATH_PRESETS"] = presets
    save_settings(settings)
    return jsonify({"ok": True, "name": name})


@app.route("/api/drum-path-presets/<preset_name>", methods=["DELETE"])
def api_drum_path_presets_delete(preset_name: str):
    """Delete a drum path preset except the default preset."""
    name = (preset_name or "").strip()
    if not name:
        return jsonify({"error": "Preset name is required"}), 400
    if name == DEFAULT_DRUM_PATH_PRESET_NAME:
        return jsonify({"error": "Default preset cannot be deleted"}), 400

    settings = load_settings()
    presets = get_drum_path_presets(settings)
    if name not in presets:
        return jsonify({"error": f'Preset "{name}" does not exist'}), 404

    del presets[name]
    settings["DRUM_PATH_PRESETS"] = presets
    active = get_active_drum_path_preset_name(settings)
    if active == name:
        settings["ACTIVE_DRUM_PATH_PRESET"] = DEFAULT_DRUM_PATH_PRESET_NAME
    save_settings(settings)
    return jsonify({"ok": True})


@app.route("/api/drum-path-presets/active", methods=["POST"])
def api_drum_path_presets_set_active():
    """Switch active drum path preset."""
    data = request.get_json() or {}
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON"}), 400
    name = (data.get("name") or "").strip()
    ok, result = set_active_drum_path_preset(name)
    if not ok:
        return jsonify({"error": result}), 400
    return jsonify({"ok": True, "activePreset": result})




@app.route("/api/settings/pick-folder", methods=["POST"])
def api_settings_pick_folder():
    """Open native folder picker and return selected path. Safe from Flask worker threads (no Tk)."""
    result = pick_folder_subprocess()
    if result.status == "ok":
        return jsonify({"path": result.path})
    if result.status == "cancelled":
        return jsonify({"path": "", "cancelled": True})
    return (
        jsonify(
            {
                "path": "",
                "error": "No folder picker available (install zenity/kdialog on Linux, or PowerShell on Windows).",
            }
        ),
        500,
    )


@app.route("/api/health", methods=["GET"])
def api_health():
    """Lightweight readiness probe for desktop bootstrap."""
    return jsonify({"ok": True})


@app.route("/api/update/check", methods=["GET"])
def api_update_check():
    """Report whether a newer GitHub release exists than this build."""
    from dronmakr.core.updater import fetch_update_info_throttled

    info = fetch_update_info_throttled()
    if not info:
        return jsonify({"available": False})
    return jsonify(
        {
            "available": True,
            "tag": info.tag,
            "releaseUrl": info.release_url,
        }
    )


# Register auditionr and beatbuildr routes and socket handlers on the unified app.
register_auditionr(app, socketio)
register_beatbuildr(app, socketio)
register_folysplitr(app)
register_flask_server_error_signals(app)


def _print_webui_startup_header() -> None:
    """Print version banner first, then errors.log path and other startup lines."""
    print(get_version())
    ensure_server_error_file_logging(announce=True)
    mirror_errors_log_to("werkzeug", "dronmakr.server", "flask.app")
    log_server_session_start(get_version())


def _health_probe_url(host: str, port: int) -> str:
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    return f"http://{probe_host}:{int(port)}/api/health"


def _wait_for_server_health(host: str, port: int, timeout_s: float = 30.0) -> bool:
    url = _health_probe_url(host, port)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.getcode() == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            pass
        time.sleep(0.15)
    return False


_DESKTOP_STDERR_LOGGING_CONFIGURED = False
_QUIET_PROBE_LOG_FILTER: logging.Filter | None = None
_QUIET_PROBE_REQUEST_RE = re.compile(
    r'"(?:GET|HEAD) (?:'
    + "|".join(re.escape(path) for path in ("/api/health", "/dev/reload-check"))
    + r")(?:\?[^\"]*)? HTTP/[^\"]+\" (\d{3})"
)


class _QuietProbeLogFilter(logging.Filter):
    """Drop successful health/reload probe lines from werkzeug access logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        match = _QUIET_PROBE_REQUEST_RE.search(record.getMessage())
        if match is None:
            return True
        status = int(match.group(1))
        return not (200 <= status < 400)


def _configure_quiet_probe_request_logs() -> logging.Filter:
    global _QUIET_PROBE_LOG_FILTER
    if _QUIET_PROBE_LOG_FILTER is None:
        _QUIET_PROBE_LOG_FILTER = _QuietProbeLogFilter()
        for logger_name in ("werkzeug", "geventwebsocket.handler"):
            logging.getLogger(logger_name).addFilter(_QUIET_PROBE_LOG_FILTER)
    return _QUIET_PROBE_LOG_FILTER


def configure_desktop_process_logging(level: int = logging.INFO) -> None:
    """
    Send Werkzeug/Flask/Socket.IO logs to stderr so desktop mode shows HTTP
    and engine traffic in the terminal (or PyInstaller console window).
    """
    global _DESKTOP_STDERR_LOGGING_CONFIGURED
    if _DESKTOP_STDERR_LOGGING_CONFIGURED:
        return
    _DESKTOP_STDERR_LOGGING_CONFIGURED = True

    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    datefmt = "%H:%M:%S"
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    handler.addFilter(_configure_quiet_probe_request_logs())
    root.addHandler(handler)
    root.setLevel(level)
    for name in (
        "werkzeug",
        "flask.app",
        "engineio.server",
        "socketio.server",
        "geventwebsocket.handler",
    ):
        logging.getLogger(name).setLevel(level)


def start_server(
    debug: bool = False,
    port: int = 3766,
    host: str = "127.0.0.1",
    build_sample_cache: bool = True,
    stderr_logging: bool = True,
    dev_frontend: bool = False,
) -> threading.Thread:
    """Start web server in a background thread for desktop runtime."""
    global DEBUG_WEBSOCKETS
    DEBUG_WEBSOCKETS = debug
    _print_webui_startup_header()
    _configure_quiet_probe_request_logs()
    if dev_frontend:
        enable_dev_frontend()
    if stderr_logging:
        configure_desktop_process_logging(
            logging.DEBUG if debug else logging.INFO
        )
        app.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        print(
            with_prompt("[desktop] process logging: stderr (werkzeug/flask/socketio)"),
            flush=True,
        )
    print(with_prompt("[desktop] startup: ensure settings"))
    ensure_settings()
    if has_configured_files_root():
        print(with_prompt("[desktop] startup: ensure managed files root"))
        ensure_managed_files_root()
        ensure_folysplitr_drum_path_preset()
    print(with_prompt("[desktop] startup: ensure beat patterns"))
    ensure_beat_patterns()
    print(with_prompt("[desktop] startup: ensure drum kits"))
    ensure_drum_kits()
    print(with_prompt("[desktop] startup: ensure recordings dir"))
    ensure_recordings_dir()
    print(with_prompt("[desktop] startup: ensure splits dirs"))
    ensure_splits_dirs()
    try:
        validate_server_config_names()
    except ValueError as e:
        print(with_prompt(str(e)))
        raise SystemExit(1) from e
    if build_sample_cache:
        print(with_prompt("[desktop] startup: build sample cache (blocking)"))
        ensure_all_sample_caches()
    else:
        print(with_prompt("[desktop] startup: sample cache warmup in background"))

    def _run_server():
        try:
            print(with_prompt(f"[desktop] server thread: binding http://{host}:{port}"))
            socketio.run(
                app,
                host=host,
                port=int(port),
                debug=debug,
                use_reloader=False,
                allow_unsafe_werkzeug=True,
            )
        except Exception as e:
            print(with_prompt(f"[desktop] server thread error: {e}"))
            raise

    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()

    if not _wait_for_server_health(host, port):
        print(with_prompt("[desktop] server did not become ready in time."), flush=True)

    if not build_sample_cache:
        def _warm_cache_background():
            try:
                print(with_prompt("[desktop] cache thread: warmup started"))
                ensure_all_sample_caches()
                print(with_prompt("[desktop] cache thread: warmup completed"))
            except Exception:
                # Keep desktop startup resilient even if cache warmup fails.
                print(with_prompt("[desktop] cache thread: warmup failed"))

        threading.Thread(target=_warm_cache_background, daemon=True).start()

    return server_thread
