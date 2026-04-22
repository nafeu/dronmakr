"""
Unified Flask web UI for dronmakr: serves auditionr and beatbuildr on a single server.
"""

import eventlet

eventlet.monkey_patch()

import threading
import time
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO
from markdown_it import MarkdownIt

from version import __version__
from utils import get_version, with_final_main_prompt, with_main_prompt as with_prompt

# Import registration and helpers from sub-apps
from auditionr import register_auditionr
from beatbuildr import (
    register_beatbuildr,
    ensure_beat_patterns,
    ensure_drum_kits,
    ensure_all_sample_caches,
)
from settings import (
    DRUM_PATH_KEYS,
    DEFAULT_DRUM_PATH_PRESET_NAME,
    ensure_settings,
    get_active_drum_path_preset_name,
    get_drum_path_presets,
    get_setting,
    load_settings,
    save_settings,
    set_active_drum_path_preset,
)

# Helpers for unified socket connect
from utils import (
    SAVED_DIR,
    export_collections_package,
    get_auditionr_folder_counts,
    get_latest_exports,
    get_presets,
    get_saved_files,
    trash_selected_saved_samples,
    validate_saved_paths_for_package,
)
from generate_midi import get_patterns
from config_validation import validate_server_config_names
from processing_actions import get_processing_actions_payload
from folysplitr import ensure_recordings_dir, register_folysplitr

app = Flask(__name__, static_folder="static", template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*")

DEBUG_WEBSOCKETS = False
ROOT_DIR = Path(__file__).resolve().parent
README_PATH = ROOT_DIR / "README.md"


def _render_readme_html() -> str:
    """Load README.md and render it as safe HTML for the home page."""
    if not README_PATH.exists():
        return "<p>README.md not found.</p>"
    try:
        markdown_text = README_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        return f"<p>Unable to read README.md: {exc}</p>"

    md = (
        MarkdownIt("commonmark", {"html": False, "linkify": True, "typographer": True})
        .enable("table")
        .enable("strikethrough")
    )
    return md.render(markdown_text)


README_HTML = _render_readme_html()


@socketio.on("connect")
def handle_connect():
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
    socketio.emit("exports", {"files": get_latest_exports()})
    socketio.emit("folder_counts", get_auditionr_folder_counts())


@app.route("/")
def index():
    """Landing page with app links and rendered README content."""
    page_html = render_template("index.html", version=__version__)
    readme_block = (
        '<section id="readme-content" style="margin-top: 2rem;">'
        f"{README_HTML}"
        "</section>"
    )
    return page_html.replace("</main>", f"{readme_block}</main>", 1)


@app.route("/auditionr")
def auditionr_page():
    """Auditionr single-page app."""
    return render_template("auditionr.html", version=__version__)


@app.route("/beatbuildr")
def beatbuildr_page():
    """Beatbuildr single-page app."""
    return render_template("beatbuildr.html", version=__version__)


@app.route("/settings")
def settings_page():
    """Settings page for editing config/settings.json values."""
    settings = load_settings()
    return render_template("settings.html", version=__version__, settings=settings)


@app.route("/collections")
def collections_page():
    """Collections view: saved folder as waveform grid with filters and packaging sidebar."""
    return render_template("collections.html", version=__version__)


@app.route("/folysplitr")
def folysplitr_page():
    """Folysplitr placeholder page."""
    return render_template("folysplitr.html", version=__version__)


@app.route("/api/collections/saved")
def api_collections_saved():
    """Return list of saved .wav files with name, path, and type for collections."""
    return jsonify({"files": get_saved_files()})


@app.route("/api/collections/package-selection", methods=["POST"])
def api_collections_package_selection():
    """
    Validate a list of /saved/... paths for packaging (must exist on disk).
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
    return jsonify(load_settings())


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
            else:
                settings[k] = v
    drum_presets[active_preset] = active_paths
    settings["DRUM_PATH_PRESETS"] = drum_presets
    settings["ACTIVE_DRUM_PATH_PRESET"] = active_preset
    save_settings(settings)
    return jsonify({"ok": True})


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


def _pick_folder_native():
    """Open native folder picker. Tries tkinter first, then platform fallbacks."""
    import subprocess
    import sys

    # 1. Try tkinter (works when Python was built with tk support)
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(parent=root, title="Select folder")
        root.destroy()
        return (path or "").strip()
    except ImportError:
        pass
    except Exception:
        pass

    # 2. Fallback: macOS osascript (always available on macOS)
    if sys.platform == "darwin":
        try:
            # Activate Finder so the folder dialog appears in front of the browser
            script = 'tell application "Finder" to activate\nreturn POSIX path of (choose folder)'
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout.strip()
            # returncode 1 = user cancelled
            return ""
        except FileNotFoundError:
            pass
        except subprocess.TimeoutExpired:
            return ""

    # 3. Fallback: Linux zenity / kdialog
    if sys.platform.startswith("linux"):
        for cmd in [
            ["zenity", "--file-selection", "--directory"],
            ["kdialog", "--getexistingdirectory"],
        ]:
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0 and result.stdout:
                    return result.stdout.strip()
                if result.returncode != 0:
                    return ""  # user cancelled
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired:
                return ""

    return None  # no picker available


@app.route("/api/settings/pick-folder", methods=["POST"])
def api_settings_pick_folder():
    """Open native folder picker and return selected path. Requires display."""
    path = _pick_folder_native()
    if path is not None:
        return jsonify({"path": path})
    return (
        jsonify(
            {
                "path": "",
                "error": "No folder picker available. Install python3-tk (Linux) or use a Python build with tkinter.",
            }
        ),
        500,
    )


# Register auditionr and beatbuildr routes and socket handlers on the unified app.
register_auditionr(app, socketio)
register_beatbuildr(app, socketio)
register_folysplitr(app)


def run(
    debug: bool = False,
    port: int = 3766,
    open_browser: bool = True,
):
    """Run the unified web server (auditionr + beatbuildr on one port)."""
    global DEBUG_WEBSOCKETS
    DEBUG_WEBSOCKETS = debug

    if debug:
        app.config["TEMPLATES_AUTO_RELOAD"] = True

    print(get_version())
    ensure_settings()
    ensure_beat_patterns()
    ensure_drum_kits()
    ensure_recordings_dir()
    try:
        validate_server_config_names()
    except ValueError as e:
        print(with_prompt(str(e)))
        raise SystemExit(1) from e

    print(with_prompt("Building sample cache..."))
    ensure_all_sample_caches()
    print(with_prompt("Sample cache ready."))

    if debug:
        print(with_prompt(f"Open: http://localhost:{port} in your browser."))
        print(
            with_final_main_prompt(
                "Dev mode: template changes apply on refresh (no restart needed)."
            )
        )
    else:
        print(with_final_main_prompt(f"Open: http://localhost:{port} in your browser."))

    def _run_server():
        socketio.run(
            app,
            host="0.0.0.0",
            port=int(port),
            debug=debug,
            use_reloader=False,
        )

    server_thread = threading.Thread(target=_run_server, daemon=False)
    server_thread.start()

    time.sleep(1)

    if open_browser and server_thread.is_alive():
        try:
            webbrowser.open(f"http://localhost:{port}")
        except Exception:
            pass

    server_thread.join()
