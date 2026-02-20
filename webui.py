"""
Unified Flask web UI for dronmakr: serves auditionr and beatbuildr on a single server.
"""

import eventlet

eventlet.monkey_patch()

import threading
import time
import webbrowser

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO

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
from settings import ensure_settings, get_setting, load_settings, save_settings

# Helpers for unified socket connect
from utils import get_auditionr_folder_counts, get_latest_exports, get_presets
from generate_midi import get_patterns
from beatbuildr import generate_random_drum_kit

app = Flask(__name__, static_folder="static", template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*")

DEBUG_WEBSOCKETS = False


@socketio.on("connect")
def handle_connect():
    """Unified connect: send auditionr exports/configs and beatbuildr kit to all clients."""
    if DEBUG_WEBSOCKETS:
        print("Client connected via WebSocket")
    socketio.emit("exports", {"files": get_latest_exports()})
    socketio.emit("folder_counts", get_auditionr_folder_counts())
    socketio.emit("configs", {"presets": get_presets(), "patterns": get_patterns()})
    drum_kit = generate_random_drum_kit()
    socketio.emit("kit", drum_kit)


@app.route("/")
def index():
    """Landing page with links to auditionr and beatbuildr."""
    return render_template("index.html", version=__version__)


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
    for k, v in data.items():
        if isinstance(v, str):
            settings[k] = v
    save_settings(settings)
    return jsonify({"ok": True})


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
