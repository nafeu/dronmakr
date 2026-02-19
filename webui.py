"""
Unified Flask web UI for dronmakr: serves auditionr and beatbuildr on a single server.
"""

import eventlet

eventlet.monkey_patch()

import threading
import time
import webbrowser

from flask import Flask, render_template
from flask_socketio import SocketIO

from version import __version__
from utils import get_version, with_final_main_prompt, with_main_prompt as with_prompt

# Import registration and helpers from sub-apps
from auditionr import register_auditionr
from beatbuildr import register_beatbuildr, ensure_beat_patterns

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

    ensure_beat_patterns()

    print(get_version())
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
