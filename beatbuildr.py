import eventlet

eventlet.monkey_patch()

import os
import random
import shutil
import threading
import time
import webbrowser
from pathlib import Path

import typer
from dotenv import load_dotenv
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
from version import __version__

from utils import (
    TEMP_DIR,
    delete_all_files,
    get_beatbuildr_version,
    with_final_beatbuildr_prompt,
)

app = Flask(__name__, static_folder="static", template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*")
cli = typer.Typer()

# Controls whether to log WebSocket connections; set from main()'s debug flag.
DEBUG_WEBSOCKETS = False


DRUM_ROW_ORDER = [
    "kick",
    "snar",
    "ghos",
    "clap",
    "hhat",
    "halt",
    "shkr",
    "prca",
    "prcb",
    "prcc",
    "tomm",
    "cymb",
]


ENV_TO_ROW_MAPPING = {
    "kick": "DRUM_KICK_PATHS",
    "snar": "DRUM_SNARE_PATHS",
    "ghos": "DRUM_SNARE_PATHS",
    "clap": "DRUM_CLAP_PATHS",
    "hhat": "DRUM_HIHAT_PATHS",
    "halt": "DRUM_HIHAT_PATHS",
    "shkr": "DRUM_SHAKER_PATHS",
    "prca": "DRUM_PERC_PATHS",
    "prcb": "DRUM_PERC_PATHS",
    "prcc": "DRUM_PERC_PATHS",
    "tomm": "DRUM_TOM_PATHS",
    "cymb": "DRUM_CYMBAL_PATHS",
}


def _choose_random_file(folder: Path) -> Path | None:
    """Return a random .wav file path from a folder, or None if unavailable."""
    if not folder.exists() or not folder.is_dir():
        return None
    candidates = [
        f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".wav"
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _get_random_sample_for_env(env_key: str) -> Path | None:
    """
    Pick a random WAV file from one of the comma-separated paths in the given env var.
    Returns a small descriptor dict with name and path, or None if not resolvable.
    """
    paths = os.getenv(env_key, "")
    roots = [p.strip() for p in paths.split(",") if p.strip()]
    if not roots:
        return None

    root = Path(random.choice(roots))
    return _choose_random_file(root)


def generate_random_drum_kit() -> dict:
    """
    Build a random drum kit using the same env-configured sample folders
    that power the `generate-beat` CLI command.
    """
    load_dotenv()

    # Prepare temp directory to host the current kit's samples so they can be
    # served over HTTP to the browser.
    kit_temp_root = Path(TEMP_DIR) / "beatbuildr"
    kit_temp_root.mkdir(parents=True, exist_ok=True)
    delete_all_files(str(kit_temp_root))

    kit: dict[str, dict | None] = {}
    for row in DRUM_ROW_ORDER:
        env_key = ENV_TO_ROW_MAPPING.get(row)
        if not env_key:
            continue
        src_path = _get_random_sample_for_env(env_key)
        if not src_path:
            kit[row] = None
            continue

        filename = f"{row}_{src_path.name}"
        dest_path = kit_temp_root / filename
        try:
            shutil.copy2(src_path, dest_path)
        except Exception:
            kit[row] = None
            continue

        kit[row] = {
            "name": src_path.stem,
            "path": str(src_path),
            "url": f"/kit-samples/{filename}",
        }

    return {"rows": DRUM_ROW_ORDER, "kit": kit, "base_url": "/kit-samples"}


def replace_sample_for_row(row: str) -> dict | None:
    """
    Replace a single row's sample with a new random one from the same env.
    Overwrites the file in the existing kit temp dir. Returns descriptor or None.
    """
    load_dotenv()
    env_key = ENV_TO_ROW_MAPPING.get(row)
    if not env_key:
        return None
    src_path = _get_random_sample_for_env(env_key)
    if not src_path:
        return None
    kit_temp_root = Path(TEMP_DIR) / "beatbuildr"
    kit_temp_root.mkdir(parents=True, exist_ok=True)
    filename = f"{row}_{src_path.name}"
    dest_path = kit_temp_root / filename
    try:
        shutil.copy2(src_path, dest_path)
    except Exception:
        return None
    return {
        "name": src_path.stem,
        "path": str(src_path),
        "url": f"/kit-samples/{filename}",
    }


@socketio.on("connect")
def handle_connect():
    """Handle WebSocket connections."""
    if DEBUG_WEBSOCKETS:
        print("Client connected via WebSocket (beatbuildr)")

    drum_kit = generate_random_drum_kit()
    socketio.emit("kit", drum_kit)


@socketio.on("requestNewKit")
def handle_request_new_kit():
    """Client requested a full new random drum kit."""
    drum_kit = generate_random_drum_kit()
    socketio.emit("kit", drum_kit)


@socketio.on("replaceSample")
def handle_replace_sample(payload):
    """Replace a single row's sample. Payload: { "row": "kick" }."""
    row = (payload or {}).get("row")
    if row not in DRUM_ROW_ORDER:
        return
    descriptor = replace_sample_for_row(row)
    if descriptor:
        socketio.emit("sampleReplaced", {"row": row, **descriptor})


@app.route("/kit-samples/<path:filename>")
def serve_kit_sample(filename: str):
    """Serve the currently selected drum kit samples to the browser."""
    kit_temp_root = Path(TEMP_DIR) / "beatbuildr"
    return send_from_directory(kit_temp_root, filename)


@app.route("/")
def index():
    return render_template("beatbuildr.html", version=__version__)


@cli.command()
def main(
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Enable debug logs in server"
    ),
    port: int = typer.Option(
        3767, "--port", "-p", help="The port for the beatbuildr webui server to run on"
    ),
    open_browser: bool = typer.Option(
        True,
        "--open-browser/--no-open-browser",
        help="Automatically open the beatbuildr UI in a browser",
    ),
):
    """Run the beatbuildr web server."""
    global DEBUG_WEBSOCKETS
    DEBUG_WEBSOCKETS = debug

    print(get_beatbuildr_version())
    print(
        with_final_beatbuildr_prompt(
            f"Open http://localhost:{port} in a browser to view beatbuildr"
        )
    )

    def _run_server():
        socketio.run(app, host="0.0.0.0", port=port, debug=debug)

    server_thread = threading.Thread(target=_run_server, daemon=False)
    server_thread.start()

    # Give the server a moment to bind and start listening
    time.sleep(1)

    if open_browser and server_thread.is_alive():
        try:
            webbrowser.open(f"http://localhost:{port}")
        except Exception:
            # Fail silently if the browser can't be opened.
            pass

    # Block until the server thread exits
    server_thread.join()


if __name__ == "__main__":
    cli()
