import eventlet

eventlet.monkey_patch()

import threading
import subprocess
import os
import shutil
import typer
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from utils import (
    get_server_version,
    get_latest_exports,
    get_presets,
    with_final_main_prompt,
    delete_all_files,
    EXPORTS_DIR,
    ARCHIVE_DIR,
    SAVED_DIR,
    TRASH_DIR,
)
from generate_midi import get_patterns
from generate_sample import apply_effect
from process_sample import (
    trim_sample_start,
    trim_sample_end,
    fade_sample_start,
    fade_sample_end,
    increase_sample_gain,
    decrease_sample_gain,
    reverse_sample,
    apply_granular_synthesis,
)
from version import __version__

app = Flask(__name__, static_folder="static", template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*")  # WebSockets enabled
cli = typer.Typer()


@socketio.on("connect")
def handle_connect():
    """Log WebSocket connections."""
    print("Client connected via WebSocket")
    socketio.emit("exports", {"files": get_latest_exports()})
    socketio.emit("configs", {"presets": get_presets(), "patterns": get_patterns()})


@app.route("/exports/<path:filename>")
def serve_exported_file(filename):
    """Allows direct access to exported .wav files"""
    return send_from_directory(EXPORTS_DIR, filename)


@app.route("/")
def index():
    return render_template("index.html", version=__version__)


def run_generate(params):
    """Runs the generate function as a subprocess and updates WebSocket clients."""
    try:
        socketio.emit("status", {"message": "generating...", "done": False})

        # Prepare the command
        cmd = ["python", "dronmakr.py", "generate"]
        for key, value in params.items():
            if key in ["shift_octave_down", "shift_root_note"]:
                cmd.append(f"--{key.replace('_', '-')}")
            elif value is not None:
                cmd.append(f"--{key}")
                cmd.append(str(value))

        # Run the subprocess and capture output
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait for completion
        process.wait()

        # Send completion message
        socketio.emit("status", {"message": "generation complete.", "done": True})
        socketio.emit("exports", {"files": get_latest_exports()})

    except Exception as e:
        print(f"Error in subprocess: {e}")  # Log error
        socketio.emit("status", {"message": f"Error: {e}", "done": True})


@app.route("/generate", methods=["POST"])
def generate_route():
    """Starts the generate process using a subprocess."""
    try:
        params = request.get_json() or {}

        # Run the subprocess in a background thread
        thread = threading.Thread(target=run_generate, args=(params,))
        thread.start()

        return jsonify({"output": "generator started."}), 202

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/skip", methods=["POST"])
def skip_file():
    params = request.get_json() or {}
    if not params["path"]:
        return jsonify({"error": "File path is required."}), 400

    file_path = f".{params['path']}"

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist."}), 404

    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)

    file_name = file_path.split(f"{EXPORTS_DIR}/")[1]

    shutil.move(file_path, os.path.join(ARCHIVE_DIR, file_name))

    socketio.emit("exports", {"files": get_latest_exports()})
    return jsonify({"success": "File moved to archive."}), 200


@app.route("/unarchive", methods=["GET"])
def unarchive_files():
    if not os.path.exists(ARCHIVE_DIR):
        return jsonify({"error": "Archive does not exist"}), 404

    if not os.path.exists(EXPORTS_DIR):
        return jsonify({"error": "Exports do not exist"}), 404

    move_all_files(ARCHIVE_DIR, EXPORTS_DIR)

    socketio.emit("exports", {"files": get_latest_exports()})
    return jsonify({"success": "Files moved back from archive"}), 200


@app.route("/emptytrash", methods=["GET"])
def empty_trash():
    if not os.path.exists(TRASH_DIR):
        return jsonify({"error": "Trash does not exist"}), 404

    delete_all_files(TRASH_DIR)

    return jsonify({"success": "Trash has been emptied"}), 200


@app.route("/reprocess", methods=["POST"])
def reprocess():
    params = request.get_json() or {}
    if not params["path"]:
        return jsonify({"error": "File path is required."}), 400

    file_path = f".{params['path']}"

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist."}), 404

    apply_effect(file_path, params["effect"])

    socketio.emit("exports", {"files": get_latest_exports()})
    socketio.emit("status", {"done": True})
    return jsonify({"success": "File moved to archive."}), 200


def move_all_files(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files = os.listdir(source_dir)

    for file in files:
        shutil.move(os.path.join(source_dir, file), target_dir)


@app.route("/delete", methods=["POST"])
def delete_file():
    params = request.get_json() or {}
    if not params["path"]:
        return jsonify({"error": "File path is required."}), 400

    file_path = f".{params['path']}"

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist."}), 404

    if not os.path.exists(TRASH_DIR):
        os.makedirs(TRASH_DIR)

    file_name = file_path.split(f"{EXPORTS_DIR}/")[1]

    shutil.move(file_path, os.path.join(TRASH_DIR, file_name))

    socketio.emit("exports", {"files": get_latest_exports()})
    return jsonify({"success": "File moved to trash."}), 200


@app.route("/refresh", methods=["GET"])
def refresh_configs():
    socketio.emit("configs", {"presets": get_presets(), "patterns": get_patterns()})
    return jsonify({"success": "Refreshed configurations"}), 200


@app.route("/save", methods=["POST"])
def save_file():
    params = request.get_json() or {}
    if not params["path"]:
        return jsonify({"error": "File path is required."}), 400

    file_path = f".{params['path']}"

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist."}), 404

    if not os.path.exists(SAVED_DIR):
        os.makedirs(SAVED_DIR)

    file_name = file_path.split(f"{EXPORTS_DIR}/")[1]

    shutil.move(file_path, os.path.join(SAVED_DIR, file_name))

    socketio.emit("exports", {"files": get_latest_exports()})
    return jsonify({"success": "File moved to saved."}), 200


@app.route("/process", methods=["POST"])
def process_file():
    params = request.get_json() or {}
    if not params["path"]:
        return jsonify({"error": "File path is required."}), 400

    file_path = f".{params['path']}"

    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist."}), 404

    if not params["command"]:
        return jsonify({"error": "File command is required."}), 400

    command = params["command"]

    match command:
        case "trim_sample_start":
            trim_sample_start(file_path, params["seconds"])
        case "trim_sample_end":
            trim_sample_end(file_path, params["seconds"])
        case "fade_sample_start":
            fade_sample_start(file_path, params["seconds"])
        case "fade_sample_end":
            fade_sample_end(file_path, params["seconds"])
        case "increase_sample_gain":
            increase_sample_gain(file_path, params["db"])
        case "decrease_sample_gain":
            decrease_sample_gain(file_path, params["db"])
        case "reverse_sample":
            reverse_sample(file_path)
        case "granularize_sample":
            apply_granular_synthesis(file_path)
        case _:
            return jsonify({"error": "Command not recognized"}), 400

    socketio.emit(
        "exports", {"files": get_latest_exports(sort_override=params["files"])}
    )
    return jsonify({"success": f"File processed with {command}"}), 200


@cli.command()
def main(
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Enable debug logs in server"
    ),
    port: int = typer.Option(
        3766, "--port", "-p", help="The port for the webui server on run on"
    ),
):
    print(get_server_version())
    print(
        with_final_main_prompt(
            f"Open http://localhost:{port} in a browser to view webui"
        )
    )
    socketio.run(app, host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    cli()
