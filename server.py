import eventlet

eventlet.monkey_patch()

import threading
import subprocess
import os
import shutil
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from utils import get_server_version, get_latest_exports
from process_sample import (
    trim_sample_start,
    trim_sample_end,
    fade_sample_start,
    fade_sample_end,
    increase_sample_gain,
    decrease_sample_gain,
)


EXPORTS_DIR = "exports"
ARCHIVE_DIR = "archive"

app = Flask(__name__, static_folder="static", template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*")  # WebSockets enabled


@socketio.on("connect")
def handle_connect():
    """Log WebSocket connections."""
    print("Client connected via WebSocket")
    socketio.emit("exports", {"files": get_latest_exports()})


@app.route("/exports/<path:filename>")
def serve_exported_file(filename):
    """Allows direct access to exported .wav files"""
    return send_from_directory(EXPORTS_DIR, filename)


@app.route("/")
def index():
    return render_template("index.html")


def run_generate(params):
    """Runs the generate function as a subprocess and updates WebSocket clients."""
    try:
        socketio.emit("status", {"message": "generating...", "done": False})

        # Prepare the command
        cmd = ["python", "dronmakr.py", "generate"]
        for key, value in params.items():
            if value is not None:
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
        case _:
            return jsonify({"error": "Command not recognized"}), 400

    socketio.emit("exports", {"files": get_latest_exports()})
    return jsonify({"success": f"File processed with {command}"}), 200


if __name__ == "__main__":
    print(get_server_version())
    socketio.run(app, host="0.0.0.0", port=3766, debug=True)
