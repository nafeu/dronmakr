import eventlet

eventlet.monkey_patch()

import threading
import subprocess
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO
from utils import get_server_version, get_latest_exports


EXPORTS_DIR = "exports"

app = Flask(__name__, static_folder="static", template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*")  # WebSockets enabled


@socketio.on("connect")
def handle_connect():
    """Log WebSocket connections."""
    print("Client connected via WebSocket")
    sorted_files = get_latest_exports()
    socketio.emit(
        "status", {"message": "generate sample", "done": False, "files": sorted_files}
    )


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
        sorted_files = get_latest_exports()
        socketio.emit(
            "status",
            {"message": "generation complete.", "done": True, "files": sorted_files},
        )

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


if __name__ == "__main__":
    print(get_server_version())
    socketio.run(app, host="0.0.0.0", port=3766, debug=True)
