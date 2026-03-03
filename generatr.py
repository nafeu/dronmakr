"""
Generatr: web view that mirrors CLI generate options (drone, bass, transition).
Calls the same functions the CLI uses, directly (no threads), so logs go to server stdout.
"""

import os
import sys

from flask import jsonify, render_template, request

from settings import ensure_settings
from utils import (
    EXPORTS_DIR,
    format_name,
    generate_beat_name,
    generate_drone_name,
    generate_id,
    with_main_prompt as with_prompt,
    RED,
    RESET,
)
from generate_midi import generate_drone_midi
from generate_sample import generate_drone_sample, generate_beat_sample
from generate_transition import (
    generate_closh_sample,
    generate_kickboom_sample,
    generate_sweep_sample,
    parse_closh_config,
    parse_sweep_config,
)
from generate_bass import (
    generate_donk_sample,
    generate_reese_sample,
    parse_donk_config,
    parse_reese_config,
)
from process_sample import process_drone_sample


def run_generate_drone():
    """
    Run one iteration of drone generation (same flow as CLI generate-drone).
    Returns list of generated file paths. Logs go to server stdout.
    """
    if not os.path.exists("presets/presets.json"):
        raise FileNotFoundError(
            "presets/presets.json does not exist, please run build_preset.py"
        )

    filters = {}
    midi_file, selected_chart = generate_drone_midi(
        pattern=None,
        shift_octave_down=None,
        shift_root_note=None,
        filters=filters,
        notes=None,
    )
    base_sample_name = (
        f"{generate_drone_name()}_-_{selected_chart}_-_{generate_id()}"
    )
    sample_name = format_name(f"drone___{base_sample_name}")
    output_path = f"{EXPORTS_DIR}/{sample_name}"
    generated_sample = generate_drone_sample(
        input_path=midi_file,
        output_path=f"{output_path}.wav",
        instrument=None,
        effect=None,
    )
    (
        generated_sample_stretched,
        generated_sample_stretched_reverberated,
        generated_sample_stretched_reverberated_transposed,
    ) = process_drone_sample(input_path=generated_sample)

    return [
        midi_file,
        generated_sample,
        generated_sample_stretched,
        generated_sample_stretched_reverberated,
        generated_sample_stretched_reverberated_transposed,
    ]


def run_generate_bass(subcommand: str):
    """
    Run one iteration of bass generation. subcommand: "reese" | "donk".
    Returns list of one path (the WAV). Logs go to server stdout via CLI-style prints.
    """
    if subcommand == "reese":
        config = parse_reese_config(
            sound=None,
            movement=None,
            distortion=None,
            fx=None,
            disable=None,
        )
        beat_name = generate_beat_name()
        name_parts = [
            "reese",
            beat_name,
            "170bpm",
            "4bars",
            generate_id(),
        ]
        sample_name = format_name("___".join(name_parts))
        output_path = f"{EXPORTS_DIR}/{sample_name}.wav"
        output_path, _ = generate_reese_sample(
            tempo=170,
            bars=4,
            output=output_path,
            config=config,
        )
        print(with_prompt(f"generated: {output_path}"))
        return [output_path]
    elif subcommand == "donk":
        config = parse_donk_config(sound=None)
        beat_name = generate_beat_name()
        name_parts = [
            "donk",
            beat_name,
            "120bpm",
            "1bars",
            generate_id(),
        ]
        sample_name = format_name("___".join(name_parts))
        output_path = f"{EXPORTS_DIR}/{sample_name}.wav"
        output_path, _ = generate_donk_sample(
            tempo=120,
            bars=1,
            output=output_path,
            config=config,
        )
        print(with_prompt(f"generated: {output_path}"))
        return [output_path]
    else:
        raise ValueError(f"Unknown bass subcommand: {subcommand}")


def run_generate_transition(subcommand: str):
    """
    Run one iteration of transition generation. subcommand: "sweep" | "closh" | "kickboom".
    Returns list of one path (the WAV).
    """
    if subcommand == "sweep":
        config = parse_sweep_config(
            sound=None,
            curve=None,
            filter_str=None,
            tremolo=None,
            phaser=None,
            chorus=None,
            flanger=None,
            disable=None,
        )
        beat_name = generate_beat_name()
        name_parts = [
            "transition_sweep",
            beat_name,
            "120bpm",
            "8bars",
            generate_id(),
        ]
        sample_name = format_name("___".join(name_parts))
        output_path = f"{EXPORTS_DIR}/{sample_name}.wav"
        output_path, _ = generate_sweep_sample(
            tempo=120, bars=8, output=output_path, config=config
        )
        print(with_prompt(f"generated: {output_path}"))
        return [output_path]
    elif subcommand == "closh":
        config = parse_closh_config(reverb=None, delay=None)
        beat_name = generate_beat_name()
        name_parts = [
            "transition_closh",
            beat_name,
            "120bpm",
            "4bars",
            generate_id(),
        ]
        sample_name = format_name("___".join(name_parts))
        output_path = f"{EXPORTS_DIR}/{sample_name}.wav"
        output_path, _ = generate_closh_sample(
            tempo=120, bars=4, output=output_path, config=config
        )
        print(with_prompt(f"generated: {output_path}"))
        return [output_path]
    elif subcommand == "kickboom":
        config = parse_closh_config(reverb=None, delay=None)
        beat_name = generate_beat_name()
        name_parts = [
            "transition_kickboom",
            beat_name,
            "120bpm",
            "4bars",
            generate_id(),
        ]
        sample_name = format_name("___".join(name_parts))
        output_path = f"{EXPORTS_DIR}/{sample_name}.wav"
        output_path, _ = generate_kickboom_sample(
            tempo=120, bars=4, output=output_path, config=config
        )
        print(with_prompt(f"generated: {output_path}"))
        return [output_path]
    else:
        raise ValueError(f"Unknown transition subcommand: {subcommand}")


def _handle_api_generate():
    """
    POST /api/generatr/generate
    Body: { "type": "drone" | "bass" | "transition", "subcommand": optional }
    - For bass: subcommand "reese" | "donk"
    - For transition: subcommand "sweep" | "closh" | "kickboom"
    Returns: { "paths": [...], "error": null } or { "paths": [], "error": "..." }
    """
    ensure_settings()
    data = request.get_json() or {}
    gen_type = (data.get("type") or "").strip().lower()
    subcommand = (data.get("subcommand") or "").strip().lower()

    if not gen_type:
        return jsonify({"paths": [], "error": "Missing type (drone, bass, or transition)"}), 400

    if gen_type not in ("drone", "bass", "transition"):
        return jsonify({"paths": [], "error": f"Unknown type: {gen_type}"}), 400

    if gen_type == "bass" and subcommand not in ("reese", "donk"):
        return jsonify({"paths": [], "error": "Bass requires subcommand: reese or donk"}), 400
    if gen_type == "transition" and subcommand not in ("sweep", "closh", "kickboom"):
        return jsonify(
            {"paths": [], "error": "Transition requires subcommand: sweep, closh, or kickboom"}
        ), 400

    try:
        print(f"{RED}│{RESET} generatr: {gen_type}" + (f" {subcommand}" if subcommand else ""))
        if gen_type == "drone":
            paths = run_generate_drone()
        elif gen_type == "bass":
            paths = run_generate_bass(subcommand)
        else:
            paths = run_generate_transition(subcommand)
        print(f"{RED}■ generatr completed{RESET}")
        return jsonify({"paths": paths, "error": None})
    except Exception as e:
        print(with_prompt(f"generatr error: {e}"))
        return jsonify({"paths": [], "error": str(e)}), 500


def register_generatr(app):
    """Register generatr route and API on the given Flask app."""
    @app.route("/generatr")
    def generatr_page():
        from version import __version__
        return render_template("generatr.html", version=__version__)

    app.add_url_rule(
        "/api/generatr/generate",
        "generatr_api_generate",
        _handle_api_generate,
        methods=["POST"],
    )
