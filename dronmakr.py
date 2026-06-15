import audio_host  # noqa: F401 — DawDreamer before numba-backed dsp

import fnmatch
import time
import os
import shutil
import sys
import json
import tempfile
import random
import builtins
import subprocess

if len(sys.argv) >= 2 and sys.argv[1] == "desktop":
    os.environ["DRONMAKR_ASYNC_MODE"] = "threading"

import typer

from settings import ensure_settings
from settings import get_active_drum_path_preset_name, set_active_drum_path_preset
from config_validation import validate_server_config_names
from preset_authoring import list_presets
from webui import run as run_webui
from beatbuildr import generate_random_drum_kit
from processing_actions import (
    parse_post_processing_spec,
    apply_post_processing_actions,
    apply_processing_command,
    actions_without_normalize,
)
from generate_midi import (
    coerce_drone_midi_length_bars,
    coerce_drone_midi_padding_bars,
    generate_drone_midi,
    get_pattern_config,
)
from generate_sample import generate_drone_sample, generate_beat_sample
from generate_transition import (
    generate_sweep_sample,
    generate_wash_sample,
    parse_sweep_config,
    parse_wash_config,
)
from generate_bass import (
    generate_donk_sample,
    generate_reese_sample,
    parse_donk_config,
    parse_reese_config,
)
from utils import (
    BLUE,
    format_name,
    generate_beat_header,
    generate_beat_name,
    generate_drone_name,
    generate_id,
    generate_transition_header,
    get_cli_version,
    get_version,
    process_drone_sample_header,
    RED,
    rename_samples,
    RESET,
    with_main_prompt as with_prompt,
    with_process_drone_sample_prompt,
    with_generate_beat_prompt,
    delete_all_files,
    EXPORTS_DIR,
    MIDI_DIR,
    PRESETS_PATH,
    TRASH_DIR,
)
from paths import get_managed_file
from version import __version__

GENERATED_LABEL = f"{RED}...{RESET}"

cli = typer.Typer(invoke_without_command=True)


def open_files_with_default_player(file_paths):
    """Open one or more files with the system default application."""
    if not file_paths:
        return

    try:
        if sys.platform.startswith("darwin"):
            subprocess.run(["open"] + file_paths)
        elif sys.platform.startswith("win"):
            for file_path in file_paths:
                os.startfile(file_path)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open"] + file_paths)
    except Exception as e:
        print(with_prompt(f"Failed to open files: {e}"))


def apply_post_processing_to_wavs(wav_paths: list[str], post_processing: str | None):
    try:
        actions = parse_post_processing_spec(post_processing)
    except ValueError as e:
        print(with_prompt(f"Error: {e}"))
        sys.exit(1)

    def _pp_step_banner(i: int, total: int, action: dict) -> None:
        label = action.get("token") or action.get("command", "")
        print(with_process_drone_sample_prompt(f"[{i}/{total}] {label}"))

    def _pp_normalize_banner() -> None:
        print(with_process_drone_sample_prompt("normalize"))

    for wav_path in wav_paths:
        print(process_drone_sample_header())
        print(with_process_drone_sample_prompt(os.path.basename(wav_path)))
        apply_post_processing_actions(
            wav_path,
            actions,
            on_before_chain_step=_pp_step_banner,
            on_before_finalize_normalize=_pp_normalize_banner,
        )
        print(f"{BLUE}│{RESET}")


def _export_split_post_processing_variants(
    output_stem: str, source_wav: str, actions: list[dict]
) -> list[str]:
    """
    Rename source_wav to {stem}_1ofN.wav, then for each action apply in place on a
    temp copy and write {stem}_2ofN.wav … {stem}_NofN.wav (cumulative chain).
    Normalize is not part of the cumulative chain user steps; each milestone file is
    normalized once after that step's audio is exported.
    """
    chain = actions_without_normalize(actions)
    n = len(chain) + 1
    first_path = f"{output_stem}_1of{n}.wav"
    stem_short = os.path.basename(output_stem)

    print(process_drone_sample_header())
    print(with_process_drone_sample_prompt(f"split-processing ({n} milestones) — {stem_short}"))

    os.replace(source_wav, first_path)
    paths: list[str] = [first_path]
    print(with_process_drone_sample_prompt(f"[1/{n}] _1of{n} baseline → normalize"))
    apply_processing_command(first_path, "normalize_sample", {})

    if not chain:
        print(f"{BLUE}│{RESET}")
        return paths

    out_dir = os.path.dirname(os.path.abspath(first_path)) or os.getcwd()
    fd, work_path = tempfile.mkstemp(suffix=".wav", dir=out_dir)
    os.close(fd)
    try:
        shutil.copy2(first_path, work_path)
        for i, action in enumerate(chain):
            tok = action.get("token") or action.get("command", "")
            milestone = i + 2
            step_path = f"{output_stem}_{milestone}of{n}.wav"
            print(
                with_process_drone_sample_prompt(
                    f"[{milestone}/{n}] {tok} → _{milestone}of{n} → normalize"
                )
            )
            apply_processing_command(
                work_path,
                action.get("command", ""),
                action.get("params", {}),
            )
            shutil.copy2(work_path, step_path)
            apply_processing_command(step_path, "normalize_sample", {})
            paths.append(step_path)
    finally:
        if os.path.exists(work_path):
            try:
                os.remove(work_path)
            except OSError:
                pass
    print(f"{BLUE}│{RESET}")
    return paths


def version_callback(ctx: typer.Context, value: bool):
    if value:
        ensure_settings()
        try:
            validate_server_config_names()
        except ValueError as e:
            print(with_prompt(str(e)))
            raise typer.Exit(code=1)
        print(get_cli_version())
        raise typer.Exit()


@cli.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
):
    """CLI entrypoint. With no subcommand, launches the unified web UI."""
    ensure_settings()
    try:
        validate_server_config_names()
    except ValueError as e:
        print(with_prompt(str(e)))
        raise typer.Exit(code=1)
    if ctx.invoked_subcommand is None:
        ctx.invoke(webui, debug=False, port=3766, open_browser=True)


# ---------------------------------------------------------------------------
# generate-bass (group with subcommands: reese, ...)
# ---------------------------------------------------------------------------

bass_app = typer.Typer(help="Generate bass loops. Subcommands: reese, donk.")


@bass_app.command("reese")
def bass_reese(
    tempo: int = typer.Option(
        170, "--tempo", "-t", help="Tempo in BPM (default: 170)."
    ),
    bars: int = typer.Option(
        4, "--bars", "-b", help="Length in bars (default: 4)."
    ),
    sound: str | None = typer.Option(
        None,
        "--sound",
        "-s",
        help=(
            "Sound: sub, neuro, wave_a, wave_b, sub_level, reese_level, detune_left, detune_right (root C1). "
            "Flags: sub, neuro. Oscillators: wave_a, wave_b = saw|tri|square|pulse (random if omitted). "
            'E.g. "sub;neuro", "wave_a:saw;wave_b:tri", "wave_a:square;wave_b:pulse;reese_level:1.8". Use _ for random.'
        ),
    ),
    movement: str | None = typer.Option(
        None,
        "--movement",
        "-m",
        help=(
            "Movement: filter_cutoff_low, filter_cutoff_high, filter_resonance, "
            "lfo1_rate_hz, lfo1_depth, lfo2_rate_hz, lfo2_cents."
        ),
    ),
    distortion: str | None = typer.Option(
        None,
        "--distortion",
        "-d",
        help="Distortion: drive_soft, drive_hard, hard_mix (two-stage + mix).",
    ),
    fx: str | None = typer.Option(
        None,
        "--fx",
        "-f",
        help=(
            "FX: stereo_width, haas_ms, chorus_mix, phaser_mix. "
            "E.g. \"stereo_width:0.8;haas_ms:10;chorus_mix:0.3\"."
        ),
    ),
    disable: str | None = typer.Option(
        None,
        "--disable",
        help="Disable sections: comma-separated list of sub,fx,movement,distortion.",
    ),
    iterations: int = typer.Option(
        1, "--iterations", "-n", help="Number of Reese loops to generate."
    ),
    play: bool = typer.Option(
        False, "--play", "-p", help="Open output with default WAV player."
    ),
    post_processing: str | None = typer.Option(
        None,
        "--post-processing",
        "-x",
        help=(
            "Post-processing pipeline: separate steps with commas or semicolons. "
            "Legacy tokens (e.g. fade:in 2s) or bracket syntax "
            "(e.g. fade:[style=in][duration_ms=2000];filter:[kind=lpf][cutoff_hz=800])."
        ),
    ),
):
    """Generate a Reese bass loop (raw by default; use --sound neuro for neuro-style, --sound sub for sub)."""
    ensure_settings()

    start_time = time.time()
    iterations = max(1, iterations)

    print(get_version())
    print(with_prompt("generate-bass reese"))
    print(with_prompt(f"  tempo               {tempo}"))
    print(with_prompt(f"  bars                {bars}"))
    print(with_prompt(f"  iterations          {iterations}"))
    print(with_prompt(f"  post-processing     {post_processing if post_processing else GENERATED_LABEL}"))
    print(with_prompt(f"  play when done      {play}"))
    print(f"{RED}│{RESET}")

    results: list[str] = []
    for i in range(iterations):
        # Re-parse config each iteration so each loop can have different random params
        config = parse_reese_config(
            sound=sound,
            movement=movement,
            distortion=distortion,
            fx=fx,
            disable=disable,
        )
        beat_name = generate_beat_name()
        name_parts = [
            "reese",
            beat_name,
            f"{tempo}bpm",
            f"{bars}bars",
            generate_id(),
        ]
        sample_name = format_name("___".join(name_parts))
        output_path = f"{EXPORTS_DIR}/{sample_name}.wav"
        output_path, params_used = generate_reese_sample(
            tempo=tempo,
            bars=bars,
            output=output_path,
            config=config,
        )
        results.append(output_path)

        if iterations == 1:
            p = params_used
            filt = "res" if p.get("use_resonant_filter") else "smooth"
            waves = f"{p.get('wave_a', 'saw')}/{p.get('wave_b', 'saw')}"
            desc = (
                f"C1 sub={p['sub_level']:.2f} reese={p['reese_level']:.2f} "
                f"osc={waves} detune=({p['detune_left']:.0f},{p['detune_right']:.0f})c "
                f"filter={filt} {p['main_cutoff_low']:.0f}-{p['main_cutoff_high']:.0f}Hz "
                f"dry={p.get('reese_dry_mix', 0):.2f} drive=({p['drive_soft']:.2f},{p['drive_hard']:.2f}) stereo={p['stereo_width']:.2f}"
            )
            print(with_prompt(f"generated: {output_path}"))
            print(with_prompt(f"  used: {desc}"))
        else:
            print(with_prompt(f"  [{i + 1}/{iterations}] {output_path}"))

    end_time = time.time()
    time_elapsed = round(end_time - start_time)
    print(f"{RED}■ completed in {time_elapsed}s{RESET}")
    if iterations > 1:
        for r in results:
            print(with_prompt(f"generated: {r}"))
    apply_post_processing_to_wavs(results, post_processing)
    if play and results:
        open_files_with_default_player(results)
    return results


@bass_app.command("donk")
def bass_donk(
    tempo: int = typer.Option(
        120, "--tempo", "-t", help="Tempo in BPM (default: 120)."
    ),
    bars: int = typer.Option(
        1, "--bars", "-b", help="Length in bars (default: 1)."
    ),
    sound: str | None = typer.Option(
        None,
        "--sound",
        "-s",
        help=(
            "Sound: base_freq (40-80), wave (sine|tri), pitch_start_semitones (12-24), "
            "pitch_decay_ms (5-30), amp_attack_ms, amp_decay_ms, amp_sustain, amp_release_ms, "
            "click (flag), click_level, sat_drive, sat_mix, lpf_cutoff (800-3000), lpf_resonance. Use _ for random."
        ),
    ),
    iterations: int = typer.Option(
        1, "--iterations", "-n", help="Number of donk loops to generate."
    ),
    play: bool = typer.Option(
        False, "--play", "-p", help="Open output with default WAV player."
    ),
    post_processing: str | None = typer.Option(
        None,
        "--post-processing",
        "-x",
        help=(
            "Post-processing pipeline: separate steps with commas or semicolons. "
            "Legacy tokens (e.g. fade:in 2s) or bracket syntax "
            "(e.g. fade:[style=in][duration_ms=2000];filter:[kind=lpf][cutoff_hz=800])."
        ),
    ),
):
    """Generate a donk bass loop: short percussive hits with pitch-drop, mono. UK donk / hard bounce."""
    ensure_settings()

    start_time = time.time()
    iterations = max(1, iterations)

    print(get_version())
    print(with_prompt("generate-bass donk"))
    print(with_prompt(f"  tempo               {tempo}"))
    print(with_prompt(f"  bars                {bars}"))
    print(with_prompt(f"  iterations          {iterations}"))
    print(with_prompt(f"  post-processing     {post_processing if post_processing else GENERATED_LABEL}"))
    print(with_prompt(f"  play when done      {play}"))
    print(f"{RED}│{RESET}")

    results: list[str] = []
    for i in range(iterations):
        config = parse_donk_config(sound=sound)
        beat_name = generate_beat_name()
        name_parts = [
            "donk",
            beat_name,
            f"{tempo}bpm",
            f"{bars}bars",
            generate_id(),
        ]
        sample_name = format_name("___".join(name_parts))
        output_path = f"{EXPORTS_DIR}/{sample_name}.wav"
        output_path, params_used = generate_donk_sample(
            tempo=tempo,
            bars=bars,
            output=output_path,
            config=config,
        )
        results.append(output_path)

        if iterations == 1:
            p = params_used
            desc = (
                f"base={p['base_freq']:.0f}Hz wave={p['wave']} "
                f"pitch={p['pitch_start_semitones']:.0f}st decay={p['pitch_decay_ms']:.0f}ms "
                f"amp_d={p['amp_decay_ms']:.0f}ms sat_mix={p['sat_mix']:.2f} lpf={p['lpf_cutoff']:.0f}Hz"
            )
            print(with_prompt(f"generated: {output_path}"))
            print(with_prompt(f"  used: {desc}"))
        else:
            print(with_prompt(f"  [{i + 1}/{iterations}] {output_path}"))

    end_time = time.time()
    time_elapsed = round(end_time - start_time)
    print(f"{RED}■ completed in {time_elapsed}s{RESET}")
    if iterations > 1:
        for r in results:
            print(with_prompt(f"generated: {r}"))
    apply_post_processing_to_wavs(results, post_processing)
    if play and results:
        open_files_with_default_player(results)
    return results


cli.add_typer(bass_app, name="generate-bass")


@cli.command(name="generate-drone")
def generate_drone(
    name: str = typer.Option(
        None, "--name", "-n", help="Name for the generated sample."
    ),
    notes: str = typer.Option(
        None,
        "--notes",
        "-N",
        help="Comma separated list of notes with octave numbers (e.g., C2,D#3,F#3). Overrides other MIDI generation options.",
    ),
    chart_name: str = typer.Option(
        None, "--chart-name", "-c", help="Chart name to filter chords/scales."
    ),
    instrument: str = typer.Option(
        None, "--instrument", "-i", help="Name of the instrument."
    ),
    effect: str = typer.Option(
        None, "--effect", "-e", help="Name of a saved single effect or FX chain."
    ),
    tags: str = typer.Option(
        None,
        "--tags",
        "-t",
        help="Comma delimited list of tags to filter chords/scales.",
    ),
    roots: str = typer.Option(
        None,
        "--roots",
        "-r",
        help="Comma delimited list of roots to filter chords/scales.",
    ),
    chart_type: str = typer.Option(
        None,
        "--chart-type",
        "-y",
        help="Type of chart used for midi, either 'chord' or 'scale'.",
    ),
    pattern: str = typer.Option(
        None,
        "--pattern",
        "-s",
        help="Name of midi pattern used to play virtual instrument.",
    ),
    length_bars: int = typer.Option(
        16,
        "--length",
        "--bars",
        help="MIDI musical length in bars: 4, 8, 16 (default), 32, or 64.",
    ),
    padded_silence_bars: int = typer.Option(
        0,
        "--padded-silence",
        help=(
            "Silence appended after the pattern (bars): 0 (default), "
            "or 4 / 8 / 16 / 32 / 64. Extends MIDI duration for DawDreamer offline renders."
        ),
    ),
    iterations: int = typer.Option(
        1,
        "--iterations",
        "-I",
        help="Number of times to generate samples (default: 1).",
    ),
    shift_octave_down: bool = typer.Option(
        None, "--shift-octave-down", "-O", help="Shift all notes one octave down."
    ),
    shift_root_note: bool = typer.Option(
        None, "--shift-root-note", "-R", help="Shift root note one octave down."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="Verify CLI options"),
    log_server: bool = typer.Option(
        False, "--log-server", "-v", help="Run logs as server mode"
    ),
    play: bool = typer.Option(
        False,
        "--play",
        help="Open all generated files with the system's default player",
    ),
    post_processing: str | None = typer.Option(
        None,
        "--post-processing",
        "-x",
        help=(
            "Post-processing pipeline: separate steps with commas or semicolons. "
            "Legacy tokens (e.g. fade:in 2s) or bracket syntax "
            "(e.g. fade:[style=in][duration_ms=2000];filter:[kind=lpf][cutoff_hz=800])."
        ),
    ),
    split_processing: bool = typer.Option(
        False,
        "--split-processing",
        help=(
            "With --post-processing, write cumulative WAVs named _1ofN … _NofN after "
            "each step (original, then after each action in order) instead of one in-place chain."
        ),
    ),
):
    """Generate n iterations of samples (.wav) with parameters"""
    start_time = time.time()

    try:
        length_bars = coerce_drone_midi_length_bars(length_bars)
        padded_silence_bars = coerce_drone_midi_padding_bars(padded_silence_bars)
    except ValueError as e:
        print(with_prompt(f"Error: {e}"))
        sys.exit(1)

    if not log_server:
        print(get_version())

    from utils import resolve_presets_index_path

    if not resolve_presets_index_path():
        print(
            with_prompt(
                "'config/presets.json' does not exist — open Patchcraftr from the desktop tray (Launch patchcraftr)."
            )
        )
        sys.exit(1)

    print(with_prompt(f"sample name          {name if name else GENERATED_LABEL}"))
    print(with_prompt(f"sound design"))
    print(
        with_prompt(
            f"  instrument         {instrument if instrument else GENERATED_LABEL}"
        )
    )
    print(with_prompt(f"  effect             {effect if effect else GENERATED_LABEL}"))
    if notes:
        print(with_prompt("notes                " + notes))
    else:
        print(with_prompt(f"thematics"))
        print(
            with_prompt(
                f"  chart name         {chart_name if chart_name else GENERATED_LABEL}"
            )
        )
        print(with_prompt(f"  tags               {tags if tags else GENERATED_LABEL}"))
        print(
            with_prompt(f"  roots              {roots if roots else GENERATED_LABEL}")
        )
        print(
            with_prompt(
                f"  chart type         {chart_type if chart_type else GENERATED_LABEL}"
            )
        )
        print(with_prompt(f"midi customization"))
        print(
            with_prompt(
                f"  pattern            {pattern if pattern else GENERATED_LABEL}"
            )
        )
        print(with_prompt(f"  length (bars)      {length_bars}"))
        print(with_prompt(f"  padded silence     {padded_silence_bars} bars"))
        print(
            with_prompt(
                f"  shift octave down  {shift_octave_down if shift_octave_down else GENERATED_LABEL}"
            )
        )
        print(
            with_prompt(
                f"  shift root note    {shift_root_note if shift_root_note else GENERATED_LABEL}"
            )
        )

    print(
        with_prompt(
            f"iterations           {iterations if iterations else GENERATED_LABEL}"
        )
    )
    print(with_prompt(f"post-processing      {post_processing if post_processing else GENERATED_LABEL}"))
    print(with_prompt(f"split-processing       {split_processing}"))
    print(with_prompt(f"play when done        {play}"))
    print(f"{RED}│{RESET}")

    split_actions: list[dict] | None = None
    if split_processing:
        if not post_processing or not post_processing.strip():
            print(
                with_prompt(
                    "Error: --split-processing requires --post-processing with at least one action."
                )
            )
            sys.exit(1)
        try:
            split_actions = parse_post_processing_spec(post_processing)
        except ValueError as e:
            print(with_prompt(f"Error: {e}"))
            sys.exit(1)
        if not split_actions:
            print(
                with_prompt(
                    "Error: --split-processing requires at least one post-processing action."
                )
            )
            sys.exit(1)

    if dry_run:
        print(f"{RED}■ dry run completed{RESET}")
        return [os.path.join(MIDI_DIR, "dry_run_example.mid"), os.path.join(EXPORTS_DIR, "dry_run_export.wav")]

    filters = {}

    if tags:
        filters["tags"] = tags.split(",")
    if roots:
        filters["roots"] = roots.split(",")
    if chart_type:
        filters["type"] = chart_type
    if chart_name:
        filters["name"] = chart_name

    results = []

    for iteration in range(iterations):
        if iterations > 1:
            print(f"{RED}■ preparing")
            print(f"{RED}│{RESET}   iteration {iteration + 1} of {iterations}")
            print(f"{RED}│{RESET}")

        midi_file, selected_chart = generate_drone_midi(
            pattern=pattern,
            shift_octave_down=shift_octave_down,
            shift_root_note=shift_root_note,
            filters=filters,
            notes=notes.split(",") if notes else None,
            num_bars=length_bars,
            padded_silence_bars=padded_silence_bars,
        )
        base_sample_name = f"{name or generate_drone_name()}_-_{selected_chart}_-_{generate_id()}"
        sample_name = format_name(f"drone___{base_sample_name}")
        output_path = f"{EXPORTS_DIR}/{sample_name}"
        generated_sample = generate_drone_sample(
            input_path=midi_file,
            output_path=f"{output_path}.wav",
            instrument=instrument,
            effect=effect,
        )
        results.append(midi_file)
        if split_actions is not None:
            variant_paths = _export_split_post_processing_variants(
                output_path, generated_sample, split_actions
            )
            results.extend(variant_paths)
        else:
            results.append(generated_sample)

    end_time = time.time()
    time_elapsed = round(end_time - start_time)
    print(f"{RED}■ completed in {time_elapsed}s{RESET}")

    for index, result in enumerate(results):
        if index == 0:
            print(with_prompt(f"generated: {result}"))
        else:
            print(with_prompt(f"           {result}"))

    wav_files = [f for f in results if f.endswith(".wav")]
    if split_actions is None:
        apply_post_processing_to_wavs(wav_files, post_processing)

    # Open all generated .wav files if play is enabled
    if play and results:
        if wav_files:
            open_files_with_default_player(wav_files)

    return results


def _load_drum_kits_for_cli():
    """Load drum kits from config/drum-kits.json for CLI use."""
    path = get_managed_file("config", "drum-kits.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


CLI_SUPPORTED_TEMPO_VARIANTS: dict[str, float] = {
    "1/2": 0.5,
    "1/4": 0.25,
    "3/4": 0.75,
}


def _parse_cli_variants(raw: str | None) -> list[str]:
    """Parse -a/--variants CSV into ordered unique supported values."""
    if not raw:
        return []
    out: list[str] = []
    for token in [x.strip() for x in str(raw).split(",")]:
        if not token:
            continue
        if token not in CLI_SUPPORTED_TEMPO_VARIANTS:
            raise typer.BadParameter(
                f"Unsupported variant '{token}'. Supported values: {', '.join(CLI_SUPPORTED_TEMPO_VARIANTS.keys())}"
            )
        if token not in out:
            out.append(token)
    return out


@cli.command(name="generate-beat")
def generate_beat(
    tempo: int = typer.Option(
        None, "--tempo", "-t", help="Tempo in BPM (uses pattern tempo when not specified)"
    ),
    loops: int = typer.Option(
        2, "--loops", "-l", help="Number of bars per pattern loop"
    ),
    pattern: str = typer.Option(
        None,
        "--pattern",
        "-p",
        help="Drum pattern style (random from config if not specified)",
    ),
    kit: str = typer.Option(
        None,
        "--kit",
        "-k",
        help="Drum kit name from config/drum-kits.json (uses env sample folders if not specified)",
    ),
    randomization_config: str | None = typer.Option(
        None,
        "--randomization-config",
        "-r",
        help="Drum path preset name to use when --kit is not provided.",
    ),
    variants: str | None = typer.Option(
        None,
        "--variants",
        "-a",
        help='Comma-separated tempo variants to also export. Supported: "1/2,1/4,3/4".',
    ),
    swing: float = typer.Option(
        0.0,
        "--swing",
        "-w",
        min=0.0,
        max=1.0,
        help="Rhythmic swing amount between 0 (straight) and 1 (strong swing).",
    ),
    play: bool = typer.Option(
        False,
        help="Open ALL generated WAV files together with the system's default player",
    ),
    post_processing: str | None = typer.Option(
        None,
        "--post-processing",
        "-x",
        help=(
            "Post-processing pipeline: separate steps with commas or semicolons. "
            "Legacy tokens (e.g. fade:in 2s) or bracket syntax "
            "(e.g. fade:[style=in][duration_ms=2000];filter:[kind=lpf][cutoff_hz=800])."
        ),
    ),
    iterations: int = typer.Option(
        1,
        "--iterations",
        "-I",
        help="Number of times to generate beats (default: 1).",
    ),
):
    """Generate n iterations of drum loops from env-configured sample folders."""
    start_time = time.time()

    # Load available patterns from config
    try:
        with open(get_managed_file("config", "beat-patterns.json"), "r") as f:
            beat_patterns_data = json.load(f)
            available_patterns = (
                builtins.list(beat_patterns_data.keys()) if beat_patterns_data else []
            )
            if not available_patterns:
                print(
                    with_prompt(
                        f"Error: No patterns found in config/beat-patterns.json"
                    )
                )
                sys.exit(1)
    except FileNotFoundError:
        print(with_prompt("Error: config/beat-patterns.json not found"))
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(with_prompt(f"Error: Invalid JSON in config/beat-patterns.json: {e}"))
        sys.exit(1)

    # Resolve pattern: exact match, wildcard match (*), or random if not specified
    matching_patterns = None
    if pattern is not None and pattern:
        if "*" in pattern:
            matching_patterns = [
                p for p in available_patterns if fnmatch.fnmatch(p, pattern)
            ]
            if not matching_patterns:
                print(with_prompt(f"Error: No patterns match '{pattern}'"))
                sys.exit(1)
        else:
            if pattern not in available_patterns:
                print(
                    with_prompt(
                        f"Error: Pattern '{pattern}' not found in config/beat-patterns.json"
                    )
                )
                sys.exit(1)
            matching_patterns = [pattern]

    # Resolve kit: exact match, wildcard match (*), or settings-based samples if not specified
    matching_kits = None
    drum_kits = None
    if kit is not None and kit:
        drum_kits = _load_drum_kits_for_cli()
        available_kits = builtins.list(drum_kits.keys()) if drum_kits else []
        if not available_kits:
            print(with_prompt("Error: No kits found in config/drum-kits.json"))
            sys.exit(1)

        if "*" in kit:
            matching_kits = [k for k in available_kits if fnmatch.fnmatch(k, kit)]
            if not matching_kits:
                print(with_prompt(f"Error: No kits match '{kit}'"))
                sys.exit(1)
        else:
            if kit not in drum_kits:
                print(with_prompt(f"Error: Kit '{kit}' not found in config/drum-kits.json"))
                sys.exit(1)
            matching_kits = [kit]

    print(get_version())
    print(with_prompt(f"tempo"))
    print(with_prompt(f"  tempo               {tempo if tempo else GENERATED_LABEL}"))
    print(with_prompt(f"  loops               {loops}"))
    print(
        with_prompt(f"pattern               {pattern if pattern else GENERATED_LABEL}")
    )
    print(with_prompt(f"kit                   {kit if kit else GENERATED_LABEL}"))
    print(
        with_prompt(
            f"randomization-config  {randomization_config if randomization_config else GENERATED_LABEL}"
        )
    )
    print(with_prompt(f"swing                 {swing}"))
    parsed_variants = _parse_cli_variants(variants)
    print(with_prompt(f"variants              {','.join(parsed_variants) if parsed_variants else GENERATED_LABEL}"))
    print(with_prompt(f"post-processing       {post_processing if post_processing else GENERATED_LABEL}"))
    print(with_prompt(f"play when done        {play}"))
    print(
        with_prompt(
            f"iterations            {iterations if iterations else GENERATED_LABEL}"
        )
    )
    print(f"{RED}│{RESET}")

    results = []

    original_preset_name = None
    if matching_kits is None and randomization_config:
        original_preset_name = get_active_drum_path_preset_name()
        ok, result = set_active_drum_path_preset(randomization_config)
        if not ok:
            print(with_prompt(f"Error: {result}"))
            sys.exit(1)
        print(with_prompt(f"Using drum randomization preset: {result}"))

    try:
        for iteration in range(iterations):
            if iterations > 1:
                print(f"{RED}■ preparing")
                print(f"{RED}│{RESET}   iteration {iteration + 1} of {iterations}")
                print(f"{RED}│{RESET}")

            # Determine pattern for this iteration
            if matching_patterns is not None:
                current_pattern = random.choice(matching_patterns)
            else:
                current_pattern = random.choice(available_patterns)

            # Determine kit for this iteration
            current_kit_name = ""
            current_kit_paths = None
            if matching_kits is not None and drum_kits is not None:
                current_kit_name = random.choice(matching_kits)
                current_kit_paths = drum_kits.get(current_kit_name, {})
                if not isinstance(current_kit_paths, dict):
                    current_kit_paths = {}
            else:
                # Freeze one concrete random kit per iteration so tempo-variant exports
                # are true variants (tempo-only changes) of the same sample choices.
                random_kit_payload = generate_random_drum_kit()
                random_kit_rows = random_kit_payload.get("kit", {})
                current_kit_paths = {}
                if isinstance(random_kit_rows, dict):
                    for row_key, descriptor in random_kit_rows.items():
                        if not isinstance(descriptor, dict):
                            continue
                        path = descriptor.get("path")
                        if isinstance(path, str) and path.strip():
                            current_kit_paths[row_key] = path.strip()

            raw = beat_patterns_data.get(current_pattern, {})
            gs, ts, ln, meta_tempo, meta_swing = (None, None, None, None, None)
            if isinstance(raw, dict) and raw:
                gs, ts, ln, meta_tempo, meta_swing = get_pattern_config(raw)

            pattern_config = None
            if gs is not None:
                pattern_config = {"gridSize": gs, "timeSignature": ts, "length": ln}

            # Determine tempo: CLI --tempo overrides; otherwise use pattern tempo when available.
            if tempo is not None:
                current_tempo = tempo
            elif meta_tempo is not None:
                current_tempo = int(meta_tempo)
            else:
                current_tempo = random.randint(80, 180)

            # Use pattern's swing from _meta when available, otherwise CLI swing
            current_swing = meta_swing if meta_swing is not None else swing

            # Generate beat name
            beat_name = generate_beat_name()

            tempo_variants = [current_tempo]
            for token in parsed_variants:
                factor = CLI_SUPPORTED_TEMPO_VARIANTS.get(token)
                if factor is None:
                    continue
                vtempo = max(1, int(round(current_tempo * factor)))
                if vtempo not in tempo_variants:
                    tempo_variants.append(vtempo)

            for export_tempo in tempo_variants:
                # Generate output filename: drumpattern___beatname___pattern___kit?___bpm___id
                name_parts = ["drumpattern", beat_name, current_pattern]
                if current_kit_name:
                    name_parts.append(format_name(current_kit_name))
                name_parts.append(f"{export_tempo}bpm")
                name_parts.append(generate_id())
                sample_name = format_name("___".join(name_parts))
                output_path = f"{EXPORTS_DIR}/{sample_name}.wav"

                print(generate_beat_header())
                print(with_generate_beat_prompt(f"tempo: {export_tempo}"))
                print(with_generate_beat_prompt(f"pattern: {current_pattern}"))
                if current_kit_name:
                    print(with_generate_beat_prompt(f"kit: {current_kit_name}"))

                generate_beat_sample(
                    bpm=export_tempo,
                    bars=loops,
                    output=output_path,
                    style=current_pattern,
                    swing=current_swing,
                    play=False,  # Never play during generation
                    pattern_config=pattern_config,
                    kit_paths=current_kit_paths,
                )

                results.append(output_path)
    finally:
        if original_preset_name:
            set_active_drum_path_preset(original_preset_name)

    end_time = time.time()
    time_elapsed = round(end_time - start_time)
    print(f"{RED}■ completed in {time_elapsed}s{RESET}")

    for index, result in enumerate(results):
        if index == 0:
            print(with_prompt(f"generated: {result}"))
        else:
            print(with_prompt(f"           {result}"))

    apply_post_processing_to_wavs(results, post_processing)

    # Open all files at once if play is enabled
    if play and results:
        open_files_with_default_player(results)

    return results


# ---------------------------------------------------------------------------
# generate-transition (group with sub-commands)
# ---------------------------------------------------------------------------

transition_app = typer.Typer(help="Generate transition sounds (sweeps, washes, crashes).")


@transition_app.command("sweep")
def transition_sweep(
    tempo: int = typer.Option(
        120, "--tempo", "-t", help="Tempo in BPM (default: 120)."
    ),
    bars: int = typer.Option(8, "--bars", "-b", help="Length in bars (default: 8)."),
    voice: str | None = typer.Option(
        None,
        "--voice",
        "-v",
        help="Source voice: whitenoise, pinknoise, brownnoise, bluenoise, sine, saw, tri, square. Omit for random.",
    ),
    pitch_min: float | None = typer.Option(
        None,
        "--pitch-min",
        help="Oscillator sweep low pitch in Hz (20–20000). Ignored for noise voices. Omit for random.",
    ),
    pitch_max: float | None = typer.Option(
        None,
        "--pitch-max",
        help="Oscillator sweep high pitch in Hz (40–20000). Ignored for noise voices. Omit for random.",
    ),
    curve_shape: str | None = typer.Option(
        None,
        "--curve-shape",
        help=(
            "Build easing curve (e.g. easeInCubic, easeOutSine). "
            "Decay replays the same curve reversed in time after the peak. Omit for random."
        ),
    ),
    curve_peak_position: float | None = typer.Option(
        None,
        "--curve-peak-position",
        help=(
            "Where the sweep peaks along the export, as a fraction of total length "
            "(0.0–1.0; 0 = DROP, 0.5 = SWEEP, 1 = RISER). Omit for random."
        ),
    ),
    filter_type: str | None = typer.Option(
        None,
        "--filter-type",
        help="Filter type: lpf, hpf, bpf, bsf. Omit for random (filter off by default).",
    ),
    filter_cutoff_low: int | None = typer.Option(
        None, "--filter-cutoff-low", help="Filter cutoff low Hz (50–5000). Omit for random."
    ),
    filter_cutoff_high: int | None = typer.Option(
        None, "--filter-cutoff-high", help="Filter cutoff high Hz (1000–20000). Omit for random."
    ),
    tremolo_rate_min: float | None = typer.Option(
        None, "--tremolo-rate-min", help="Tremolo LFO rate min Hz. Omit for random."
    ),
    tremolo_rate_max: float | None = typer.Option(
        None, "--tremolo-rate-max", help="Tremolo LFO rate max Hz. Omit for random."
    ),
    tremolo_depth: float | None = typer.Option(
        None, "--tremolo-depth", help="Tremolo depth 0–1. Omit for random."
    ),
    phaser_rate_min: float | None = typer.Option(
        None, "--phaser-rate-min", help="Phaser LFO rate min Hz. Omit for random."
    ),
    phaser_rate_max: float | None = typer.Option(
        None, "--phaser-rate-max", help="Phaser LFO rate max Hz. Omit for random."
    ),
    phaser_depth: float | None = typer.Option(
        None, "--phaser-depth", help="Phaser depth 0–1. Omit for random."
    ),
    phaser_centre: float | None = typer.Option(
        None, "--phaser-centre", help="Phaser centre frequency Hz. Omit for random."
    ),
    phaser_feedback: float | None = typer.Option(
        None, "--phaser-feedback", help="Phaser feedback 0–1. Omit for random."
    ),
    phaser_mix: float | None = typer.Option(
        None, "--phaser-mix", help="Phaser wet mix 0–1. Omit for random."
    ),
    chorus_rate_min: float | None = typer.Option(
        None, "--chorus-rate-min", help="Chorus LFO rate min Hz. Omit for random."
    ),
    chorus_rate_max: float | None = typer.Option(
        None, "--chorus-rate-max", help="Chorus LFO rate max Hz. Omit for random."
    ),
    chorus_depth: float | None = typer.Option(
        None, "--chorus-depth", help="Chorus depth 0–1. Omit for random."
    ),
    chorus_delay: float | None = typer.Option(
        None, "--chorus-delay", help="Chorus centre delay ms. Omit for random."
    ),
    chorus_mix: float | None = typer.Option(
        None, "--chorus-mix", help="Chorus wet mix 0–1. Omit for random."
    ),
    flanger_rate_min: float | None = typer.Option(
        None, "--flanger-rate-min", help="Flanger LFO rate min Hz. Omit for random."
    ),
    flanger_rate_max: float | None = typer.Option(
        None, "--flanger-rate-max", help="Flanger LFO rate max Hz. Omit for random."
    ),
    flanger_depth: float | None = typer.Option(
        None, "--flanger-depth", help="Flanger depth 0–1. Omit for random."
    ),
    flanger_delay: float | None = typer.Option(
        None, "--flanger-delay", help="Flanger centre delay ms. Omit for random."
    ),
    flanger_feedback: float | None = typer.Option(
        None, "--flanger-feedback", help="Flanger feedback 0–0.8. Omit for random."
    ),
    flanger_mix: float | None = typer.Option(
        None, "--flanger-mix", help="Flanger wet mix 0–1. Omit for random."
    ),
    gain_min: float | None = typer.Option(
        None, "--gain-min", help="Sweep gain at curve start (0–1). Omit for random."
    ),
    gain_max: float | None = typer.Option(
        None, "--gain-max", help="Sweep gain at curve peak (0–1). Omit for random."
    ),
    disable: str | None = typer.Option(
        None,
        "--disable",
        "-d",
        help="Disable effects: filter,tremolo,phaser,chorus,flanger,gain or fx (all).",
    ),
    iterations: int = typer.Option(
        1, "--iterations", "-n", help="Number of samples to generate."
    ),
    play: bool = typer.Option(
        False, "--play", "-p", help="Open output with default WAV player."
    ),
    post_processing: str | None = typer.Option(
        None,
        "--post-processing",
        "-x",
        help=(
            "Post-processing pipeline: separate steps with commas or semicolons. "
            "Legacy tokens (e.g. fade:in 2s) or bracket syntax "
            "(e.g. fade:[style=in][duration_ms=2000];filter:[kind=lpf][cutoff_hz=800])."
        ),
    ),
):
    """Generate a sweep transition (noise or oscillator with modulated filter)."""
    start_time = time.time()
    iterations = max(1, iterations)

    print(get_version())
    print(generate_transition_header())
    print(with_prompt(f"sweep"))
    print(with_prompt(f"  tempo               {tempo}"))
    print(with_prompt(f"  bars                {bars}"))
    print(with_prompt(f"  iterations          {iterations}"))
    print(with_prompt(f"  post-processing     {post_processing if post_processing else GENERATED_LABEL}"))
    print(with_prompt(f"  play when done      {play}"))
    print(f"{RED}│{RESET}")

    results = []
    for i in range(iterations):
        config = parse_sweep_config(
            voice=voice,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            curve_shape=curve_shape,
            curve_peak_position=curve_peak_position,
            filter_type=filter_type,
            filter_cutoff_low=filter_cutoff_low,
            filter_cutoff_high=filter_cutoff_high,
            tremolo_rate_min=tremolo_rate_min,
            tremolo_rate_max=tremolo_rate_max,
            tremolo_depth=tremolo_depth,
            phaser_rate_min=phaser_rate_min,
            phaser_rate_max=phaser_rate_max,
            phaser_depth=phaser_depth,
            phaser_centre=phaser_centre,
            phaser_feedback=phaser_feedback,
            phaser_mix=phaser_mix,
            chorus_rate_min=chorus_rate_min,
            chorus_rate_max=chorus_rate_max,
            chorus_depth=chorus_depth,
            chorus_delay=chorus_delay,
            chorus_mix=chorus_mix,
            flanger_rate_min=flanger_rate_min,
            flanger_rate_max=flanger_rate_max,
            flanger_depth=flanger_depth,
            flanger_delay=flanger_delay,
            flanger_feedback=flanger_feedback,
            flanger_mix=flanger_mix,
            gain_min=gain_min,
            gain_max=gain_max,
            disable=disable,
        )
        beat_name = generate_beat_name()
        name_parts = [
            "transition_sweep",
            beat_name,
            f"{tempo}bpm",
            f"{bars}bars",
            generate_id(),
        ]
        sample_name = format_name("___".join(name_parts))
        output_path = f"{EXPORTS_DIR}/{sample_name}.wav"
        output_path, params_used = generate_sweep_sample(
            tempo=tempo, bars=bars, output=output_path, config=config
        )
        results.append(output_path)
        t = params_used
        voice_desc = t.get("sweep_voice") or t.get("voice", "noise")
        if t.get("voice") == "noise":
            voice_desc = t.get("sweep_voice") or f"{t.get('noise_type', 'white')}noise"
        elif t.get("osc_freq_low") is not None:
            voice_desc = (
                f"{voice_desc} {t.get('osc_freq_low', 0):.0f}–{t.get('osc_freq_high', 0):.0f}Hz"
            )
        mod_str = ", ".join(m for m in ["phaser", "chorus", "flanger"] if t.get(m))
        fx_str = f", fx=[{mod_str}]" if mod_str else ""
        if iterations == 1:
            print(with_prompt(f"generated: {output_path}"))
            print(
                with_prompt(
                    f"  used: {voice_desc}, cutoff {t['cutoff_low']}–{t['cutoff_high']}Hz, gain {t.get('gain_min', 0):.2f}–{t.get('gain_max', 1):.2f}, tremolo depth={t['tremolo_depth']:.2f} rate={t['tremolo_rate_min']:.1f}–{t['tremolo_rate_max']:.1f}Hz{fx_str}"
                )
            )
        else:
            print(with_prompt(f"  [{i + 1}/{iterations}] {output_path}"))

    end_time = time.time()
    time_elapsed = round(end_time - start_time)
    print(f"{RED}■ completed in {time_elapsed}s{RESET}")
    if iterations > 1:
        for r in results:
            print(with_prompt(f"generated: {r}"))
    apply_post_processing_to_wavs(results, post_processing)
    if play and results:
        open_files_with_default_player(results)
    return results


@transition_app.command("wash")
def transition_wash(
    tempo: int = typer.Option(
        120, "--tempo", "-t", help="Tempo in BPM (default: 120)."
    ),
    bars: int = typer.Option(8, "--bars", "-b", help="Length in bars (default: 8)."),
    reverb: str | None = typer.Option(
        None,
        "--reverb",
        "-r",
        help="Reverb: wet_level, length_sec, decay_sec, early_reflections, highpass_hz, tail_diffusion (0.65-0.9). Use _ for random.",
    ),
    delay: str | None = typer.Option(
        None,
        "--delay",
        "-d",
        help="Tempo-synced delay: division (1/4|1/8|1/8d|1/16|1/16d|1/32), feedback, mix. Omit or 'off' to disable.",
    ),
    stretch: float = typer.Option(
        3.0, "--stretch", "-s", help="Paulstretch factor applied after reverb (default: 3.0)."
    ),
    window_size: float = typer.Option(
        0.25,
        "--window-size",
        "-w",
        help="Paulstretch window size in seconds (default: 0.25).",
    ),
    library: str | None = typer.Option(
        None,
        "--library",
        "-l",
        help="Drum path preset name from settings. Omit to use the active preset.",
    ),
    percussion: str | None = typer.Option(
        None,
        "--percussion",
        help="Percussion folder: kick, snare, hihat, clap, perc, tom, shaker, cymbal. Omit or _ for random.",
    ),
    disable: str | None = typer.Option(
        None,
        "--disable",
        help="Disable wash effects: reverb, delay, paulstretch, or fx (all).",
    ),
    iterations: int = typer.Option(
        1, "--iterations", "-n", help="Number of samples to generate."
    ),
    play: bool = typer.Option(
        False, "--play", help="Open output with default WAV player."
    ),
    post_processing: str | None = typer.Option(
        None,
        "--post-processing",
        "-x",
        help=(
            "Post-processing pipeline: separate steps with commas or semicolons. "
            "Legacy tokens (e.g. fade:in 2s) or bracket syntax "
            "(e.g. fade:[style=in][duration_ms=2000];filter:[kind=lpf][cutoff_hz=800])."
        ),
    ),
):
    """Generate a washed percussion transition with long reverb, optional delay, and Paulstretch."""
    start_time = time.time()
    iterations = max(1, iterations)

    config = parse_wash_config(
        reverb=reverb,
        delay=delay,
        library=library,
        percussion=percussion,
        stretch=stretch,
        window_size=window_size,
        disable=disable,
    )

    original_preset_name = None
    if library and library.strip():
        original_preset_name = get_active_drum_path_preset_name()
        ok, result = set_active_drum_path_preset(library.strip())
        if not ok:
            raise typer.BadParameter(result)

    print(get_version())
    print(generate_transition_header())
    print(with_prompt(f"wash"))
    print(with_prompt(f"  tempo               {tempo}"))
    print(with_prompt(f"  bars                {bars}"))
    print(with_prompt(f"  iterations          {iterations}"))
    print(with_prompt(f"  post-processing     {post_processing if post_processing else GENERATED_LABEL}"))
    print(
        with_prompt(
            f"  reverb              {'on' if config.get('reverb_enabled', True) else 'off'}"
        )
    )
    print(
        with_prompt(
            f"  delay               {'on' if config['delay_enabled'] else 'off'}"
        )
    )
    print(
        with_prompt(
            f"  paulstretch         {'on' if config.get('paulstretch_enabled', True) else 'off'}"
        )
    )
    print(with_prompt(f"  stretch             {config.get('stretch', stretch)}"))
    print(with_prompt(f"  window_size         {config.get('window_size', window_size)}"))
    print(with_prompt(f"  library             {config.get('library') or '(active preset)'}"))
    print(with_prompt(f"  percussion          {config.get('percussion') or 'random'}"))
    print(f"{RED}│{RESET}")

    results = []
    try:
        for i in range(iterations):
            beat_name = generate_beat_name()
            perc_label = config.get("percussion") or "random"
            name_parts = [
                "transition_wash",
                str(perc_label),
                beat_name,
                f"{tempo}bpm",
                f"{bars}bars",
                generate_id(),
            ]
            sample_name = format_name("___".join(name_parts))
            output_path = f"{EXPORTS_DIR}/{sample_name}.wav"
            output_path, params_used = generate_wash_sample(
                tempo=tempo,
                bars=bars,
                output=output_path,
                config=config,
            )
            results.append(output_path)
            p = params_used
            if iterations == 1:
                rev_str = f"wet={p['reverb_wet_level']:.2f} len={p['reverb_length_sec']:.1f}s decay={p['reverb_decay_sec']:.1f}s"
                dl_str = (
                    f" delay={p['delay_division']} fb={p['delay_feedback']:.2f} mix={p['delay_mix']:.2f}"
                    if p["delay_enabled"]
                    else ""
                )
                ps_str = f" stretch={p['stretch']:.2f} win={p['window_size']:.3f}s"
                perc_str = f" perc={p['percussion']}"
                print(with_prompt(f"generated: {output_path}"))
                print(with_prompt(f"  used: {perc_str}{rev_str}{dl_str}{ps_str}"))
            else:
                print(with_prompt(f"  [{i + 1}/{iterations}] {output_path}"))
    finally:
        if original_preset_name is not None:
            set_active_drum_path_preset(original_preset_name)

    end_time = time.time()
    time_elapsed = round(end_time - start_time)
    print(f"{RED}■ completed in {time_elapsed}s{RESET}")
    if iterations > 1:
        for r in results:
            print(with_prompt(f"generated: {r}"))
    apply_post_processing_to_wavs(results, post_processing)
    if play and results:
        open_files_with_default_player(results)
    return results


cli.add_typer(transition_app, name="generate-transition")


@cli.command()
def list(
    show_chain_plugins: bool = typer.Option(
        False,
        "--show-chain-plugins",
        "-p",
        help="List all the plugins used within an effect chain",
    ),
    show_patterns: bool = typer.Option(
        False,
        "--show-patterns",
        "-t",
        help="List all available midi patterns and descriptions",
    ),
):
    """List all available presets"""
    list_presets(show_chain_plugins=show_chain_plugins, show_patterns=show_patterns)


@cli.command()
def pack(
    pack_name: str = typer.Option(
        None, "--name", "-n", help="Name for the sample pack"
    ),
    artist_name: str = typer.Option(
        None, "--artist", "-a", help="Artist name for sample pack"
    ),
    affix: bool = typer.Option(
        False, "--affix", "-f", help="Simply attach the meta to the end of the filename"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-d", help="Check updated filenames"
    ),
    delimiter: str = typer.Option(
        "^", "--delimiter", "-l", help="Original filename delimiter"
    ),
):
    """Rename all samples inside of saved folder for packaging"""
    rename_samples(
        pack_name=pack_name,
        artist_name=artist_name,
        dry_run=dry_run,
        affix=affix,
        delimiter=delimiter,
    )


@cli.command()
def webui(
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Enable debug logs in the web server"
    ),
    port: int = typer.Option(3766, "--port", "-p", help="Port for the unified web UI"),
    host: str = typer.Option("0.0.0.0", "--host", "-H", help="Host interface for the unified web UI"),
    open_browser: bool = typer.Option(
        True,
        "--open-browser/--no-open-browser",
        help="Open the web UI in the default browser on start",
    ),
):
    """Run the unified web UI (auditionr + beatbuildr on one server)."""
    run_webui(debug=debug, port=port, open_browser=open_browser, host=host)


@cli.command()
def desktop(
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Verbose stderr logging (DEBUG) for the embedded web server",
    ),
):
    """Run menu bar / system tray launcher (local server + open in browser)."""
    from desktop_app import main as run_desktop_app

    run_desktop_app(debug=debug)


@cli.command()
def reset(
    force: bool = typer.Option(
        False, "--force", "-f", help="Empty directories without confirmation"
    )
):
    """Delete all files within the exports, trash and midi directories"""
    if not force:
        confirmation = typer.confirm(
            "Are you sure you want to empty the exports, midi and trash directories?"
        )
        if not confirmation:
            return
    directories = [EXPORTS_DIR, MIDI_DIR, TRASH_DIR]
    for directory in directories:
        deleted_files_count = delete_all_files(directory)
        print(with_prompt(f"Deleted {deleted_files_count} files in {directory}"))


if __name__ == "__main__":
    cli()
