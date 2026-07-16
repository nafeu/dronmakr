#!/usr/bin/env python3
"""Validate exported WAV files for E2E smoke tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def validate_wav(
    path: str | Path,
    *,
    min_duration_s: float = 0.05,
    min_rms: float = 1e-5,
    min_peak: float = 1e-5,
    expected_sample_rate: int | None = None,
) -> dict:
    wav_path = Path(path)
    if not wav_path.is_file():
        raise FileNotFoundError(f"WAV not found: {wav_path}")

    data, sample_rate = sf.read(str(wav_path), always_2d=True)
    if data.size == 0:
        raise ValueError("WAV contains no samples")

    duration_s = float(data.shape[0]) / float(sample_rate)
    if duration_s < min_duration_s:
        raise ValueError(f"duration {duration_s:.4f}s below minimum {min_duration_s}s")

    if expected_sample_rate is not None and int(sample_rate) != int(expected_sample_rate):
        raise ValueError(
            f"sample rate {sample_rate} != expected {expected_sample_rate}"
        )

    mono = np.mean(data.astype(np.float64), axis=1)
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono * mono)))

    if peak < min_peak:
        raise ValueError(f"peak {peak:.8f} below silence threshold {min_peak}")
    if rms < min_rms:
        raise ValueError(f"RMS {rms:.8f} below silence threshold {min_rms}")

    return {
        "ok": True,
        "path": str(wav_path.resolve()),
        "durationS": duration_s,
        "sampleRate": int(sample_rate),
        "channels": int(data.shape[1]),
        "peak": peak,
        "rms": rms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a rendered WAV for E2E tests")
    parser.add_argument("wav", help="Path to WAV file")
    parser.add_argument("--min-duration", type=float, default=0.05)
    parser.add_argument("--min-rms", type=float, default=1e-5)
    parser.add_argument("--min-peak", type=float, default=1e-5)
    parser.add_argument("--expected-sample-rate", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON result on success")
    args = parser.parse_args()

    try:
        result = validate_wav(
            args.wav,
            min_duration_s=args.min_duration,
            min_rms=args.min_rms,
            min_peak=args.min_peak,
            expected_sample_rate=args.expected_sample_rate,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result))
    else:
        print(
            f"ok path={result['path']} duration={result['durationS']:.3f}s "
            f"sr={result['sampleRate']} peak={result['peak']:.6f} rms={result['rms']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
