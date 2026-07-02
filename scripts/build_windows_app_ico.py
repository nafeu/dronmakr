#!/usr/bin/env python3
"""Compose and write a multi-resolution Windows .ico for the Tauri bundle."""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "assets/static/branding/windows"
_OUT_ICO = _OUT_DIR / "dronmakr.ico"
_TAURI_ICO = _REPO_ROOT / "src-tauri/icons/icon.ico"
_COMPOSE = _REPO_ROOT / "scripts/compose_windows_app_icon_layer.py"

# Standard Windows shell / shortcut sizes (16–256 px).
_ICO_SIZES = ((16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256))


def _compose_layer(*, source: Path, css: Path, theme_a: str | None, out_png: Path) -> None:
    cmd = [
        sys.executable,
        str(_COMPOSE),
        "--source",
        str(source),
        "--css",
        str(css),
        "--out",
        str(out_png),
    ]
    if theme_a:
        cmd.extend(["--theme-a", theme_a])
    subprocess.run(cmd, check=True)


def _save_multi_size_ico(layer: Image.Image, dest: Path) -> None:
    images = [
        layer.convert("RGBA").resize(size, Image.Resampling.LANCZOS) for size in _ICO_SIZES
    ]
    ordered = sorted(images, key=lambda im: im.size[0], reverse=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ordered[0].save(
        dest,
        format="ICO",
        sizes=list(_ICO_SIZES),
        append_images=ordered[1:],
    )


def _ico_entry_count(path: Path) -> int:
    data = path.read_bytes()
    if len(data) < 6 or data[:4] != b"\0\0\x01\0":
        raise ValueError(f"{path} is not a valid ICO file")
    return struct.unpack_from("<H", data, 4)[0]


def build(*, source: Path, css: Path, theme_a: str | None) -> None:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        layer_path = Path(tmp.name)
    try:
        print("Composing Windows icon layer (Pillow)…")
        _compose_layer(source=source, css=css, theme_a=theme_a, out_png=layer_path)
        layer = Image.open(layer_path)
        _save_multi_size_ico(layer, _OUT_ICO)
        _save_multi_size_ico(layer, _TAURI_ICO)
    finally:
        layer_path.unlink(missing_ok=True)

    count = _ico_entry_count(_TAURI_ICO)
    if count < len(_ICO_SIZES):
        raise RuntimeError(
            f"expected {len(_ICO_SIZES)} embedded ICO sizes, got {count} in {_TAURI_ICO}"
        )

    print("Wrote:")
    print(f"  {_OUT_ICO} ({count} sizes, {_OUT_ICO.stat().st_size} bytes)")
    print(f"  {_TAURI_ICO} ({count} sizes, {_TAURI_ICO.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=_REPO_ROOT / "assets/static/branding/logo.png",
    )
    parser.add_argument(
        "--css",
        type=Path,
        default=_REPO_ROOT / "assets/templates/_app_css_root.html",
    )
    parser.add_argument("--theme-a", dest="theme_a", default=None)
    args = parser.parse_args()
    build(source=args.source, css=args.css, theme_a=args.theme_a)


if __name__ == "__main__":
    main()
