#!/usr/bin/env python3
"""Build a 256×256 PNG layer for Windows .ico (square tile; OS applies rounding)."""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CANVAS = 256
# Windows shell icons read best with edge-to-edge colour and a centred logo safe zone.
_LOGO_RATIO = 0.68


def _parse_theme_a(css_text: str) -> str:
    match = re.search(r"^\s*--theme-a\s*:\s*([^;}\s]+)\s*;", css_text, re.MULTILINE)
    if not match:
        raise ValueError("could not find --theme-a in CSS")
    raw = match.group(1).strip().strip('"').strip("'")
    if not raw.startswith("#"):
        raise ValueError(f"--theme-a value must be hex, got {raw!r}")
    return raw


def _hex_to_rgba(bg_hex: str) -> tuple[int, int, int, int]:
    h = bg_hex.strip().removeprefix("#")
    if len(h) != 6 or any(c not in "0123456789abcdefABCDEF" for c in h):
        raise ValueError(f"unsupported hex colour: {bg_hex!r}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255


def _square_gradient(rgba: tuple[int, int, int, int], size: int) -> Image.Image:
    """Subtle top-to-bottom gradient on a full square canvas."""
    r, g, b, a = rgba
    top = (min(r + 18, 255), min(g + 18, 255), min(b + 18, 255), a)
    bottom = (max(r - 12, 0), max(g - 12, 0), max(b - 12, 0), a)
    gradient = Image.new("RGBA", (size, size))
    draw = ImageDraw.Draw(gradient)
    for y in range(size):
        t = y / max(size - 1, 1)
        row = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(4))
        draw.line([(0, y), (size, y)], fill=row)
    return gradient


def compose(
    *,
    source_png: Path,
    css_path: Path,
    theme_hex_override: str | None,
    out_png: Path,
) -> None:
    if theme_hex_override:
        rgba = _hex_to_rgba(theme_hex_override)
    else:
        theme = _parse_theme_a(css_path.read_text(encoding="utf-8"))
        rgba = _hex_to_rgba(theme)

    logo_box = round(_CANVAS * _LOGO_RATIO)
    canvas = _square_gradient(rgba, _CANVAS)

    with Image.open(source_png) as im:
        logo = ImageOps.contain(im.convert("RGBA"), (logo_box, logo_box), Image.Resampling.LANCZOS)

    x = (_CANVAS - logo.width) // 2
    y = (_CANVAS - logo.height) // 2
    canvas.alpha_composite(logo, dest=(x, y))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png, format="PNG", compress_level=6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=_REPO_ROOT / "assets/static/branding/logo.png",
        help="PNG with transparency (waveform logo).",
    )
    parser.add_argument(
        "--css",
        type=Path,
        default=_REPO_ROOT / "assets/templates/_app_css_root.html",
        help="Reads --theme-a from this file unless --theme-a is passed.",
    )
    parser.add_argument(
        "--theme-a",
        dest="theme_a",
        default=None,
        help="Override backdrop colour as #RRGGBB (otherwise parsed from CSS).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(tempfile.gettempdir()) / "dronmakr-windows-icon-256.png",
        help="Intermediate 256² PNG (build_windows_app_ico.py passes a tempfile).",
    )
    args = parser.parse_args()
    compose(
        source_png=args.source,
        css_path=args.css,
        theme_hex_override=args.theme_a,
        out_png=args.out,
    )


if __name__ == "__main__":
    main()
