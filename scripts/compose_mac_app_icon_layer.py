#!/usr/bin/env python3
"""Build a 1024×1024 PNG for macOS ICNS with inset squircle plate + centred logo."""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CANVAS = 1024
# Inset rounded plate — reads as a proper macOS icon tile, not edge-to-edge artwork.
_PLATE_RATIO = 0.86
# Logo fill inside the squircle plate (~58% of canvas).
_LOGO_RATIO = 0.58
# macOS squircle corner radius approximation (superellipse ~22.37% of edge length).
_SQUIRCLE_RADIUS_RATIO = 0.2237


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


def _squircle_radius(edge: int) -> int:
    return max(1, round(edge * _SQUIRCLE_RADIUS_RATIO))


def _rounded_plate(size: int, radius: int, rgba: tuple[int, int, int, int]) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    plate = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    fill = Image.new("RGBA", (size, size), rgba)
    plate.alpha_composite(fill, dest=(0, 0))
    plate.putalpha(mask)
    return plate


def _gradient_plate(
    size: int, radius: int, rgba: tuple[int, int, int, int]
) -> Image.Image:
    """Subtle top-to-bottom gradient on the squircle plate for depth."""
    r, g, b, a = rgba
    top = (min(r + 24, 255), min(g + 24, 255), min(b + 24, 255), a)
    bottom = (max(r - 16, 0), max(g - 16, 0), max(b - 16, 0), a)
    gradient = Image.new("RGBA", (size, size))
    draw = ImageDraw.Draw(gradient)
    for y in range(size):
        t = y / max(size - 1, 1)
        row = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(4))
        draw.line([(0, y), (size, y)], fill=row)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=radius, fill=255
    )
    gradient.putalpha(mask)
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

    plate_size = round(_CANVAS * _PLATE_RATIO)
    plate_inset = (_CANVAS - plate_size) // 2
    plate_radius = _squircle_radius(plate_size)
    logo_box = round(_CANVAS * _LOGO_RATIO)

    canvas = Image.new("RGBA", (_CANVAS, _CANVAS), (0, 0, 0, 0))

    shadow = _rounded_plate(plate_size, plate_radius, (0, 0, 0, 72))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=16))
    canvas.alpha_composite(shadow, dest=(plate_inset, plate_inset + 8))

    plate = _gradient_plate(plate_size, plate_radius, rgba)
    canvas.alpha_composite(plate, dest=(plate_inset, plate_inset))

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
        default=Path(tempfile.gettempdir()) / "dronmakr-icns-layer-1024.png",
        help="Intermediate 1024² PNG (build_mac_app_icns.sh passes a tempfile).",
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
