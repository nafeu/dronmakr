#!/usr/bin/env python3
"""Build a 1024×1024 PNG for macOS ICNS with full-bleed --theme-a backdrop + centred logo."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageOps

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CANVAS = 1024
_LOGO_BOX = round(_CANVAS * 0.72)


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

    # Full-bleed opaque canvas: translucent margins read as muddy white in Finder/Dock previews.
    base = Image.new("RGBA", (_CANVAS, _CANVAS), rgba)

    with Image.open(source_png) as im:
        logo = ImageOps.contain(im.convert("RGBA"), (_LOGO_BOX, _LOGO_BOX), Image.Resampling.LANCZOS)

    x = (_CANVAS - logo.width) // 2
    y = (_CANVAS - logo.height) // 2
    base.alpha_composite(logo, dest=(x, y))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    base.save(out_png, format="PNG", compress_level=6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=_REPO_ROOT / "static/branding/android-chrome-512x512.png",
        help="PNG with transparency (typically the 512² PWA icon).",
    )
    parser.add_argument(
        "--css",
        type=Path,
        default=_REPO_ROOT / "templates/_app_css_root.html",
        help="Reads --theme-a from this file unless --theme-a is passed.",
    )
    parser.add_argument(
        "--theme-a",
        dest="theme_a",
        default=None,
        help='Override backdrop colour as #RRGGBB (otherwise parsed from CSS).',
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "packaging/macos/icns-layer-1024.png",
        help="Intermediate 1024² PNG consumed by scripts/build_mac_app_icns.sh",
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
