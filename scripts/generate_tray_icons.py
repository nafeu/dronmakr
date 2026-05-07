#!/usr/bin/env python3
"""Build monochrome tray icons (alpha + solid FG) from static/branding/favicon-32x32.png."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "static" / "branding" / "favicon-32x32.png"
OUT_LIGHT = ROOT / "static" / "branding" / "tray-icon-light-mode.png"
OUT_DARK = ROOT / "static" / "branding" / "tray-icon-dark-mode.png"


def _solid_rgba(src: Path, rgb: tuple[int, int, int], dest: Path) -> None:
    im = Image.open(src).convert("RGBA")
    *_, alpha = im.split()
    solid = Image.merge(
        "RGBA",
        (
            Image.new("L", im.size, rgb[0]),
            Image.new("L", im.size, rgb[1]),
            Image.new("L", im.size, rgb[2]),
            alpha,
        ),
    )
    solid.save(dest, format="PNG")
    print(f"Wrote {dest.relative_to(ROOT)}")


def main() -> None:
    if not SRC.is_file():
        print(f"Missing source: {SRC}", file=sys.stderr)
        sys.exit(1)
    # Names follow UI convention: "light mode" asset is white (for dark surfaces);
    # "dark mode" asset is black (for light surfaces).
    _solid_rgba(SRC, (255, 255, 255), OUT_LIGHT)
    _solid_rgba(SRC, (0, 0, 0), OUT_DARK)


if __name__ == "__main__":
    main()
