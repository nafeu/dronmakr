#!/usr/bin/env python3
"""Download pinned FFmpeg binaries for desktop packaging into resources/ffmpeg/."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

RELEASE_TAG = "b6.0"
UPSTREAM = "https://github.com/eugeneware/ffmpeg-static"
BASE_URL = f"{UPSTREAM}/releases/download/{RELEASE_TAG}"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUTPUT_DIR = _REPO_ROOT / "resources" / "ffmpeg"
_MANIFEST_PATH = Path(__file__).resolve().parent / "ffmpeg_vendor_checksums.tsv"


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _download(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "dronmakr-vendor-ffmpeg-script"})
    with urlopen(req, timeout=300) as resp:
        return resp.read()


def _parse_manifest() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    text = _MANIFEST_PATH.read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 5:
            raise ValueError(f"Bad manifest line ({len(parts)} cols): {line!r}")
        rows.append(
            {
                "profile": parts[0].strip(),
                "binary_asset": parts[1].strip(),
                "sha256": parts[2].strip().lower(),
                "license_asset": parts[3].strip(),
                "readme_asset": parts[4].strip(),
            }
        )
    return rows


def _machine() -> str:
    import platform

    return platform.machine().lower()


def detect_profile(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    plat = sys.platform
    if plat == "win32":
        # Release targets amd64 installers only today.
        return "windows-x64"
    if plat.startswith("linux"):
        mid = _machine()
        if mid in ("aarch64", "arm64"):
            sys.stderr.write(
                "vendor_ffmpeg.py: linux aarch64 vendor profile not in manifest — "
                "add a row + asset or extend --profile support.\n"
            )
            raise SystemExit(2)
        return "linux-x64"
    if plat == "darwin":
        mid = _machine()
        if mid in ("arm64", "aarch64"):
            return "macos-arm64"
        return "macos-x64"
    sys.stderr.write(f"vendor_ffmpeg.py: unsupported platform {plat!r}.\n")
    raise SystemExit(2)


def _write_upstream_notices(
    profile: str,
    binary_asset: str,
    *,
    license_text: str,
    readme_text: str,
    dest: Path,
) -> None:
    parts: list[str] = [
        "Third-party FFmpeg (vendored for Folysplitr browser recording conversion)\n"
        f"Pinned upstream: {UPSTREAM} tag {RELEASE_TAG}\n\n"
        f"Profile: {profile}\nBinary asset: {binary_asset}\n"
        "------------------------------------------------------------\n",
        "[LICENSE]\n",
        license_text.strip(),
        "\n\n[README]\n",
        readme_text.strip(),
        "\n",
    ]
    dest.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        choices=("windows-x64", "linux-x64", "macos-arm64", "macos-x64"),
        default=None,
        help="Vendor target profile (default: auto-detect)",
    )
    args = ap.parse_args()
    profile = detect_profile(args.profile)
    manifest = _parse_manifest()
    row = next((r for r in manifest if r["profile"] == profile), None)
    if not row:
        sys.stderr.write(f"No manifest entry for profile {profile!r}.\n")
        raise SystemExit(2)

    binary_url = f"{BASE_URL}/{row['binary_asset']}"
    lic_url = f"{BASE_URL}/{row['license_asset']}"
    readme_url = f"{BASE_URL}/{row['readme_asset']}"

    dest_dir = _OUTPUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[vendor_ffmpeg] profile={profile} url={binary_url}", flush=True)
    try:
        data = _download(binary_url)
    except URLError as e:
        sys.stderr.write(f"[vendor_ffmpeg] download failed: {e}\n")
        raise SystemExit(1) from e
    digest = _sha256_bytes(data).lower()
    if digest != row["sha256"]:
        sys.stderr.write(
            f"[vendor_ffmpeg] sha256 mismatch for {row['binary_asset']}: "
            f"got {digest} expected {row['sha256']}\n"
        )
        raise SystemExit(1)

    lic_text = _download(lic_url).decode("utf-8", errors="replace")
    readme_text = _download(readme_url).decode("utf-8", errors="replace")

    notices = dest_dir / "THIRD_PARTY_FFMPEG.txt"
    _write_upstream_notices(profile, row["binary_asset"], license_text=lic_text, readme_text=readme_text, dest=notices)

    if sys.platform == "win32":
        out_exe = dest_dir / "ffmpeg.exe"
        out_exe.write_bytes(data)
        old_posix = dest_dir / "ffmpeg"
        if old_posix.is_file():
            try:
                old_posix.unlink()
            except OSError:
                pass
    else:
        out_bin = dest_dir / "ffmpeg"
        out_bin.write_bytes(data)
        out_bin.chmod(out_bin.stat().st_mode | 0o755)
        old_win = dest_dir / "ffmpeg.exe"
        if old_win.is_file():
            try:
                old_win.unlink()
            except OSError:
                pass

    print(f"[vendor_ffmpeg] wrote {_OUTPUT_DIR.relative_to(_REPO_ROOT)}", flush=True)


if __name__ == "__main__":
    main()
