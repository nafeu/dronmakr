## Vendored FFmpeg (desktop bundles)

Desktop PyInstaller builds run [`scripts/vendor_ffmpeg.py`](/scripts/vendor_ffmpeg.py) before packaging. It downloads pinned static binaries listed in [`scripts/ffmpeg_vendor_checksums.tsv`](/scripts/ffmpeg_vendor_checksums.tsv) from **[eugeneware/ffmpeg-static](https://github.com/eugeneware/ffmpeg-static)** tag **`b6.0`**, verifies SHA256, and writes:

- `resources/ffmpeg/ffmpeg` (POSIX) or `resources/ffmpeg/ffmpeg.exe` (Windows)
- `resources/ffmpeg/THIRD_PARTY_FFMPEG.txt` (concatenated upstream LICENSE + README for that profile)

Bundled FFmpeg is invoked by **Folysplitr** recording upload conversion (PCM WAV). Without running the vendor script, only `ffmpeg` found on **`$PATH`** is used (`python webui` / developer workflows).

Bump the vendored release by editing the checksums manifest and `RELEASE_TAG` in `vendor_ffmpeg.py`.

See **[LICENSE.third_party.ffmpeg](./LICENSE.third_party.ffmpeg)** for redistribution attribution.
