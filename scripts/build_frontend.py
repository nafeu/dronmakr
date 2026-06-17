#!/usr/bin/env python3
"""Compile Jinja templates into self-contained HTML pages under frontend/dist/."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "assets" / "static"
DIST = ROOT / "frontend" / "dist"

sys.path.insert(0, str(ROOT / "backend"))

from dronmakr.server.frontend_pages import (  # noqa: E402
    PAGE_SPECS,
    build_url_for,
    create_jinja_env,
    pagename_for,
)
from dronmakr.version import __version__  # noqa: E402

LINK_RE = re.compile(
    r'<link\s+rel="stylesheet"\s+href="(/static/[^"]+)"\s*/?\s*>',
    re.IGNORECASE,
)
SCRIPT_SRC_RE = re.compile(
    r'<script\s+src="(/static/[^"]+)"\s*></script>',
    re.IGNORECASE,
)


def _read_static_text(url_path: str) -> str:
    rel = url_path.lstrip("/")
    if rel.startswith("static/"):
        path = STATIC / rel[len("static/") :]
    else:
        path = ROOT / rel
    if not path.is_file():
        raise FileNotFoundError(f"Missing static asset referenced in template: {path}")
    text = path.read_text(encoding="utf-8")
    if "fontawesome" in url_path:
        text = text.replace("url(../webfonts/", "url(/static/vendor/fontawesome/webfonts/")
    return text


def _inline_assets(html: str) -> str:
    def _link_repl(match: re.Match[str]) -> str:
        css = _read_static_text(match.group(1))
        return f"<style>\n{css}\n</style>"

    def _script_repl(match: re.Match[str]) -> str:
        js = _read_static_text(match.group(1))
        return f"<script>\n{js}\n</script>"

    html = LINK_RE.sub(_link_repl, html)
    html = SCRIPT_SRC_RE.sub(_script_repl, html)
    return html


def _build_page(env, template_name: str, out_name: str, active_path: str) -> None:
    template = env.get_template(template_name)
    rendered = template.render(
        version=__version__,
        pagename=pagename_for(template_name),
        active_path=active_path,
        settings={},
        url_for=build_url_for,
    )
    rendered = _inline_assets(rendered)
    out_path = DIST / out_name
    out_path.write_text(rendered, encoding="utf-8")
    print(f"  wrote {out_path.relative_to(ROOT)} ({out_path.stat().st_size // 1024} KiB)")


def main() -> int:
    DIST.mkdir(parents=True, exist_ok=True)
    env = create_jinja_env()
    print(f"Building frontend dist (v{__version__})...")
    for template_name, out_name, active_path in PAGE_SPECS:
        _build_page(env, template_name, out_name, active_path)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
