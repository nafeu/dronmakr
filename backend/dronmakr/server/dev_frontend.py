"""Live Jinja template serving and auto-reload for local frontend development."""

from __future__ import annotations

import hashlib
import sys
import threading

from flask import url_for

from dronmakr._repo import ASSETS_ROOT
from dronmakr.core.utils import with_main_prompt as with_prompt
from dronmakr.server.frontend_pages import (
    DIST_FILENAME_TO_SPEC,
    create_jinja_env,
    pagename_for,
)
from dronmakr.version import __version__

WATCH_DIRS = (ASSETS_ROOT / "templates", ASSETS_ROOT / "static")

RELOAD_SCRIPT = """
<script id="dronmakr-dev-reload">
(function () {
  var token = null;
  setInterval(function () {
    fetch("/dev/reload-check", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (token !== null && data.version !== token) {
          location.reload();
        }
        token = data.version;
      })
      .catch(function () {});
  }, 800);
})();
</script>
"""


def _inject_reload_script(html: str) -> str:
    if "</body>" in html:
        return html.replace("</body>", RELOAD_SCRIPT + "\n</body>", 1)
    return html + RELOAD_SCRIPT


def _assets_version_token() -> str:
    digest = hashlib.sha256()
    for watch_dir in WATCH_DIRS:
        if not watch_dir.is_dir():
            continue
        for path in sorted(watch_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(ASSETS_ROOT).as_posix()
            digest.update(rel.encode())
            digest.update(str(path.stat().st_mtime_ns).encode())
    return digest.hexdigest()[:16]


class DevFrontend:
    """Render templates from assets/ and track file changes for client reload."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._version = _assets_version_token()
        self._stop = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._jinja = create_jinja_env(auto_reload=True)

    def start(self) -> None:
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name="dev-frontend-watcher",
            daemon=True,
        )
        self._poll_thread.start()

    def stop(self) -> None:
        self._stop.set()

    def current_version(self) -> str:
        with self._lock:
            return self._version

    def render_page(self, dist_filename: str) -> str:
        spec = DIST_FILENAME_TO_SPEC.get(dist_filename)
        if spec is None:
            raise KeyError(dist_filename)
        template_name, active_path = spec
        template = self._jinja.get_template(template_name)
        html = template.render(
            version=__version__,
            pagename=pagename_for(template_name),
            active_path=active_path,
            settings={},
            url_for=url_for,
        )
        return _inject_reload_script(html)

    def _poll_loop(self) -> None:
        while not self._stop.wait(0.5):
            new_version = _assets_version_token()
            with self._lock:
                if new_version == self._version:
                    continue
                self._version = new_version
            print(
                with_prompt(
                    f"[dev-frontend] assets changed — reload token {new_version}"
                ),
                flush=True,
            )


_dev_frontend: DevFrontend | None = None


def dev_frontend_enabled() -> bool:
    return _dev_frontend is not None


def get_dev_frontend() -> DevFrontend | None:
    return _dev_frontend


def enable_dev_frontend() -> bool:
    """Turn on live template serving. Returns False when unavailable (frozen builds)."""
    global _dev_frontend
    if getattr(sys, "frozen", False):
        print(
            with_prompt(
                "[dev-frontend] ignored: not available in bundled backend builds"
            ),
            flush=True,
        )
        return False
    if _dev_frontend is not None:
        return True

    _dev_frontend = DevFrontend()
    _dev_frontend.start()
    print(
        with_prompt(
            "[dev-frontend] serving live templates from assets/ (auto-reload enabled)"
        ),
        flush=True,
    )
    return True
