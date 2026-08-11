"""Optional HTTPS dev mirror for LAN/Quest browsers (getUserMedia needs secure context)."""

from __future__ import annotations

import ssl
import threading
from pathlib import Path

from werkzeug.serving import make_server


def default_cert_paths(repo_root: Path) -> tuple[Path, Path]:
    cert_dir = repo_root / ".dev-certs"
    return cert_dir / "dev.crt", cert_dir / "dev.key"


def start_tls_dev_server(
    app,
    *,
    host: str,
    port: int,
    certfile: Path,
    keyfile: Path,
) -> threading.Thread:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(certfile), str(keyfile))

    def _run() -> None:
        server = make_server(host, int(port), app, ssl_context=ctx, threaded=True)
        print(f"[dev-tls] binding https://{host}:{port}", flush=True)
        server.serve_forever()

    thread = threading.Thread(target=_run, daemon=True, name="dev-tls-server")
    thread.start()
    return thread
