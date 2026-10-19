"""HTTP front of the sandbox: ``POST /run`` runs a Python snippet, ``GET /health`` answers ``ok``.

Standard library only (no web framework to keep the image small). The
service is meant to live on an internal Docker network and to be called by
the ELA application only; it carries no authentication of its own.

Request body (JSON): ``{"code": "...", "language": "python", "timeout": 30}``.
Response: :meth:`runner.RunOutcome.to_dict`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(__file__))
from runner import RunLimits, run_python  # noqa: E402

logger = logging.getLogger("ela.sandbox")

MAX_CODE_CHARS = int(os.environ.get("SANDBOX_MAX_CODE_CHARS", "60000"))
MAX_TIMEOUT = float(os.environ.get("SANDBOX_MAX_TIMEOUT", "60"))
DEFAULT_LIMITS = RunLimits(
    timeout_seconds=float(os.environ.get("SANDBOX_TIMEOUT", "30")),
    cpu_seconds=int(os.environ.get("SANDBOX_CPU_SECONDS", "20")),
    memory_bytes=int(os.environ.get("SANDBOX_MEMORY_MB", "512")) * 1024 * 1024,
)


class Handler(BaseHTTPRequestHandler):
    """One request = one run."""

    server_version = "ela-sandbox/1.0"

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/run":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError):
            self._json(400, {"error": "invalid JSON"})
            return
        code = payload.get("code")
        if not isinstance(code, str) or not code.strip():
            self._json(400, {"error": "missing code"})
            return
        if len(code) > MAX_CODE_CHARS:
            self._json(413, {"error": f"code longer than {MAX_CODE_CHARS} characters"})
            return
        if (payload.get("language") or "python").lower() != "python":
            self._json(400, {"error": "only python is supported"})
            return
        limits = RunLimits(
            timeout_seconds=min(float(payload.get("timeout") or DEFAULT_LIMITS.timeout_seconds), MAX_TIMEOUT),
            cpu_seconds=DEFAULT_LIMITS.cpu_seconds,
            memory_bytes=DEFAULT_LIMITS.memory_bytes,
        )
        outcome = run_python(code, limits)
        logger.info("run status=%s elapsed=%dms images=%d", outcome.status, outcome.elapsed_ms, len(outcome.images))
        self._json(200, outcome.to_dict())

    def log_message(self, fmt, *args) -> None:  # quieter default logging
        logger.debug(fmt, *args)


def main() -> None:
    """Serve on ``SANDBOX_HOST:SANDBOX_PORT`` (default ``0.0.0.0:8080``)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    host = os.environ.get("SANDBOX_HOST", "0.0.0.0")
    port = int(os.environ.get("SANDBOX_PORT", "8080"))
    server = ThreadingHTTPServer((host, port), Handler)
    logger.info("ELA sandbox listening on %s:%d", host, port)
    server.serve_forever()


if __name__ == "__main__":
    main()
