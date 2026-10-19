"""HTTP client of the sandbox service (``apps/sandbox/server.py``).

The service runs student scripts in an isolated container; this module only
sends the code and parses the outcome. It is disabled (``enabled()`` is
``False``) when ``SANDBOX_URL`` is empty, in which case the Atelier Code
keeps generating scripts without running them.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field

import httpx

from ela.config import get_settings

logger = logging.getLogger(__name__)


class SandboxUnavailableError(RuntimeError):
    """The sandbox service cannot be reached or refused the request."""


@dataclass
class RunResult:
    """Outcome of one execution."""

    status: str  # ok | error | timeout
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    elapsed_ms: int = 0
    images: list[bytes] = field(default_factory=list)  # PNG

    @property
    def ok(self) -> bool:
        """Whether the script ended normally."""
        return self.status == "ok"

    @classmethod
    def from_payload(cls, payload: dict) -> RunResult:
        """Build from the service's JSON."""
        images = []
        for b64 in payload.get("images") or []:
            try:
                images.append(base64.b64decode(b64))
            except (ValueError, TypeError):
                logger.warning("Unreadable image in sandbox response")
        return cls(
            status=str(payload.get("status") or "error"),
            stdout=str(payload.get("stdout") or ""),
            stderr=str(payload.get("stderr") or ""),
            exit_code=payload.get("exit_code"),
            elapsed_ms=int(payload.get("elapsed_ms") or 0),
            images=images,
        )


def enabled() -> bool:
    """Whether code execution is configured."""
    return get_settings().sandbox_enabled


async def run_code(
    code: str, *, language: str = "python", transport: httpx.AsyncBaseTransport | None = None
) -> RunResult:
    """Execute ``code`` in the sandbox.

    Args:
        code: Script source.
        language: Only ``python`` is executed today.
        transport: httpx transport override (tests).

    Returns:
        The run result.

    Raises:
        SandboxUnavailableError: Sandbox not configured, unreachable, or refusing the request.
    """
    settings = get_settings()
    if not settings.sandbox_enabled:
        raise SandboxUnavailableError("sandbox not configured (SANDBOX_URL)")
    url = settings.sandbox_url.rstrip("/") + "/run"
    timeout = settings.sandbox_timeout_seconds
    try:
        async with httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(timeout + 15, connect=5)) as client:
            response = await client.post(url, json={"code": code, "language": language.lower(), "timeout": timeout})
    except httpx.HTTPError as exc:
        raise SandboxUnavailableError(f"sandbox unreachable: {exc.__class__.__name__}") from exc
    if response.status_code != 200:
        detail = ""
        try:
            detail = response.json().get("error", "")
        except ValueError:
            pass
        raise SandboxUnavailableError(f"sandbox refused the run ({response.status_code}) {detail}".strip())
    return RunResult.from_payload(response.json())
