"""Run one untrusted Python snippet in a throw-away subprocess with hard limits.

The runner is the heart of the sandbox service (``server.py``). It never
trusts the code: the snippet runs under ``python -I`` (isolated mode, no
user site, no environment hooks) in a fresh temporary directory, with CPU
time, memory, process count and file size capped by ``resource.setrlimit``,
and a wall-clock timeout. Network isolation is provided by the container
(``docker-compose.yml``: internal network only), not by this module. Run
outside Docker (``python apps/sandbox/server.py`` on a developer machine) the
runner only limits time and memory: use it for your own scripts, never for
students.

Matplotlib figures are captured: a preamble forces the ``Agg`` backend and an
epilogue saves every open figure as PNG in the working directory; the PNGs
are returned base64-encoded.
"""

from __future__ import annotations

import base64
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

try:  # POSIX only: on Windows (local tests) the wall-clock timeout is the only limit
    import resource
except ImportError:  # pragma: no cover
    resource = None

MAX_OUTPUT_CHARS = 20_000
MAX_IMAGES = 6
MAX_IMAGE_BYTES = 2_000_000

PREAMBLE = """\
import os as _os, sys as _sys
for _stream in (_sys.stdout, _sys.stderr):  # UTF-8 whatever the host console (Windows: cp1252)
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
_os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib as _mpl
    _mpl.use("Agg")
except Exception:
    pass
"""

EPILOGUE = """

# --- ELA sandbox: save open figures ---
try:
    import matplotlib.pyplot as _plt
    for _i, _num in enumerate(_plt.get_fignums(), start=1):
        _plt.figure(_num).savefig(f"_ela_fig_{_i:02d}.png", dpi=110, bbox_inches="tight")
except Exception:
    pass
"""


@dataclass
class RunLimits:
    """Resource limits of one run."""

    timeout_seconds: float = 30.0
    cpu_seconds: int = 20
    memory_bytes: int = 512 * 1024 * 1024
    max_processes: int = 32
    max_file_bytes: int = 20 * 1024 * 1024


@dataclass
class RunOutcome:
    """Result of one run."""

    status: str  # ok | error | timeout
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    elapsed_ms: int = 0
    images: list[str] = field(default_factory=list)  # base64 PNG

    def to_dict(self) -> dict:
        """JSON-serialisable form."""
        return {
            "status": self.status,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "elapsed_ms": self.elapsed_ms,
            "images": self.images,
        }


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… (sortie tronquée à {limit} caractères)"


def _limiter(limits: RunLimits):
    if resource is None:
        return None

    def _apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds + 2))
        resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes))
        resource.setrlimit(resource.RLIMIT_NPROC, (limits.max_processes, limits.max_processes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.max_file_bytes, limits.max_file_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.setsid()

    return _apply


def run_python(code: str, limits: RunLimits | None = None, *, python: str | None = None) -> RunOutcome:
    """Execute ``code`` and collect its output and figures.

    Args:
        code: Python source.
        limits: Resource limits (defaults).
        python: Interpreter (defaults to the current one).

    Returns:
        The outcome (never raises for a failing snippet).
    """
    limits = limits or RunLimits()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ela-run-") as workdir:
        script = os.path.join(workdir, "main.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(PREAMBLE + code + EPILOGUE)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": workdir,
            "MPLCONFIGDIR": os.path.join(workdir, ".mpl"),
            "MPLBACKEND": "Agg",
            "PYTHONIOENCODING": "utf-8",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "LANG": "C.UTF-8",
        }
        try:
            proc = subprocess.run(
                # -I ignores PYTHON* variables, so UTF-8 mode is requested on the command line
                [python or sys.executable, "-I", "-B", "-X", "utf8", script],
                cwd=workdir,
                env=env,
                capture_output=True,
                timeout=limits.timeout_seconds,
                preexec_fn=_limiter(limits),
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = int((time.monotonic() - started) * 1000)
            out = (exc.stdout or b"").decode("utf-8", "replace")
            err = (exc.stderr or b"").decode("utf-8", "replace")
            return RunOutcome(
                "timeout",
                _truncate(out),
                _truncate(err + f"\n⏱️ Exécution interrompue après {limits.timeout_seconds:.0f} s."),
                None,
                elapsed,
            )
        elapsed = int((time.monotonic() - started) * 1000)
        images = _collect_images(workdir)
        stdout = proc.stdout.decode("utf-8", "replace")
        stderr = proc.stderr.decode("utf-8", "replace")
        killed = {-getattr(signal, "SIGXCPU", 24), -getattr(signal, "SIGKILL", 9)}
        if proc.returncode in killed:  # CPU limit or memory kill
            stderr += f"\n⏱️ Exécution interrompue : temps CPU ({limits.cpu_seconds} s) ou mémoire dépassés."
            return RunOutcome("timeout", _truncate(stdout), _truncate(stderr), proc.returncode, elapsed, images)
        status = "ok" if proc.returncode == 0 else "error"
        return RunOutcome(status, _truncate(stdout), _truncate(stderr), proc.returncode, elapsed, images)


def _collect_images(workdir: str) -> list[str]:
    names = sorted(n for n in os.listdir(workdir) if n.startswith("_ela_fig_") and n.endswith(".png"))
    images: list[str] = []
    for name in names[:MAX_IMAGES]:
        path = os.path.join(workdir, name)
        if os.path.getsize(path) > MAX_IMAGE_BYTES:
            continue
        with open(path, "rb") as fh:
            images.append(base64.b64encode(fh.read()).decode("ascii"))
    return images
