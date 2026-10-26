"""Run one ingestion at a time in a child process, streaming its output to the ``jobs`` table.

The ingestion runs as ``python -I -m ela.ingestion.pipeline`` rather than in
the backoffice process: the embedding model (bge-m3, ~2 GB of RAM) is loaded
only while the job runs and released afterwards, and a crash in the parser
cannot take the web application down. A thread reads the child's output
line by line, updates the progress (``🧠 123/1466 chunks vectorisés``) and
flushes the log to the database about once per second.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from ela.db import jobs as jobs_db

logger = logging.getLogger(__name__)

MODES = ("dry_run", "incremental", "full")
MODE_LABELS = {
    "dry_run": "Vérification (dry run)",
    "incremental": "Mise à jour incrémentale",
    "full": "Reconstruction complète",
}
_PROGRESS_RE = re.compile(r"(\d+)/(\d+) chunks vectoris")
_FLUSH_SECONDS = 1.0


class JobBusyError(Exception):
    """A job is already running."""


def ingest_command(mode: str, source_dir: str, result_path: str) -> list[str]:
    """Command line of the ingestion child process."""
    cmd = [
        sys.executable,
        "-I",
        "-u",
        "-X",
        "utf8",
        "-m",
        "ela.ingestion.pipeline",
        "--source-dir",
        source_dir,
        "--result",
        result_path,
    ]
    if mode == "dry_run":
        cmd.append("--dry-run")
    elif mode == "full":
        cmd.append("--full")
    return cmd


class JobRunner:
    """Start and follow background jobs (one at a time per process).

    Args:
        source_dir: Course tree passed to the ingestion.
        command_factory: Builds the child command (tests substitute a fake script).
        on_finish: Optional callback ``(job_id, status)`` after completion.
    """

    def __init__(
        self,
        source_dir: str,
        *,
        command_factory: Callable[[str, str, str], list[str]] = ingest_command,
        on_finish: Callable[[int, str], None] | None = None,
    ) -> None:
        self.source_dir = source_dir
        self._command_factory = command_factory
        self._on_finish = on_finish
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.current_id: int | None = None

    @property
    def busy(self) -> bool:
        """Whether a job is running in this process."""
        return self._thread is not None and self._thread.is_alive()

    def start_ingest(self, mode: str, *, started_by: str, target: str | None = None) -> int:
        """Create a job row and start the ingestion in the background.

        Raises:
            JobBusyError: Another job is running.
            ValueError: Unknown mode.
        """
        if mode not in MODES:
            raise ValueError(f"Unknown ingestion mode {mode!r}")
        with self._lock:
            if self.busy:
                raise JobBusyError("une ingestion est déjà en cours")
            job_id = jobs_db.create_sync(jobs_db.KIND_INGEST, mode=mode, target=target, started_by=started_by)
            self.current_id = job_id
            self._thread = threading.Thread(target=self._run, args=(job_id, mode), name=f"job-{job_id}", daemon=True)
            self._thread.start()
        return job_id

    def wait(self, timeout: float | None = None) -> None:
        """Block until the current job ends (tests)."""
        if self._thread is not None:
            self._thread.join(timeout)

    # ------------------------------------------------------------------------------------
    def _run(self, job_id: int, mode: str) -> None:
        status, result, error = jobs_db.FAILED, {}, None
        try:
            jobs_db.start_sync(job_id)
            with tempfile.TemporaryDirectory(prefix="ela-job-") as tmp:
                result_path = os.path.join(tmp, "result.json")
                cmd = self._command_factory(mode, self.source_dir, result_path)
                shown = ["python", *cmd[1:]] if cmd[0] == sys.executable else cmd
                jobs_db.append_log_sync(job_id, "$ " + " ".join(shown) + "\n", step="Démarrage")
                # ``-I`` makes the child ignore PYTHON* variables, hence ``-u -X utf8`` in
                # ``ingest_command``; the variables still cover grandchildren (Windows: cp1252 console)
                env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )
                self._pump(job_id, proc)
                code = proc.wait()
                if Path(result_path).is_file():
                    with open(result_path, encoding="utf-8") as f:
                        result = json.load(f)
                if code == 0:
                    status = jobs_db.SUCCESS
                else:
                    error = f"le processus d'ingestion s'est terminé avec le code {code}"
        except Exception as exc:  # the runner must never die silently
            logger.exception("Job %s failed", job_id)
            error = f"{exc.__class__.__name__}: {exc}"
        finally:
            try:
                jobs_db.finish_sync(job_id, status=status, result=result, error=error)
            except Exception:
                logger.exception("Could not close job %s", job_id)
            self.current_id = None
            if self._on_finish:
                try:
                    self._on_finish(job_id, status)
                except Exception:
                    logger.exception("on_finish failed for job %s", job_id)

    def _pump(self, job_id: int, proc: subprocess.Popen) -> None:
        """Forward the child's output to the job row, batching writes."""
        buffer: list[str] = []
        last_flush = time.monotonic()
        progress: int | None = None
        step: str | None = None
        assert proc.stdout is not None
        for line in proc.stdout:
            buffer.append(line)
            stripped = line.strip()
            if stripped:
                step = stripped[:120]
                m = _PROGRESS_RE.search(stripped)
                if m and int(m.group(2)) > 0:
                    progress = int(100 * int(m.group(1)) / int(m.group(2)))
            if time.monotonic() - last_flush >= _FLUSH_SECONDS or len(buffer) >= 50:
                jobs_db.append_log_sync(job_id, "".join(buffer), progress=progress, step=step)
                buffer.clear()
                last_flush = time.monotonic()
        if buffer or step:
            jobs_db.append_log_sync(job_id, "".join(buffer), progress=progress, step=step)
