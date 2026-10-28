"""Run one background job at a time in a child process, streaming its output to the ``jobs`` table.

Two kinds of jobs exist: the **ingestion** (``python -I -m ela.ingestion.pipeline``,
lot 5.3) and the **evaluation** (``python -I -m ela.eval.runner answer|retrieval``,
lot 5.5). Both run outside the backoffice process: the embedding model
(bge-m3, ~2 GB of RAM) or the agent are loaded only while the job runs, and
a crash cannot take the web application down. A thread reads the child's
output line by line, updates the progress (``🧠 123/1466 chunks vectorisés``,
``[12/71] UTS-12 …``) and flushes the log to the database about once per
second. The child writes a small JSON summary (``--result``) that becomes
the job's ``result``.
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
from dataclasses import dataclass
from pathlib import Path

from ela.db import jobs as jobs_db

logger = logging.getLogger(__name__)

MODES = ("dry_run", "incremental", "full")
EVAL_KINDS = ("retrieval", "answer")
MODE_LABELS = {
    "dry_run": "Vérification (dry run)",
    "incremental": "Mise à jour incrémentale",
    "full": "Reconstruction complète",
    "retrieval": "Évaluation du retrieval",
    "answer": "Évaluation des réponses",
}
_PROGRESS_RE = re.compile(r"(\d+)/(\d+) chunks vectoris")
_EVAL_PROGRESS_RE = re.compile(r"^\[(\d+)/(\d+)\]")
_FLUSH_SECONDS = 1.0

#: Python flags of every child: isolated, unbuffered, UTF-8 (Windows console is cp1252).
PYTHON_FLAGS = ("-I", "-u", "-X", "utf8")


class JobBusyError(Exception):
    """A job is already running."""


def ingest_command(mode: str, source_dir: str, result_path: str) -> list[str]:
    """Command line of the ingestion child process."""
    cmd = [
        sys.executable,
        *PYTHON_FLAGS,
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


@dataclass(frozen=True)
class EvalOptions:
    """What an evaluation job runs.

    Attributes:
        kind: ``retrieval`` or ``answer``.
        label: Report label (validated by the caller, see :mod:`ela.eval.reports`).
        course: Restrict to one course (``None`` = all).
        qtype: Restrict to one question type.
        limit: First N questions only (0 = all).
        judge: Call the LLM judge (``answer`` only).
        k: Top-k of the retrieval evaluation.
        model: Chat model to evaluate instead of the server's (``None`` = server setting).
        provider: Provider of ``model`` (``None`` = server setting).
        judge_model: Judge model pinned in the child when ``model`` is overridden.
        judge_provider: Provider of the pinned judge.
    """

    kind: str
    label: str
    course: str | None = None
    qtype: str | None = None
    limit: int = 0
    judge: bool = True
    k: int = 5
    model: str | None = None
    provider: str | None = None
    judge_model: str | None = None
    judge_provider: str | None = None

    @property
    def overrides_model(self) -> bool:
        """Whether the child evaluates another model / provider than the server's."""
        return bool(self.model or self.provider)

    def environment(self) -> dict[str, str]:
        """Environment variables added to the child.

        When the model is overridden, the judge is pinned to the server's
        judge (``EVAL_JUDGE_MODEL`` / ``EVAL_JUDGE_PROVIDER``) so that
        fidelities stay comparable, and the fallback model is disabled so
        that a failure of the evaluated model counts as an error rather
        than as an answer of the fallback.
        """
        env: dict[str, str] = {}
        if not self.overrides_model:
            return env
        if self.model:
            env["LLM_MODEL_CHAT"] = self.model
        if self.provider:
            env["LLM_PROVIDER"] = self.provider
        if self.judge_model:
            env["EVAL_JUDGE_MODEL"] = self.judge_model
        if self.judge_provider:
            env["EVAL_JUDGE_PROVIDER"] = self.judge_provider
        env["LLM_MODEL_FALLBACK"] = "none"
        return env

    def summary(self) -> dict[str, object]:
        """Options recorded in the audit log (no secret)."""
        out: dict[str, object] = {"kind": self.kind, "label": self.label}
        if self.course:
            out["course"] = self.course
        if self.qtype:
            out["type"] = self.qtype
        if self.limit:
            out["limit"] = self.limit
        if self.kind == "answer":
            out["judge"] = self.judge
        if self.kind == "retrieval":
            out["k"] = self.k
        if self.model:
            out["model"] = self.model
        if self.provider:
            out["provider"] = self.provider
        return out


def eval_command(options: EvalOptions, result_path: str) -> list[str]:
    """Command line of the evaluation child process."""
    if options.kind not in EVAL_KINDS:
        raise ValueError(f"Unknown evaluation kind {options.kind!r}")
    cmd = [sys.executable, *PYTHON_FLAGS, "-m", "ela.eval.runner", options.kind, "--label", options.label]
    if options.course:
        cmd += ["--course", options.course]
    if options.qtype:
        cmd += ["--type", options.qtype]
    if options.limit:
        cmd += ["--limit", str(int(options.limit))]
    if options.kind == "retrieval":
        cmd += ["--k", str(int(options.k))]
    elif not options.judge:
        cmd.append("--no-judge")
    cmd += ["--result", result_path]
    return cmd


class JobRunner:
    """Start and follow background jobs (one at a time per process).

    Args:
        source_dir: Course tree passed to the ingestion.
        command_factory: Builds the ingestion command (tests substitute a fake script).
        eval_command_factory: Builds the evaluation command (tests substitute a fake script).
        on_finish: Optional callback ``(job_id, status)`` after completion.
    """

    def __init__(
        self,
        source_dir: str,
        *,
        command_factory: Callable[[str, str, str], list[str]] = ingest_command,
        eval_command_factory: Callable[[EvalOptions, str], list[str]] = eval_command,
        on_finish: Callable[[int, str], None] | None = None,
    ) -> None:
        self.source_dir = source_dir
        self._command_factory = command_factory
        self._eval_command_factory = eval_command_factory
        self._on_finish = on_finish
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.current_id: int | None = None
        self.current_kind: str | None = None

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
        return self._start(
            jobs_db.KIND_INGEST,
            mode=mode,
            target=target,
            started_by=started_by,
            build=lambda result_path: self._command_factory(mode, self.source_dir, result_path),
            env={},
        )

    def start_eval(self, options: EvalOptions, *, started_by: str) -> int:
        """Create a job row and start an evaluation in the background.

        Raises:
            JobBusyError: Another job is running.
            ValueError: Unknown evaluation kind.
        """
        if options.kind not in EVAL_KINDS:
            raise ValueError(f"Unknown evaluation kind {options.kind!r}")
        return self._start(
            jobs_db.KIND_EVAL,
            mode=options.kind,
            target=options.label,
            started_by=started_by,
            build=lambda result_path: self._eval_command_factory(options, result_path),
            env=options.environment(),
        )

    def wait(self, timeout: float | None = None) -> None:
        """Block until the current job ends (tests)."""
        if self._thread is not None:
            self._thread.join(timeout)

    # ------------------------------------------------------------------------------------
    def _start(
        self,
        kind: str,
        *,
        mode: str,
        target: str | None,
        started_by: str,
        build: Callable[[str], list[str]],
        env: dict[str, str],
    ) -> int:
        with self._lock:
            if self.busy:
                raise JobBusyError("un job est déjà en cours")
            job_id = jobs_db.create_sync(kind, mode=mode, target=target, started_by=started_by)
            self.current_id = job_id
            self.current_kind = kind
            self._thread = threading.Thread(
                target=self._run, args=(job_id, build, env), name=f"job-{job_id}", daemon=True
            )
            self._thread.start()
        return job_id

    def _run(self, job_id: int, build: Callable[[str], list[str]], extra_env: dict[str, str]) -> None:
        status, result, error = jobs_db.FAILED, {}, None
        try:
            jobs_db.start_sync(job_id)
            with tempfile.TemporaryDirectory(prefix="ela-job-") as tmp:
                result_path = os.path.join(tmp, "result.json")
                cmd = build(result_path)
                shown = ["python", *cmd[1:]] if cmd[0] == sys.executable else cmd
                head = "$ " + " ".join(shown) + "\n"
                if extra_env:
                    head += "  env: " + " ".join(f"{k}={v}" for k, v in sorted(extra_env.items())) + "\n"
                jobs_db.append_log_sync(job_id, head, step="Démarrage")
                # ``-I`` makes the child ignore PYTHON* variables, hence ``-u -X utf8`` in the
                # commands; the variables still cover grandchildren (Windows: cp1252 console)
                env = {
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUTF8": "1",
                    **extra_env,
                }
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
                    error = f"le processus s'est terminé avec le code {code}"
        except Exception as exc:  # the runner must never die silently
            logger.exception("Job %s failed", job_id)
            error = f"{exc.__class__.__name__}: {exc}"
        finally:
            try:
                jobs_db.finish_sync(job_id, status=status, result=result, error=error)
            except Exception:
                logger.exception("Could not close job %s", job_id)
            self.current_id = None
            self.current_kind = None
            if self._on_finish:
                try:
                    self._on_finish(job_id, status)
                except Exception:
                    logger.exception("on_finish failed for job %s", job_id)

    @staticmethod
    def parse_progress(line: str) -> int | None:
        """Percentage carried by an output line, if any (ingestion or evaluation format)."""
        m = _PROGRESS_RE.search(line) or _EVAL_PROGRESS_RE.match(line)
        if m and int(m.group(2)) > 0:
            return min(100, int(100 * int(m.group(1)) / int(m.group(2))))
        return None

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
                pct = self.parse_progress(stripped)
                if pct is not None:
                    progress = pct
            if time.monotonic() - last_flush >= _FLUSH_SECONDS or len(buffer) >= 50:
                jobs_db.append_log_sync(job_id, "".join(buffer), progress=progress, step=step)
                buffer.clear()
                last_flush = time.monotonic()
        if buffer or step:
            jobs_db.append_log_sync(job_id, "".join(buffer), progress=progress, step=step)
