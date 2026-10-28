"""Index of the saved evaluation reports (``eval/reports/*.json``).

The JSON files written by ``ela-eval`` are the single source of truth,
shared by the command line and the backoffice. This module lists them
(headline metrics only, re-read when a file changes), loads one, deletes
one and builds the file name of a new report. File names are validated
so that a name coming from a URL can never leave the reports directory.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ela.eval.compare import HEADLINE, report_kind, report_metrics

logger = logging.getLogger(__name__)

#: Reports directory, relative to the working directory (project root, ``/app`` in Docker).
REPORTS_DIR = Path("eval") / "reports"

LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}\.json$")


class ReportNameError(ValueError):
    """The report name is not an acceptable file name."""


def valid_label(label: str) -> bool:
    """Whether ``label`` may be used in a report file name (letters, digits, ``._-``, 40 chars)."""
    return bool(LABEL_RE.match(label or ""))


def report_name(label: str, kind: str, now: datetime | None = None) -> str:
    """File name of a new report: ``YYYYmmdd-HHMMSS-<label>-<kind>.json``.

    Raises:
        ReportNameError: Invalid label.
    """
    if not valid_label(label):
        raise ReportNameError(f"étiquette invalide « {label} » (lettres, chiffres, . _ -, 40 caractères au plus)")
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{label}-{kind}.json"


def report_path(name: str, directory: Path | None = None) -> Path:
    """Path of a report inside the reports directory.

    Raises:
        ReportNameError: ``name`` is not a plain ``*.json`` file name.
    """
    if not NAME_RE.match(name or "") or "/" in name or "\\" in name or ".." in name:
        raise ReportNameError("nom de rapport invalide")
    return (directory or REPORTS_DIR) / name


@dataclass(frozen=True)
class ReportSummary:
    """Headline of one saved report (listing)."""

    name: str
    kind: str
    label: str
    timestamp: datetime | None
    model: str | None
    provider: str | None
    n_questions: int
    n_failures: int
    headline: tuple[tuple[str, str, float | int | None], ...]  # (key, label, value)
    size: int
    valid: bool = True
    error: str | None = None

    @property
    def stem(self) -> str:
        """Name without ``.json``."""
        return self.name[:-5] if self.name.endswith(".json") else self.name


def _parse_timestamp(report: dict[str, Any], name: str, mtime: float) -> datetime | None:
    raw = report.get("timestamp")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
    m = re.match(r"^(\d{8})-(\d{6})", name)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return datetime.fromtimestamp(mtime)


def count_failures(kind: str, report: dict[str, Any]) -> int:
    """Questions that deserve a look (misses, errors, phantom citations, low fidelity, failed runs)."""
    questions = report.get("questions") or []
    if kind == "retrieval":
        return sum(1 for q in questions if (q.get("found") or 0) == 0)
    if kind == "code":
        return sum(1 for q in questions if q.get("error") or not q.get("run_ok"))
    return sum(
        1
        for q in questions
        if q.get("error") or q.get("phantom") or (q.get("fidelity") is not None and q["fidelity"] < 0.5)
    )


def summarize_report(name: str, report: dict[str, Any], *, size: int = 0, mtime: float = 0.0) -> ReportSummary:
    """Build the listing entry of one parsed report."""
    kind = report_kind(report)
    metrics = report_metrics(report)
    return ReportSummary(
        name=name,
        kind=kind,
        label=str(report.get("label") or "run"),
        timestamp=_parse_timestamp(report, name, mtime),
        model=report.get("model"),
        provider=report.get("provider"),
        n_questions=int(report.get("n_questions") or len(report.get("questions") or [])),
        n_failures=count_failures(kind, report),
        headline=tuple((key, label, metrics.get(key)) for key, label in HEADLINE[kind]),
        size=size,
    )


class ReportIndex:
    """Cached listing of the reports directory.

    Args:
        directory: Reports directory (default :data:`REPORTS_DIR`).
    """

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else REPORTS_DIR
        self._cache: dict[str, tuple[float, int, ReportSummary]] = {}
        self._lock = threading.Lock()

    def writable(self) -> bool:
        """Whether reports can be written (directory created when missing)."""
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        return os.access(self.directory, os.W_OK)

    def list(self, kind: str | None = None) -> list[ReportSummary]:
        """List the reports, newest first (unreadable files are listed as invalid).

        Args:
            kind: Keep only this kind.
        """
        if not self.directory.is_dir():
            return []
        out: list[ReportSummary] = []
        with self._lock:
            seen: set[str] = set()
            for path in self.directory.glob("*.json"):
                if not NAME_RE.match(path.name):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                seen.add(path.name)
                cached = self._cache.get(path.name)
                if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
                    out.append(cached[2])
                    continue
                summary = self._read_summary(path, stat.st_size, stat.st_mtime)
                self._cache[path.name] = (stat.st_mtime, stat.st_size, summary)
                out.append(summary)
            for stale in [n for n in self._cache if n not in seen]:
                del self._cache[stale]
        if kind:
            out = [s for s in out if s.kind == kind]
        return sorted(out, key=lambda s: (s.valid, s.timestamp or datetime.min, s.name), reverse=True)

    @staticmethod
    def _read_summary(path: Path, size: int, mtime: float) -> ReportSummary:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise ValueError("JSON racine inattendu")
            return summarize_report(path.name, report, size=size, mtime=mtime)
        except Exception as exc:  # corrupt or foreign file: shown, never fatal
            logger.warning("Unreadable report %s: %s", path.name, exc)
            return ReportSummary(
                name=path.name,
                kind="?",
                label=path.stem,
                timestamp=datetime.fromtimestamp(mtime),
                model=None,
                provider=None,
                n_questions=0,
                n_failures=0,
                headline=(),
                size=size,
                valid=False,
                error=str(exc),
            )

    def get(self, name: str) -> ReportSummary | None:
        """Return the listing entry of one report, or ``None``."""
        try:
            path = report_path(name, self.directory)
        except ReportNameError:
            return None
        if not path.is_file():
            return None
        for s in self.list():
            if s.name == name:
                return s
        return None

    def load(self, name: str) -> dict[str, Any] | None:
        """Return the parsed JSON of one report, or ``None`` when missing / invalid."""
        try:
            path = report_path(name, self.directory)
        except ReportNameError:
            return None
        if not path.is_file():
            return None
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return report if isinstance(report, dict) else None

    def path(self, name: str) -> Path | None:
        """Path of an existing report, or ``None``."""
        try:
            path = report_path(name, self.directory)
        except ReportNameError:
            return None
        return path if path.is_file() else None

    def delete(self, name: str) -> bool:
        """Delete one report; ``False`` when it does not exist."""
        path = self.path(name)
        if path is None:
            return False
        path.unlink()
        with self._lock:
            self._cache.pop(name, None)
        return True
