"""Job runner (ela.jobs.runner) with a fake child process and an in-memory jobs table."""

from __future__ import annotations

import json
import sys
import textwrap
import time

import pytest

from ela.db import jobs as jobs_db
from ela.jobs import runner as runner_mod
from ela.jobs.runner import JobBusyError, JobRunner, ingest_command

FAKE_CHILD = textwrap.dedent(
    """
    import json, sys, time
    mode, result_path = sys.argv[1], sys.argv[2]
    print("📂 Scan récursif ...")
    for i in (5, 10):
        time.sleep(0.05)
        print(f"   🧠 {i}/10 chunks vectorisés (99.0/s)")
    if mode == "full":
        print("❌ boom")
        sys.exit(1)
    json.dump({"parsed": 10, "added": 10, "unchanged": 0, "removed": 0, "total": 10, "seconds": 0.1,
               "dry_run": mode == "dry_run", "backend": "fake", "stats": {}, "by_type": {}, "warnings": []},
              open(result_path, "w"))
    print("🎉 Ingestion terminée")
    """
)


class MemoryJobs:
    """Stand-in for the sync writes of ``ela.db.jobs``."""

    def __init__(self) -> None:
        self.rows: dict[int, dict] = {}

    def create_sync(self, kind, *, mode, target, started_by):
        job_id = len(self.rows) + 1
        self.rows[job_id] = {
            "kind": kind,
            "mode": mode,
            "target": target,
            "by": started_by,
            "status": "queued",
            "log": "",
            "progress": 0,
            "step": None,
        }
        return job_id

    def start_sync(self, job_id):
        self.rows[job_id]["status"] = "running"

    def append_log_sync(self, job_id, chunk, *, progress=None, step=None):
        row = self.rows[job_id]
        row["log"] += chunk
        if progress is not None:
            row["progress"] = progress
        if step is not None:
            row["step"] = step

    def finish_sync(self, job_id, *, status, result=None, error=None):
        self.rows[job_id].update(status=status, result=result or {}, error=error)


@pytest.fixture
def memory(monkeypatch: pytest.MonkeyPatch, tmp_path) -> MemoryJobs:
    mem = MemoryJobs()
    for name in ("create_sync", "start_sync", "append_log_sync", "finish_sync"):
        monkeypatch.setattr(jobs_db, name, getattr(mem, name))
    script = tmp_path / "fake_ingest.py"
    script.write_text(FAKE_CHILD, encoding="utf-8")
    mem.script = script  # type: ignore[attr-defined]
    return mem


def fake_factory(script):
    def factory(mode, source_dir, result_path):
        # ``-X utf8`` as in ``ingest_command``: ``-I`` ignores PYTHONIOENCODING, and the
        # Windows console is cp1252 (the fake child prints emojis like the real pipeline)
        return [sys.executable, "-I", "-u", "-X", "utf8", str(script), mode, result_path]

    return factory


def test_ingest_command():
    cmd = ingest_command("dry_run", "/data/latex", "/tmp/r.json")
    assert cmd[:5] == [sys.executable, "-I", "-u", "-X", "utf8"] and "ela.ingestion.pipeline" in cmd
    assert cmd[-1] == "--dry-run"
    assert "--source-dir" in cmd and "/data/latex" in cmd and "--result" in cmd
    assert ingest_command("full", "/d", "/r")[-1] == "--full" and "--dry-run" not in ingest_command(
        "incremental", "/d", "/r"
    )


def test_runner_success_and_progress(memory):
    finished = []
    r = JobRunner(
        "/data/latex", command_factory=fake_factory(memory.script), on_finish=lambda i, s: finished.append((i, s))
    )
    job_id = r.start_ingest("incremental", started_by="admin", target="test")
    assert job_id == 1 and r.busy
    r.wait(20)
    row = memory.rows[1]
    assert row["status"] == "success" and row["result"]["total"] == 10 and row["error"] is None
    assert row["progress"] == 100 and "chunks vectorisés" in row["log"] and row["log"].startswith("$ python -I ")
    assert "Ingestion terminée" in row["step"] and finished == [(1, "success")]
    assert not r.busy and r.current_id is None


def test_runner_failure_and_busy(memory):
    r = JobRunner("/data/latex", command_factory=fake_factory(memory.script))
    job_id = r.start_ingest("full", started_by="admin")
    with pytest.raises(JobBusyError):
        r.start_ingest("dry_run", started_by="admin")
    r.wait(20)
    row = memory.rows[job_id]
    assert row["status"] == "failed" and "code 1" in row["error"] and "boom" in row["log"]
    with pytest.raises(ValueError):
        r.start_ingest("weird", started_by="admin")
    # a second job can start once the first is over
    assert r.start_ingest("dry_run", started_by="admin") == 2
    r.wait(20)
    assert memory.rows[2]["status"] == "success" and memory.rows[2]["result"]["dry_run"] is True


def test_runner_survives_missing_command(memory):
    r = JobRunner("/data/latex", command_factory=lambda m, s, p: ["/nonexistent/binary"])
    job_id = r.start_ingest("dry_run", started_by="admin")
    r.wait(10)
    assert memory.rows[job_id]["status"] == "failed" and "FileNotFoundError" in memory.rows[job_id]["error"]


def test_pump_flushes_in_batches(memory, monkeypatch):
    monkeypatch.setattr(runner_mod, "_FLUSH_SECONDS", 0.0)
    r = JobRunner("/data/latex", command_factory=fake_factory(memory.script))
    r.start_ingest("incremental", started_by="admin")
    r.wait(20)
    assert memory.rows[1]["log"].count("\n") >= 4
    assert json.loads(json.dumps(memory.rows[1]["result"]))["parsed"] == 10
    time.sleep(0)  # keep the import of time meaningful for linters
