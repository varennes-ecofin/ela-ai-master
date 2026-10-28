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


# --- lot 5.5 : evaluation jobs -------------------------------------------------------------

FAKE_EVAL_CHILD = textwrap.dedent(
    """
    import json, os, sys
    kind, result_path = sys.argv[1], sys.argv[2]
    n = 4
    print(f"{n} questions, juge fixe")
    for i in range(1, n + 1):
        print(f"[{i}/{n}] Q-{i:02d}   0.50s  recherches=1  fidélité=1.00")
    env = {k: os.environ.get(k) for k in ("LLM_MODEL_CHAT", "LLM_PROVIDER", "EVAL_JUDGE_MODEL",
                                           "EVAL_JUDGE_PROVIDER", "LLM_MODEL_FALLBACK")}
    json.dump({"report": f"20261027-120000-test-{kind}.json", "kind": kind, "label": "test",
               "n_questions": n, "n_errors": 0, "metrics": {"fidelity_mean": 0.9}, "env": env},
              open(result_path, "w"))
    print("Rapport : eval/reports/x.json")
    """
)


def fake_eval_factory(script):
    def factory(options, result_path):
        return [sys.executable, "-I", "-u", "-X", "utf8", str(script), options.kind, result_path]

    return factory


@pytest.fixture
def eval_script(tmp_path):
    script = tmp_path / "fake_eval.py"
    script.write_text(FAKE_EVAL_CHILD, encoding="utf-8")
    return script


def test_eval_command_and_options():
    from ela.jobs.runner import EvalOptions, eval_command

    opts = EvalOptions(kind="answer", label="agent", course="UTS", qtype="multi-hop", limit=20, judge=False)
    cmd = eval_command(opts, "/tmp/r.json")
    assert cmd[:5] == [sys.executable, "-I", "-u", "-X", "utf8"] and cmd[5:8] == ["-m", "ela.eval.runner", "answer"]
    assert "--label" in cmd and "agent" in cmd and "--course" in cmd and "UTS" in cmd
    assert "--type" in cmd and "multi-hop" in cmd and "--limit" in cmd and "20" in cmd
    assert "--no-judge" in cmd and "--k" not in cmd and cmd[-2:] == ["--result", "/tmp/r.json"]
    ret = eval_command(EvalOptions(kind="retrieval", label="x", k=10), "/r")
    assert "--k" in ret and "10" in ret and "--no-judge" not in ret and "--limit" not in ret
    with pytest.raises(ValueError):
        eval_command(EvalOptions(kind="weird", label="x"), "/r")
    # no override: empty environment; override: model, pinned judge, fallback disabled
    assert opts.environment() == {} and not opts.overrides_model
    over = EvalOptions(
        kind="answer",
        label="q",
        model="qwen-3.8-27b",
        provider="ilaas",
        judge_model="gpt-oss-120b",
        judge_provider="cerebras",
    )
    assert over.environment() == {
        "LLM_MODEL_CHAT": "qwen-3.8-27b",
        "LLM_PROVIDER": "ilaas",
        "EVAL_JUDGE_MODEL": "gpt-oss-120b",
        "EVAL_JUDGE_PROVIDER": "cerebras",
        "LLM_MODEL_FALLBACK": "none",
    }
    assert over.summary() == {
        "kind": "answer",
        "label": "q",
        "judge": True,
        "model": "qwen-3.8-27b",
        "provider": "ilaas",
    }
    assert EvalOptions(kind="retrieval", label="r", k=7, limit=3).summary() == {
        "kind": "retrieval",
        "label": "r",
        "limit": 3,
        "k": 7,
    }


def test_runner_eval_job_progress_result_and_environment(memory, eval_script, monkeypatch):
    from ela.jobs.runner import EvalOptions

    monkeypatch.setenv("LLM_MODEL_CHAT", "server-model")
    monkeypatch.delenv("LLM_MODEL_FALLBACK", raising=False)
    r = JobRunner("/data/latex", eval_command_factory=fake_eval_factory(eval_script))
    opts = EvalOptions(
        kind="answer",
        label="test",
        model="other-model",
        provider="ilaas",
        judge_model="judge-m",
        judge_provider="cerebras",
    )
    job_id = r.start_eval(opts, started_by="admin")
    assert r.busy and r.current_kind == jobs_db.KIND_EVAL
    with pytest.raises(JobBusyError):
        r.start_ingest("dry_run", started_by="admin")
    r.wait(20)
    row = memory.rows[job_id]
    assert row["kind"] == "eval" and row["mode"] == "answer" and row["target"] == "test"
    assert row["status"] == "success" and row["progress"] == 100 and "[4/4] Q-04" in row["log"]
    assert row["result"]["report"] == "20261027-120000-test-answer.json" and row["result"]["metrics"] == {
        "fidelity_mean": 0.9
    }
    assert row["result"]["env"] == {
        "LLM_MODEL_CHAT": "other-model",
        "LLM_PROVIDER": "ilaas",
        "EVAL_JUDGE_MODEL": "judge-m",
        "EVAL_JUDGE_PROVIDER": "cerebras",
        "LLM_MODEL_FALLBACK": "none",
    }
    assert "env: EVAL_JUDGE_MODEL=judge-m" in row["log"] and "LLM_MODEL_FALLBACK=none" in row["log"]
    assert r.current_kind is None
    # without override the child inherits the server's environment untouched
    job2 = r.start_eval(EvalOptions(kind="retrieval", label="plain"), started_by="admin")
    r.wait(20)
    env = memory.rows[job2]["result"]["env"]
    assert env["LLM_MODEL_CHAT"] == "server-model" and env["LLM_MODEL_FALLBACK"] is None
    assert "env:" not in memory.rows[job2]["log"]
    with pytest.raises(ValueError):
        r.start_eval(EvalOptions(kind="weird", label="x"), started_by="admin")


def test_parse_progress_formats():
    assert JobRunner.parse_progress("   🧠 5/10 chunks vectorisés (99.0/s)") == 50
    assert JobRunner.parse_progress("[12/71] UTS-12   0.5s") == 16
    assert JobRunner.parse_progress("[71/71] NCE-05") == 100
    assert JobRunner.parse_progress("[0/0] x") is None and JobRunner.parse_progress("hello") is None
