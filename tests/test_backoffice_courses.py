"""Offline tests of the courses / jobs pages of the backoffice (lot 5.3)."""

from __future__ import annotations

import io
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

from ela.courses import state  # noqa: E402
from ela.courses.state import CourseStats, KnowledgeBase  # noqa: E402
from ela.db import jobs as jobs_db  # noqa: E402
from ela.db.jobs import Job  # noqa: E402
from ela.jobs.runner import JobRunner  # noqa: E402
from test_backoffice import PREFIX, FakeDb, login, make_settings  # noqa: E402
from test_jobs_runner import FAKE_CHILD, fake_factory  # noqa: E402


class MemoryJobsDb:
    """In-memory ``jobs`` table serving both the runner (sync) and the pages (async)."""

    def __init__(self) -> None:
        self.rows: dict[int, dict] = {}

    def _job(self, job_id: int) -> Job:
        r = self.rows[job_id]
        return Job(
            id=job_id,
            kind=r["kind"],
            mode=r["mode"],
            target=r["target"],
            status=r["status"],
            started_by=r["by"],
            created_at=r["created"],
            started_at=r.get("started"),
            finished_at=r.get("finished"),
            progress=r["progress"],
            step=r["step"],
            log=r["log"],
            result=r.get("result") or {},
            error=r.get("error"),
        )

    # sync (runner)
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
            "created": datetime.now(UTC),
        }
        return job_id

    def start_sync(self, job_id):
        self.rows[job_id].update(status="running", started=datetime.now(UTC))

    def append_log_sync(self, job_id, chunk, *, progress=None, step=None):
        r = self.rows[job_id]
        r["log"] += chunk
        if progress is not None:
            r["progress"] = progress
        if step is not None:
            r["step"] = step

    def finish_sync(self, job_id, *, status, result=None, error=None):
        self.rows[job_id].update(status=status, result=result or {}, error=error, finished=datetime.now(UTC))

    # async (pages)
    async def get_job(self, job_id):
        return self._job(job_id) if job_id in self.rows else None

    async def list_jobs(self, limit=30, kind=None):
        return [self._job(i) for i in sorted(self.rows, reverse=True)][:limit]

    async def active_job(self):
        for i in sorted(self.rows):
            if self.rows[i]["status"] in ("queued", "running"):
                return self._job(i)
        return None

    async def last_finished(self, kind="ingest", success_only=False):
        for i in sorted(self.rows, reverse=True):
            if self.rows[i]["status"] not in ("queued", "running"):
                return self._job(i)
        return None

    async def mark_interrupted(self):
        return 0


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Fake accounts/audit, in-memory jobs, a course tree and a fake ingestion script."""
    import ela.db.audit as audit_mod
    import ela.db.dashboard as dashboard_mod
    import ela.db.users as users_mod

    db = FakeDb()
    monkeypatch.setattr(users_mod, "authenticate", db.authenticate)
    monkeypatch.setattr(users_mod, "get_profile", db.get_profile)
    monkeypatch.setattr(audit_mod, "record", db.record)
    monkeypatch.setattr(audit_mod, "recent", db.recent)
    monkeypatch.setattr(dashboard_mod, "overview", db.overview_fn)

    jobs = MemoryJobsDb()
    for name in (
        "create_sync",
        "start_sync",
        "append_log_sync",
        "finish_sync",
        "get_job",
        "list_jobs",
        "active_job",
        "last_finished",
        "mark_interrupted",
    ):
        monkeypatch.setattr(jobs_db, name, getattr(jobs, name))

    kb = KnowledgeBase(
        backend="fake",
        total=12,
        fingerprint="f",
        courses=[CourseStats("M1", "Series_Temporelles", 12, {"slide": 12}, {"slides": 12}, datetime(2026, 10, 1))],
        by_level={"M1": 12},
        by_type={"slide": 12},
    )
    monkeypatch.setattr(state, "knowledge_base", lambda store=None, settings=None: kb)

    latex = tmp_path / "latex"
    (latex / "M1" / "Series_Temporelles").mkdir(parents=True)
    (latex / "M1" / "Series_Temporelles" / "main.tex").write_text("\\documentclass{beamer}", encoding="utf-8")
    (latex / "M2" / "Nouveau").mkdir(parents=True)
    script = tmp_path / "fake_ingest.py"
    script.write_text(FAKE_CHILD, encoding="utf-8")

    from backoffice.main import create_app

    app = create_app(make_settings(latex_source_dir=str(latex), annales_source_dir=str(tmp_path / "annales")))
    app.state.jobs = JobRunner(str(latex), command_factory=fake_factory(script))
    return {"app": app, "db": db, "jobs": jobs, "latex": latex, "kb": kb}


@pytest.fixture
async def client(env):
    transport = httpx.ASGITransport(app=env["app"])
    async with httpx.AsyncClient(transport=transport, base_url="http://test", follow_redirects=False) as c:
        yield c


async def _admin(client) -> str:
    assert (await login(client, "admin")).status_code == 303
    return client.cookies["ela_bo_csrf"]


# --- overview ---------------------------------------------------------------------------


async def test_overview(client, env):
    await _admin(client)
    r = await client.get(f"{PREFIX}/courses")
    assert r.status_code == 200
    assert "Series_Temporelles" in r.text and "indexé" in r.text and "Nouveau" in r.text and ">vide<" in r.text
    assert "12" in r.text and "Mise à jour incrémentale" in r.text and "Annales" in r.text
    assert "À traiter" in r.text and "dossier sans fichier .tex" in r.text
    r = await client.get(f"{PREFIX}/courses/partials/job", headers={"HX-Request": "true"})
    assert r.status_code == 200 and "<html" not in r.text and "Aucune ingestion" in r.text


async def test_supervisor_read_only(client, env):
    await login(client, "prof")
    r = await client.get(f"{PREFIX}/courses")
    assert r.status_code == 200 and "Créer un cours" not in r.text and 'value="incremental"' not in r.text
    r = await client.get(f"{PREFIX}/courses/latex/M1/Series_Temporelles")
    assert r.status_code == 200 and "Déposer des fichiers" not in r.text and "main.tex" in r.text
    token = client.cookies["ela_bo_csrf"]
    r = await client.post(f"{PREFIX}/ingest", data={"csrf_token": token, "mode": "dry_run"})
    assert r.status_code == 403
    r = await client.post(
        f"{PREFIX}/courses/new", data={"csrf_token": token, "root": "latex", "level": "M1", "name": "X"}
    )
    assert r.status_code == 403 and not (env["latex"] / "M1" / "X").exists()


# --- courses ----------------------------------------------------------------------------


async def test_create_upload_download_delete(client, env):
    token = await _admin(client)
    r = await client.post(
        f"{PREFIX}/courses/new", data={"csrf_token": token, "root": "latex", "level": "Commun", "name": "Algebre"}
    )
    assert r.status_code == 303 and r.headers["location"].endswith("/courses/latex/Commun/Algebre?ok=created")
    assert (env["latex"] / "Commun" / "Algebre").is_dir() and env["db"].audit[-1][0] == "course.create"
    r = await client.post(
        f"{PREFIX}/courses/new", data={"csrf_token": token, "root": "latex", "level": "Commun", "name": "Algebre"}
    )
    assert r.status_code == 400 and "existe déjà" in r.text
    r = await client.post(
        f"{PREFIX}/courses/new", data={"csrf_token": token, "root": "latex", "level": "L3", "name": "Algebre"}
    )
    assert r.status_code == 400

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Algebre/main.tex", "\\begin{document}")
        z.writestr("Algebre/img/fig.png", "p")
    files = [
        ("files", ("algebre.zip", buf.getvalue(), "application/zip")),
        ("files", ("macros.sty", b"% m", "text/plain")),
    ]
    r = await client.post(f"{PREFIX}/courses/latex/Commun/Algebre/upload", data={"csrf_token": token}, files=files)
    assert r.status_code == 303 and "report=" in r.headers["location"]
    assert (env["latex"] / "Commun" / "Algebre" / "main.tex").exists() and (
        env["latex"] / "Commun" / "Algebre" / "macros.sty"
    ).exists()
    r = await client.get(r.headers["location"])
    assert (
        r.status_code == 200 and "ajouté" in r.text and "img/fig.png" in r.text and "mise à jour incrémentale" in r.text
    )
    assert env["db"].audit[-1][0] == "course.upload" and env["db"].audit[-1][2] == {
        "written": 2,
        "replaced": 0,
        "skipped": 1,
    }

    r = await client.post(
        f"{PREFIX}/courses/latex/Commun/Algebre/upload",
        data={"csrf_token": token},
        files=[("files", ("notes.pdf", b"x", "application/pdf"))],
    )
    assert r.status_code == 400 and "refusé" in r.text

    r = await client.get(f"{PREFIX}/courses/latex/Commun/Algebre/files/main.tex")
    assert r.status_code == 200 and r.text == "\\begin{document}" and "attachment" in r.headers["content-disposition"]
    assert (await client.get(f"{PREFIX}/courses/latex/Commun/Algebre/files/../main.tex")).status_code in (400, 404)

    r = await client.post(
        f"{PREFIX}/courses/latex/Commun/Algebre/delete-file", data={"csrf_token": token, "path": "macros.sty"}
    )
    assert r.status_code == 303 and not (env["latex"] / "Commun" / "Algebre" / "macros.sty").exists()
    r = await client.post(f"{PREFIX}/courses/latex/Commun/Algebre/delete", data={"csrf_token": token})
    assert r.status_code == 303 and r.headers["location"].endswith("/courses?ok=deleted")
    assert not (env["latex"] / "Commun" / "Algebre").exists() and env["db"].audit[-1][0] == "course.delete"
    assert (await client.get(f"{PREFIX}/courses/latex/Commun/Algebre")).status_code == 404


async def test_annales_root(client, env):
    token = await _admin(client)
    r = await client.post(
        f"{PREFIX}/courses/new", data={"csrf_token": token, "root": "annales", "level": "M1", "name": "UTS"}
    )
    assert r.status_code == 303
    r = await client.get(f"{PREFIX}/courses/annales/M1/UTS")
    assert r.status_code == 200 and "Annales" in r.text and "Dans la base de connaissances" not in r.text
    r = await client.post(f"{PREFIX}/courses/annales/M1/UTS/move", data={"csrf_token": token, "to_level": "M2"})
    assert r.status_code == 303 and r.headers["location"].endswith("/courses/annales/M2/UTS?ok=moved")
    assert (await client.post(f"{PREFIX}/courses/nope/M1/UTS/delete", data={"csrf_token": token})).status_code == 400


# --- ingestion jobs ------------------------------------------------------------------------


async def test_ingest_job_lifecycle(client, env):
    token = await _admin(client)
    r = await client.post(f"{PREFIX}/ingest", data={"csrf_token": token, "mode": "weird"})
    assert r.status_code == 400
    r = await client.post(f"{PREFIX}/ingest", data={"csrf_token": token, "mode": "dry_run"})
    assert r.status_code == 303 and r.headers["location"].endswith("/jobs/1")
    assert env["db"].audit[-1][0] == "ingest.start" and env["db"].audit[-1][2] == {"mode": "dry_run"}
    # busy while running
    r = await client.post(f"{PREFIX}/ingest", data={"csrf_token": token, "mode": "dry_run"})
    assert r.status_code == 409
    r = await client.get(f"{PREFIX}/courses/partials/job")
    assert "en cours" in r.text or "en attente" in r.text
    env["app"].state.jobs.wait(20)
    r = await client.get(f"{PREFIX}/jobs/1")
    assert (
        r.status_code == 200 and "réussi" in r.text and "vérification seule" in r.text and "chunks vectorisés" in r.text
    )
    r = await client.get(f"{PREFIX}/jobs/1/partial", headers={"HX-Request": "true"})
    assert r.status_code == 200 and "hx-trigger" not in r.text  # finished: no more polling
    r = await client.get(f"{PREFIX}/jobs")
    assert r.status_code == 200 and "10 chunks analysés" in r.text
    r = await client.get(f"{PREFIX}/courses")
    assert "Dernier job" in r.text and "réussi" in r.text
    assert (await client.get(f"{PREFIX}/jobs/99")).status_code == 404


async def test_move_course_starts_incremental_job(client, env):
    token = await _admin(client)
    r = await client.post(
        f"{PREFIX}/courses/latex/M1/Series_Temporelles/move", data={"csrf_token": token, "to_level": "M2"}
    )
    assert r.status_code == 303 and r.headers["location"].endswith("/jobs/1")
    assert (env["latex"] / "M2" / "Series_Temporelles" / "main.tex").exists()
    assert [a for a, _, _ in env["db"].audit][-2:] == ["course.move", "ingest.start"]
    env["app"].state.jobs.wait(20)
    job = env["jobs"].rows[1]
    assert (
        job["status"] == "success"
        and job["mode"] == "incremental"
        and "déplacement M1/Series_Temporelles → M2" == job["target"]
    )
    r = await client.post(
        f"{PREFIX}/courses/latex/M2/Series_Temporelles/move", data={"csrf_token": token, "to_level": "M2"}
    )
    assert r.status_code == 400 and "déjà dans ce niveau" in r.text


async def test_move_refused_while_job_runs(client, env):
    token = await _admin(client)
    await client.post(f"{PREFIX}/ingest", data={"csrf_token": token, "mode": "incremental"})
    r = await client.post(
        f"{PREFIX}/courses/latex/M1/Series_Temporelles/move", data={"csrf_token": token, "to_level": "M2"}
    )
    assert r.status_code == 409 and (env["latex"] / "M1" / "Series_Temporelles").exists()
    env["app"].state.jobs.wait(20)
