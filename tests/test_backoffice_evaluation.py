"""Offline tests of the evaluation pages of the backoffice (lot 5.5a)."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps"))

from ela.db import jobs as jobs_db  # noqa: E402
from test_backoffice import PREFIX, FakeDb, login, make_settings  # noqa: E402
from test_backoffice_courses import MemoryJobsDb  # noqa: E402
from test_jobs_runner import FAKE_EVAL_CHILD, fake_eval_factory  # noqa: E402

DATASET = textwrap.dedent(
    """
    version: 3
    questions:
    - id: UTS-01
      level: M1
      course: Univariate_Time_Series
      type: definition
      difficulty: 1
      question: "Stationnarité faible ?"
      expected_sources: [{source: UTS.tex, slide_title: "Stationnarité"}]
      answer_points: ["moments constants"]
    - id: UTS-02
      level: M1
      course: Univariate_Time_Series
      type: multi-hop
      difficulty: 3
      question: "ADF puis PP ?"
      expected_sources: [{source: UTS.tex, slide_title: "ADF"}]
      answer_points: ["ADF", "PP"]
    - id: MTS-01
      level: M2
      course: Multivariate_Time_Series
      type: formula
      difficulty: 2
      question: "VAR(1) ?"
      expected_sources: [{source: MTS.tex, slide_title: "VAR"}]
      answer_points: ["A1"]
    - id: OOD-01
      level: M1
      course: Univariate_Time_Series
      type: out-of-domain
      difficulty: 1
      question: "Recette de la tarte ?"
    """
)

RETRIEVAL_REPORT = {
    "kind": "retrieval",
    "k": 5,
    "n_questions": 2,
    "hit_at_k": 0.5,
    "any_hit_at_k": 0.5,
    "mrr": 0.5,
    "any_hit_candidates": 1.0,
    "mean_latency_s": 0.2,
    "label": "before",
    "model": "gpt-oss-120b",
    "provider": "cerebras",
    "timestamp": "2026-10-20T09:00:00",
    "dataset_version": 3,
    "by_course": {
        "Univariate_Time_Series": {"questions": 2, "any_hit": 1, "expected": 2, "found": 1, "any_hit_rate": 0.5}
    },
    "questions": [
        {
            "id": "UTS-01",
            "level": "M1",
            "course": "Univariate_Time_Series",
            "type": "definition",
            "difficulty": 1,
            "question": "Stationnarité faible ?",
            "expected": 1,
            "found": 1,
            "first_rank": 1,
            "candidate_rank": 1,
            "n_candidates": 20,
            "latency_s": 0.2,
            "retrieved": [{"source": "UTS.tex", "slide_title": "Stationnarité"}],
        },
        {
            "id": "UTS-02",
            "level": "M1",
            "course": "Univariate_Time_Series",
            "type": "multi-hop",
            "difficulty": 3,
            "question": "ADF puis PP ?",
            "expected": 1,
            "found": 0,
            "first_rank": None,
            "candidate_rank": 9,
            "n_candidates": 20,
            "latency_s": 0.2,
            "retrieved": [{"source": "UTS.tex", "slide_title": "KPSS"}],
        },
    ],
}


def _answer_report(label: str, fid_a: float, fid_b: float) -> dict:
    return {
        "kind": "answer",
        "label": label,
        "model": "qwen-3.8-27b",
        "provider": "ilaas",
        "judge_model": "gpt-oss-120b",
        "judge_provider": "cerebras",
        "prompt_version": "v1",
        "dataset_version": 3,
        "timestamp": "2026-10-21T09:00:00",
        "n_questions": 2,
        "metrics": {
            "n": 2,
            "n_errors": 0,
            "latency_median_s": 1.1,
            "latency_p90_s": 1.5,
            "searches_mean": 1.0,
            "tool_calls_mean": 1.0,
            "citation_valid_rate": 1.0,
            "questions_with_phantom": 0,
            "fidelity_mean": round((fid_a + fid_b) / 2, 3),
            "fidelity_judged_n": 2,
            "contradictions": 0,
            "unwanted_refusals": 0,
            "ood_refusal_rate": None,
            "nic_general_rate": None,
            "multihop_two_searches_rate": 0.0,
        },
        "by_course": {
            "Univariate_Time_Series": {"questions": 2, "fidelity_mean": 0.6, "latency_median_s": 1.1, "phantom": 0}
        },
        "questions": [
            {
                "id": "UTS-01",
                "course": "Univariate_Time_Series",
                "type": "definition",
                "difficulty": 1,
                "expected_behavior": "answer",
                "question": "Stationnarité faible ?",
                "answer": "Réponse [S1].",
                "error": None,
                "latency_s": 1.0,
                "n_searches": 1,
                "n_tool_calls": 1,
                "steps": ["🔎 Recherche"],
                "sources": ["S1 UTS.tex"],
                "citations_valid": ["S1"],
                "citations_invalid": [],
                "phantom": False,
                "refused": False,
                "general_knowledge": False,
                "fidelity": fid_a,
                "judgment": {
                    "points": [{"point": "moments constants", "covered": fid_a > 0.5, "evidence": "ok"}],
                    "refused": False,
                    "acknowledges_not_in_course": False,
                    "contradiction": False,
                    "comment": "Bien.",
                },
            },
            {
                "id": "UTS-02",
                "course": "Univariate_Time_Series",
                "type": "multi-hop",
                "difficulty": 3,
                "expected_behavior": "answer",
                "question": "ADF puis PP ?",
                "answer": "Réponse [S9].",
                "error": None,
                "latency_s": 1.2,
                "n_searches": 1,
                "n_tool_calls": 1,
                "steps": [],
                "sources": [],
                "citations_valid": [],
                "citations_invalid": ["S9"],
                "phantom": True,
                "refused": False,
                "general_knowledge": False,
                "fidelity": fid_b,
                "judgment": None,
            },
        ],
    }


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Fake accounts/audit, in-memory jobs, a small dataset, a reports directory and a fake child."""
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
    dataset = tmp_path / "questions.yaml"
    dataset.write_text(DATASET, encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    script = tmp_path / "fake_eval.py"
    script.write_text(FAKE_EVAL_CHILD, encoding="utf-8")
    settings = make_settings(
        eval_dataset=str(dataset),
        eval_reports_dir=str(reports),
        cerebras_api_key="k1",
        ilaas_api_key="k2",
        ilaas_chat_base_url="http://ilaas.test/v1",
        llm_model_chat="gpt-oss-120b",
        llm_provider="cerebras",
    )
    return {"db": db, "jobs": jobs, "settings": settings, "reports": reports, "script": script}


@pytest.fixture
async def client(env):
    from backoffice.main import create_app

    app = create_app(env["settings"])
    app.state.jobs._eval_command_factory = fake_eval_factory(env["script"])
    env["app"] = app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", follow_redirects=False) as c:
        yield c


def _write(env, name: str, report: dict) -> None:
    (env["reports"] / name).write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")


# --- access and form ----------------------------------------------------------------------


async def test_index_form_for_admin_and_read_only_for_supervisor(client, env):
    await login(client, "admin")
    r = await client.get(f"{PREFIX}/evaluation")
    assert r.status_code == 200
    assert 'name="label"' in r.text and 'value="20"' in r.text  # default limit
    assert 'value="cerebras"' in r.text and 'value="ilaas"' in r.text and 'value="groq"' not in r.text
    assert "Juge fixe : gpt-oss-120b (cerebras)" in r.text and "Univariate_Time_Series" in r.text
    assert "Aucun rapport" in r.text and 'href="/backoffice/evaluation"' in r.text  # nav tab active
    await client.post(f"{PREFIX}/logout", data={"csrf_token": client.cookies["ela_bo_csrf"]})
    await login(client, "prof")
    r = await client.get(f"{PREFIX}/evaluation")
    assert r.status_code == 200 and 'name="label"' not in r.text and "réservé aux administrateurs" in r.text
    await client.post(f"{PREFIX}/logout", data={"csrf_token": client.cookies["ela_bo_csrf"]})
    await login(client, "etu1")  # students are refused at login: the page redirects to the login form
    assert (await client.get(f"{PREFIX}/evaluation")).status_code in (303, 403)


async def test_start_validation_errors(client, env):
    await login(client, "admin")
    token = client.cookies["ela_bo_csrf"]
    base = {"csrf_token": token, "kind": "answer", "label": "ok", "limit": "20", "judge": "1"}
    for bad, message in (
        ({"label": "bad label"}, "Étiquette invalide"),
        ({"kind": "weird"}, "valuation inconnu"),
        ({"course": "Nope"}, "Cours inconnu"),
        ({"type": "weird"}, "Type de question inconnu"),
        ({"limit": "-1"}, "positive"),
        ({"limit": "abc"}, "entier"),
        ({"provider": "groq"}, "non configuré"),
        ({"model": "bad model!"}, "Nom de modèle invalide"),
        ({"kind": "retrieval", "k": "0"}, "k doit"),
    ):
        r = await client.post(f"{PREFIX}/evaluation/start", data={**base, **bad})
        assert r.status_code == 400 and message in r.text, (bad, message)
    assert env["jobs"].rows == {} and all(a[0] != "eval.start" for a in env["db"].audit)


async def test_start_runs_job_and_records_audit(client, env):
    await login(client, "admin")
    token = client.cookies["ela_bo_csrf"]
    data = {
        "csrf_token": token,
        "kind": "answer",
        "label": "agent-v2",
        "course": "Univariate_Time_Series",
        "type": "multi-hop",
        "limit": "",
        "judge": "1",
        "provider": "ilaas",
        "model": "qwen-3.8-27b",
    }
    r = await client.post(f"{PREFIX}/evaluation/start", data=data)
    assert r.status_code == 303 and r.headers["location"] == f"{PREFIX}/jobs/1"
    env["app"].state.jobs.wait(20)
    row = env["jobs"].rows[1]
    assert row["kind"] == "eval" and row["mode"] == "answer" and row["target"] == "agent-v2" and row["by"] == "admin"
    assert row["status"] == "success" and row["result"]["report"] == "20261027-120000-test-answer.json"
    # pinned judge and disabled fallback reached the child
    assert row["result"]["env"]["LLM_MODEL_CHAT"] == "qwen-3.8-27b" and row["result"]["env"]["LLM_PROVIDER"] == "ilaas"
    assert row["result"]["env"]["EVAL_JUDGE_MODEL"] == "gpt-oss-120b"
    assert row["result"]["env"]["EVAL_JUDGE_PROVIDER"] == "cerebras"
    assert row["result"]["env"]["LLM_MODEL_FALLBACK"] == "none"
    action, actor, details = env["db"].audit[-1]
    assert action == "eval.start" and actor == "admin"
    assert details == {
        "kind": "answer",
        "label": "agent-v2",
        "course": "Univariate_Time_Series",
        "type": "multi-hop",
        "judge": True,
        "model": "qwen-3.8-27b",
        "provider": "ilaas",
    }
    r = await client.get(f"{PREFIX}/jobs/1")
    assert r.status_code == 200 and "Évaluation des réponses" in r.text and "agent-v2" in r.text
    assert "/evaluation/reports/20261027-120000-test-answer.json" in r.text and "fidelity_mean" in r.text
    r = await client.get(f"{PREFIX}/evaluation")
    assert "Dernière évaluation" in r.text and "#1" in r.text
    r = await client.get(f"{PREFIX}/jobs")
    assert "Évaluation des réponses · agent-v2" in r.text and "rapport" in r.text


async def test_start_without_override_keeps_server_model(client, env):
    await login(client, "admin")
    token = client.cookies["ela_bo_csrf"]
    data = {"csrf_token": token, "kind": "retrieval", "label": "r1", "k": "10", "model": "gpt-oss-120b", "judge": "1"}
    r = await client.post(f"{PREFIX}/evaluation/start", data=data)
    assert r.status_code == 303
    env["app"].state.jobs.wait(20)
    row = env["jobs"].rows[1]
    assert row["mode"] == "retrieval" and row["result"]["env"]["LLM_MODEL_FALLBACK"] is None
    assert env["db"].audit[-1][2] == {"kind": "retrieval", "label": "r1", "k": 10}


async def test_start_refused_while_a_job_runs_and_for_supervisor(client, env):
    env["jobs"].create_sync("ingest", mode="full", target=None, started_by="admin")
    await login(client, "admin")
    token = client.cookies["ela_bo_csrf"]
    r = await client.post(f"{PREFIX}/evaluation/start", data={"csrf_token": token, "kind": "retrieval", "label": "x"})
    assert r.status_code == 409 and "déjà en cours" in r.text
    r = await client.get(f"{PREFIX}/evaluation/partials/launch")
    assert r.status_code == 200 and 'hx-trigger="every 2s"' in r.text and "Une ingestion occupe" in r.text
    r = await client.get(f"{PREFIX}/evaluation")
    assert 'name="label"' not in r.text  # the form is hidden while a job runs
    await client.post(f"{PREFIX}/logout", data={"csrf_token": client.cookies["ela_bo_csrf"]})
    env["jobs"].rows.clear()
    await login(client, "prof")
    token = client.cookies["ela_bo_csrf"]
    r = await client.post(f"{PREFIX}/evaluation/start", data={"csrf_token": token, "kind": "retrieval", "label": "x"})
    assert r.status_code == 403


async def test_start_refused_when_reports_dir_not_writable(client, env):
    await login(client, "admin")
    token = client.cookies["ela_bo_csrf"]
    env["app"].state.reports.writable = lambda: False
    r = await client.post(f"{PREFIX}/evaluation/start", data={"csrf_token": token, "kind": "retrieval", "label": "x"})
    assert r.status_code == 500 and "inscriptible" in r.text
    r = await client.get(f"{PREFIX}/evaluation")
    assert "inscriptible" in r.text and 'name="label"' not in r.text


# --- reports ------------------------------------------------------------------------------


async def test_report_pages_csv_download_and_delete(client, env):
    _write(env, "20261020-090000-before-retrieval.json", RETRIEVAL_REPORT)
    _write(env, "20261021-090000-agent-answer.json", _answer_report("agent", 1.0, 0.25))
    _write(env, "20261019-090000-old-answer.json", _answer_report("old", 0.5, 0.5))
    await login(client, "admin")
    r = await client.get(f"{PREFIX}/evaluation")
    assert r.status_code == 200 and "3 rapport(s)" in r.text and "before" in r.text and "agent" in r.text
    assert "any-hit@k" in r.text and "fidélité" in r.text and "qwen-3.8-27b" in r.text
    r = await client.get(f"{PREFIX}/evaluation?kind=answer")
    assert "2 rapport(s)" in r.text and "before" not in r.text

    r = await client.get(f"{PREFIX}/evaluation/reports/20261020-090000-before-retrieval.json")
    assert r.status_code == 200 and "Rapport « before »" in r.text and "non trouvée" in r.text
    assert r.text.index("UTS-02") < r.text.index("UTS-01")  # failures first
    assert "KPSS" in r.text and "1 à regarder" in r.text

    r = await client.get(f"{PREFIX}/evaluation/reports/20261021-090000-agent-answer.json")
    assert r.status_code == 200 and "citation fantôme" in r.text and "juge gpt-oss-120b" in r.text
    assert "moments constants" in r.text and "Bien." in r.text and "1 à regarder" in r.text
    assert "compare?a=20261019-090000-old-answer.json&b=20261021-090000-agent-answer.json" in r.text

    r = await client.get(f"{PREFIX}/evaluation/reports/20261021-090000-agent-answer.json/questions.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    lines = r.text.lstrip("﻿").splitlines()
    assert lines[0].startswith("Id;Cours;Type") and lines[1].startswith("UTS-02;") and "citation fantôme" in lines[1]

    r = await client.get(f"{PREFIX}/evaluation/reports/20261020-090000-before-retrieval.json/download")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    assert json.loads(r.content)["mrr"] == 0.5

    for missing in ("nope.json", "..%2Fx.json", "x.txt"):
        assert (await client.get(f"{PREFIX}/evaluation/reports/{missing}")).status_code == 404

    token = client.cookies["ela_bo_csrf"]
    name = "20261020-090000-before-retrieval.json"
    r = await client.post(f"{PREFIX}/evaluation/reports/{name}/delete", data={"csrf_token": token})
    assert r.status_code == 303 and not (env["reports"] / "20261020-090000-before-retrieval.json").exists()
    assert env["db"].audit[-1][0] == "eval.delete"
    r = await client.post(f"{PREFIX}/evaluation/reports/nope.json/delete", data={"csrf_token": token})
    assert r.status_code == 404


async def test_supervisor_reads_but_cannot_delete(client, env):
    _write(env, "20261021-090000-agent-answer.json", _answer_report("agent", 1.0, 0.25))
    await login(client, "prof")
    r = await client.get(f"{PREFIX}/evaluation/reports/20261021-090000-agent-answer.json")
    assert r.status_code == 200 and "Supprimer" not in r.text
    token = client.cookies["ela_bo_csrf"]
    name = "20261021-090000-agent-answer.json"
    r = await client.post(f"{PREFIX}/evaluation/reports/{name}/delete", data={"csrf_token": token})
    assert r.status_code == 403 and (env["reports"] / "20261021-090000-agent-answer.json").exists()


async def test_unreadable_report_is_listed_but_not_opened(client, env):
    (env["reports"] / "20261022-000000-broken-answer.json").write_text("{", encoding="utf-8")
    _write(env, "20261022-000001-foreign-answer.json", {"foo": 1})
    await login(client, "admin")
    r = await client.get(f"{PREFIX}/evaluation")
    assert "illisible" in r.text
    assert (await client.get(f"{PREFIX}/evaluation/reports/20261022-000000-broken-answer.json")).status_code == 404
    assert (await client.get(f"{PREFIX}/evaluation/reports/20261022-000001-foreign-answer.json")).status_code == 415


# --- comparison ---------------------------------------------------------------------------


async def test_compare_page(client, env):
    after = dict(RETRIEVAL_REPORT, label="after", mrr=0.8, hit_at_k=1.0, timestamp="2026-10-22T09:00:00")
    after["questions"] = [dict(q, found=1, first_rank=1) for q in RETRIEVAL_REPORT["questions"]]
    _write(env, "20261020-090000-before-retrieval.json", RETRIEVAL_REPORT)
    _write(env, "20261022-090000-after-retrieval.json", after)
    _write(env, "20261021-090000-agent-answer.json", _answer_report("agent", 1.0, 0.25))
    await login(client, "prof")
    r = await client.get(f"{PREFIX}/evaluation/compare")
    assert r.status_code == 200 and 'name="a"' in r.text and "Choisissez" not in r.text
    r = await client.get(
        f"{PREFIX}/evaluation/compare?a=20261020-090000-before-retrieval.json&b=20261022-090000-after-retrieval.json"
    )
    assert r.status_code == 200 and "« before » → « after »" in r.text
    assert "+0.300" in r.text and "delta-better" in r.text and "Questions gagnées" in r.text and "UTS-02" in r.text
    r = await client.get(
        f"{PREFIX}/evaluation/compare?a=20261020-090000-before-retrieval.json&b=20261021-090000-agent-answer.json"
    )
    assert r.status_code == 200 and "même type" in r.text
    r = await client.get(f"{PREFIX}/evaluation/compare?a=20261020-090000-before-retrieval.json")
    assert "Choisissez deux rapports" in r.text
    r = await client.get(f"{PREFIX}/evaluation/compare?a=nope.json&b=20261022-090000-after-retrieval.json")
    assert "introuvable" in r.text
