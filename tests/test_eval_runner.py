"""Command-line pieces of ``ela-eval`` that need no model: filters, provenance, report files, ``--result``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from ela.eval import runner
from ela.eval.dataset import EvalQuestion


def _q(qid: str, course: str = "UTS", qtype: str = "definition") -> EvalQuestion:
    return EvalQuestion(
        id=qid,
        level="M1",
        course=course,
        type=qtype,
        difficulty=1,
        question="?",
        expected_sources=[{"source": "UTS.tex", "slide_title": "x"}],
        answer_points=["p"],
    )


def test_select_questions_filters():
    qs = [_q("A"), _q("B", "MTS"), _q("C", qtype="multi-hop"), _q("D")]
    assert [q.id for q in runner.select_questions(qs)] == ["A", "B", "C", "D"]
    assert [q.id for q in runner.select_questions(qs, course="MTS")] == ["B"]
    assert [q.id for q in runner.select_questions(qs, qtype="multi-hop")] == ["C"]
    assert [q.id for q in runner.select_questions(qs, ids="D, A")] == ["A", "D"]
    assert [q.id for q in runner.select_questions(qs, limit=2)] == ["A", "B"]
    assert [q.id for q in runner.select_questions(qs, course="UTS", limit=1)] == ["A"]


def test_progress_line_format():
    assert runner.progress_line(3, 71, "UTS-03", "0.5s") == "[3/71] UTS-03   0.5s"
    assert runner.progress_line(71, 71, "NCE-05") == "[71/71] NCE-05"


def test_run_metadata_records_model_and_provider(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_CHAT", "m-test")
    monkeypatch.setenv("LLM_PROVIDER", "ilaas")
    monkeypatch.setenv("PROMPT_VERSION", "v9")
    from ela.config import get_settings

    get_settings.cache_clear()
    try:
        meta = runner.run_metadata("answer", "lab", 3)
    finally:
        get_settings.cache_clear()
    assert meta["kind"] == "answer" and meta["label"] == "lab" and meta["dataset_version"] == 3
    assert meta["model"] == "m-test" and meta["provider"] == "ilaas" and meta["prompt_version"] == "v9"
    assert meta["timestamp"].startswith("20")


def test_save_report_and_result_summary(tmp_path: Path):
    report = {
        "kind": "retrieval",
        "label": "lab",
        "model": "m",
        "provider": "cerebras",
        "n_questions": 1,
        "k": 5,
        "hit_at_k": 1.0,
        "any_hit_at_k": 1.0,
        "mrr": 1.0,
        "any_hit_candidates": 1.0,
        "mean_latency_s": 0.1,
        "by_course": {},
        "questions": [],
    }
    out = runner.save_report(report, "lab", "retrieval", reports_dir=tmp_path / "r")
    assert out.parent == tmp_path / "r" and out.name.endswith("-lab-retrieval.json")
    assert json.loads(out.read_text(encoding="utf-8"))["mrr"] == 1.0
    summary = runner.result_summary(report, out)
    assert summary["report"] == out.name and summary["kind"] == "retrieval" and summary["label"] == "lab"
    assert summary["model"] == "m" and summary["n_errors"] == 0 and summary["metrics"]["mrr"] == 1.0
    assert "questions" not in summary and "by_course" not in summary
    result_file = tmp_path / "result.json"
    runner.write_result(argparse.Namespace(result=str(result_file)), report, out)
    assert json.loads(result_file.read_text(encoding="utf-8"))["report"] == out.name
    runner.write_result(argparse.Namespace(result=None), report, out)  # no-op
    with pytest.raises(runner.ReportNameError):
        runner.save_report(report, "bad label", "retrieval", reports_dir=tmp_path)


def test_cmd_retrieval_writes_report_and_result(tmp_path: Path, monkeypatch, capsys):
    """``ela-eval retrieval`` end to end with a fake retriever, settings pointing at tmp dirs."""
    dataset = tmp_path / "q.yaml"
    dataset.write_text(
        "version: 2\nquestions:\n"
        "- {id: A, level: M1, course: UTS, type: definition, difficulty: 1, question: 'a?',"
        " expected_sources: [{source: UTS.tex, slide_title: 'ACF'}]}\n"
        "- {id: B, level: M1, course: UTS, type: formula, difficulty: 1, question: 'b?',"
        " expected_sources: [{source: UTS.tex, slide_title: 'PACF'}]}\n"
        "- {id: OOD, level: M1, course: UTS, type: out-of-domain, difficulty: 1, question: 'c?'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EVAL_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("LLM_MODEL_CHAT", "fake-model")
    from ela.config import get_settings

    get_settings.cache_clear()

    def fake_evaluate(questions, k, progress=None):
        records = []
        for i, q in enumerate(questions):
            rec = {
                "id": q.id,
                "level": q.level,
                "course": q.course,
                "type": q.type,
                "difficulty": 1,
                "question": q.question,
                "expected": 1,
                "found": 1 - i,
                "first_rank": 1 if i == 0 else None,
                "candidate_rank": 1,
                "n_candidates": 3,
                "latency_s": 0.1,
                "retrieved": [],
            }
            records.append(rec)
            if progress:
                progress(rec)
        return {
            "k": k,
            "n_questions": len(records),
            "hit_at_k": 0.5,
            "any_hit_at_k": 0.5,
            "mrr": 0.5,
            "any_hit_candidates": 1.0,
            "mean_latency_s": 0.1,
            "by_course": {"UTS": {"questions": 2, "any_hit": 1, "expected": 2, "found": 1, "any_hit_rate": 0.5}},
            "questions": records,
        }

    monkeypatch.setattr(runner, "evaluate_retrieval", fake_evaluate)
    result_file = tmp_path / "result.json"
    args = argparse.Namespace(
        dataset=str(dataset), k=5, label="fake", course=None, type=None, limit=0, result=str(result_file)
    )
    try:
        runner.cmd_retrieval(args)
    finally:
        get_settings.cache_clear()
    out = capsys.readouterr().out
    assert "[1/2] A" in out and "[2/2] B" in out and "non trouvée" in out and "Rapport :" in out
    files = list((tmp_path / "reports").glob("*-fake-retrieval.json"))
    assert len(files) == 1
    report = json.loads(files[0].read_text(encoding="utf-8"))
    assert report["kind"] == "retrieval" and report["label"] == "fake" and report["model"] == "fake-model"
    assert report["dataset_version"] == 2 and report["n_questions"] == 2  # OOD question has no expected source
    result = json.loads(result_file.read_text(encoding="utf-8"))
    assert result["report"] == files[0].name and result["metrics"]["mrr"] == 0.5


def test_cmd_retrieval_rejects_bad_label_and_empty_selection(tmp_path: Path, monkeypatch):
    dataset = tmp_path / "q.yaml"
    dataset.write_text(
        "version: 1\nquestions:\n"
        "- {id: A, level: M1, course: UTS, type: definition, difficulty: 1, question: 'a?',"
        " expected_sources: [{source: UTS.tex, slide_title: 'ACF'}]}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EVAL_REPORTS_DIR", str(tmp_path / "reports"))
    from ela.config import get_settings

    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit, match="invalide"):
            runner.cmd_retrieval(
                argparse.Namespace(dataset=str(dataset), k=5, label="a b", course=None, type=None, limit=0, result=None)
            )
        with pytest.raises(SystemExit, match="Aucune question"):
            runner.cmd_retrieval(
                argparse.Namespace(
                    dataset=str(dataset), k=5, label="ok", course="Nope", type=None, limit=0, result=None
                )
            )
    finally:
        get_settings.cache_clear()


def test_judge_model_uses_its_own_provider(monkeypatch):
    calls = []
    import ela.llm as llm_mod

    monkeypatch.setattr(
        llm_mod, "build_chat_model", lambda model, provider, **kw: calls.append((model, provider, kw)) or "judge"
    )
    monkeypatch.setenv("LLM_MODEL_CHAT", "evaluated")
    monkeypatch.setenv("LLM_PROVIDER", "ilaas")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "gpt-oss-120b")
    monkeypatch.setenv("EVAL_JUDGE_PROVIDER", "cerebras")
    from ela.config import get_settings

    get_settings.cache_clear()
    try:
        judge, model, provider = runner._judge_model()
    finally:
        get_settings.cache_clear()
    assert judge == "judge" and (model, provider) == ("gpt-oss-120b", "cerebras")
    assert calls == [("gpt-oss-120b", "cerebras", {"reasoning_effort": "medium", "streaming": False})]


def test_main_has_no_engine_option(capsys):
    import sys

    argv = sys.argv
    sys.argv = ["ela-eval", "answer", "--engine", "legacy"]
    try:
        with pytest.raises(SystemExit) as exc:
            runner.main()
        assert exc.value.code == 2 and "--engine" in capsys.readouterr().err
    finally:
        sys.argv = argv
