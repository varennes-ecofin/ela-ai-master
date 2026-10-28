"""Report comparison (ela.eval.compare) and report index (ela.eval.reports)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from ela.eval.compare import (
    KIND_ANSWER,
    KIND_RETRIEVAL,
    ReportKindError,
    compare_reports,
    format_comparison,
    report_kind,
    report_metrics,
)
from ela.eval.reports import (
    ReportIndex,
    ReportNameError,
    count_failures,
    report_name,
    report_path,
    summarize_report,
    valid_label,
)

RETRIEVAL_A = {
    "k": 5,
    "n_questions": 3,
    "hit_at_k": 0.5,
    "any_hit_at_k": 0.667,
    "mrr": 0.4,
    "any_hit_candidates": 1.0,
    "mean_latency_s": 1.2,
    "by_course": {},
    "label": "before",
    "questions": [
        {"id": "A", "found": 1, "expected": 1},
        {"id": "B", "found": 0, "expected": 2},
        {"id": "C", "found": 1, "expected": 1},
    ],
}
RETRIEVAL_B = {
    "kind": "retrieval",
    "k": 5,
    "n_questions": 3,
    "hit_at_k": 0.9,
    "any_hit_at_k": 1.0,
    "mrr": 0.7,
    "any_hit_candidates": 1.0,
    "mean_latency_s": 0.3,
    "by_course": {},
    "label": "after",
    "model": "gpt-oss-120b",
    "provider": "cerebras",
    "timestamp": "2026-10-27T10:00:00",
    "questions": [
        {"id": "A", "found": 0, "expected": 1},
        {"id": "B", "found": 2, "expected": 2},
        {"id": "C", "found": 1, "expected": 1},
        {"id": "D", "found": 1, "expected": 1},
    ],
}


def _answer(label: str, fidelities: dict[str, float | None], **metrics) -> dict:
    base = {
        "latency_median_s": 1.0,
        "latency_p90_s": 2.0,
        "searches_mean": 1.1,
        "tool_calls_mean": 1.3,
        "citation_valid_rate": 1.0,
        "questions_with_phantom": 0,
        "fidelity_mean": 0.8,
        "fidelity_judged_n": len(fidelities),
        "contradictions": 0,
        "unwanted_refusals": 0,
        "ood_refusal_rate": 1.0,
        "nic_general_rate": 0.5,
        "multihop_two_searches_rate": 0.0,
        "n_errors": 0,
    }
    base.update(metrics)
    return {
        "kind": "answer",
        "label": label,
        "n_questions": len(fidelities),
        "metrics": base,
        "by_course": {},
        "questions": [
            {"id": i, "fidelity": f, "error": None, "phantom": False, "expected_behavior": "answer"}
            for i, f in fidelities.items()
        ],
    }


# --- compare ------------------------------------------------------------------------------


def test_report_kind_is_inferred_for_old_reports():
    assert report_kind(RETRIEVAL_A) == KIND_RETRIEVAL
    assert report_kind(RETRIEVAL_B) == KIND_RETRIEVAL
    assert report_kind(_answer("x", {})) == KIND_ANSWER
    assert report_metrics(_answer("x", {"A": 1.0}))["fidelity_mean"] == 0.8
    with pytest.raises(ReportKindError):
        report_kind({"foo": 1})


def test_compare_retrieval_rows_and_question_changes():
    cmp = compare_reports(RETRIEVAL_A, RETRIEVAL_B)
    assert cmp.kind == KIND_RETRIEVAL and (cmp.label_before, cmp.label_after) == ("before", "after")
    rows = {r.key: r for r in cmp.rows}
    assert rows["hit_at_k"].delta == pytest.approx(0.4) and rows["hit_at_k"].trend == "better"
    assert rows["mean_latency_s"].delta == pytest.approx(-0.9) and rows["mean_latency_s"].trend == "better"
    assert rows["any_hit_candidates"].trend == "same"
    assert cmp.gained == ["B"] and cmp.lost == ["A"] and cmp.only_after == ["D"] and cmp.only_before == []
    text = format_comparison(cmp)
    assert "Questions gagnées : ['B']" in text and "hit@k" in text and "+0.400" in text


def test_compare_answer_fidelity_changes_and_missing_metric():
    a = _answer("before", {"A": 0.5, "B": 1.0, "C": None})
    b = _answer("after", {"A": 1.0, "B": 0.5, "C": 0.5}, fidelity_mean=0.9, questions_with_phantom=2)
    del a["metrics"]["nic_general_rate"]
    cmp = compare_reports(a, b)
    rows = {r.key: r for r in cmp.rows}
    assert rows["fidelity_mean"].trend == "better" and rows["questions_with_phantom"].trend == "worse"
    assert rows["nic_general_rate"].before is None and rows["nic_general_rate"].delta is None
    assert rows["nic_general_rate"].trend == "n/a"
    assert cmp.gained == ["A"] and cmp.lost == ["B"]  # C: no fidelity before → ignored
    assert cmp.change_labels == ("Fidélité en hausse", "Fidélité en baisse")


def test_compare_refuses_different_kinds():
    with pytest.raises(ReportKindError, match="même type"):
        compare_reports(RETRIEVAL_A, _answer("x", {"A": 1.0}))


# --- reports index ------------------------------------------------------------------------


def test_labels_and_names_are_validated():
    assert valid_label("agent-v2") and valid_label("bge_m3.k5")
    assert not valid_label("") and not valid_label("a b") and not valid_label("../x") and not valid_label("x" * 41)
    name = report_name("agent", "answer", datetime(2026, 10, 27, 10, 5, 7))
    assert name == "20261027-100507-agent-answer.json"
    with pytest.raises(ReportNameError):
        report_name("bad label", "answer")
    for bad in ("../x.json", "x/y.json", "x.txt", "", ".json", "a\\b.json"):
        with pytest.raises(ReportNameError):
            report_path(bad, Path("/tmp"))
    assert report_path("ok-1.json", Path("/tmp")) == Path("/tmp/ok-1.json")


def test_summarize_and_failures():
    s = summarize_report("20261027-100000-after-retrieval.json", RETRIEVAL_B, size=10, mtime=0)
    assert s.kind == "retrieval" and s.label == "after" and s.model == "gpt-oss-120b" and s.provider == "cerebras"
    assert s.timestamp == datetime(2026, 10, 27, 10, 0, 0) and s.n_questions == 3 and s.n_failures == 1
    assert [k for k, _, _ in s.headline] == ["any_hit_at_k", "mrr", "mean_latency_s"]
    assert s.stem == "20261027-100000-after-retrieval"
    old = summarize_report("20261014-120000-before-retrieval.json", RETRIEVAL_A, mtime=0)
    assert old.timestamp == datetime(2026, 10, 14, 12, 0, 0)  # from the file name
    ans = _answer("a", {"A": 0.2, "B": 1.0})
    ans["questions"][1]["phantom"] = True
    assert count_failures("answer", ans) == 2


def test_index_lists_loads_caches_and_deletes(tmp_path: Path):
    index = ReportIndex(tmp_path / "reports")
    assert index.list() == [] and index.writable()
    (tmp_path / "reports" / "20261027-100000-after-retrieval.json").write_text(
        json.dumps(RETRIEVAL_B), encoding="utf-8"
    )
    (tmp_path / "reports" / "20261014-120000-before-retrieval.json").write_text(
        json.dumps(RETRIEVAL_A), encoding="utf-8"
    )
    (tmp_path / "reports" / "20261020-000000-x-answer.json").write_text(json.dumps(_answer("x", {"A": 1.0})))
    (tmp_path / "reports" / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "reports" / "notes.txt").write_text("ignored", encoding="utf-8")
    listed = index.list()
    assert [s.label for s in listed] == ["after", "x", "before", "broken"]
    assert listed[-1].valid is False and listed[-1].kind == "?"
    assert [s.label for s in index.list("retrieval")] == ["after", "before"]
    # cache: same mtime/size → same object
    assert index.list()[0] is listed[0]
    assert index.get("20261027-100000-after-retrieval.json").label == "after"
    assert index.get("../etc/passwd") is None and index.get("missing.json") is None
    assert index.load("20261027-100000-after-retrieval.json")["mrr"] == 0.7
    assert index.load("broken.json") is None and index.load("x/y.json") is None
    assert index.path("20261027-100000-after-retrieval.json").is_file() and index.path("nope.json") is None
    assert index.delete("20261027-100000-after-retrieval.json")
    assert not index.delete("20261027-100000-after-retrieval.json")
    assert [s.label for s in index.list("retrieval")] == ["before"]


def test_index_without_directory_is_empty(tmp_path: Path):
    index = ReportIndex(tmp_path / "missing")
    assert index.list() == [] and index.get("x.json") is None
