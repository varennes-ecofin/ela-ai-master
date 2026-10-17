"""Schema checks on the reference question set (no corpus needed)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from langchain_core.documents import Document

from ela.eval.dataset import ExpectedSource, load_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_loads_and_is_consistent():
    ds = load_dataset(ROOT / "eval" / "questions.yaml")
    assert ds.version == 1
    assert len(ds.questions) >= 50
    for q in ds.questions:
        assert q.question.strip().endswith(("?", ".")), q.id
        assert q.answer_points, q.id
        if q.type in ("out-of-domain", "not-in-corpus"):
            assert not q.expected_sources, q.id
        else:
            assert q.expected_sources, q.id
    counts = Counter(q.course for q in ds.questions)
    assert set(counts) >= {
        "Univariate_Time_Series",
        "Multivariate_Time_Series",
        "Financial_Econometrics",
        "Duration_Models",
        "Research_Econometrics",
    }


def test_expected_source_matching():
    exp = ExpectedSource(source="UTS.tex", slide_title="Stationnarité faible")
    ok = Document(page_content="", metadata={"source": "UTS.tex", "slide_title": "Stationnarité faible"})
    wrong_file = Document(page_content="", metadata={"source": "MTS.tex", "slide_title": "Stationnarité faible"})
    wrong_title = Document(page_content="", metadata={"source": "UTS.tex", "slide_title": "Stationnarité stricte"})
    assert exp.matches(ok) and not exp.matches(wrong_file) and not exp.matches(wrong_title)
    sec = ExpectedSource(source="hdr_main.tex", section="La filtration des MAR")
    hdr = Document(page_content="", metadata={"source": "hdr_main.tex", "section": "La filtration des MAR"})
    assert sec.matches(hdr)


def test_hits_and_rank():
    ds = load_dataset(ROOT / "eval" / "questions.yaml")
    q = next(q for q in ds.questions if q.id == "MTS-05")
    docs = [
        Document(page_content="", metadata={"source": "MTS.tex", "slide_title": "Ordonnancement"}),
        Document(
            page_content="", metadata={"source": "MTS.tex", "slide_title": "Définition de la causalité instantanée"}
        ),
    ]
    assert q.hits(docs) == [True, False]
    assert q.first_hit_rank(docs) == 2
    assert q.first_hit_rank([]) is None


def test_runner_metrics_with_fake_retriever(monkeypatch):
    """evaluate_retrieval computes hit@k / any-hit / MRR from a fake retriever."""
    from ela.eval import runner
    from ela.retrieval import retriever as retriever_mod

    ds = load_dataset(ROOT / "eval" / "questions.yaml")
    questions = [q for q in ds.questions if q.id in ("UTS-01", "MTS-05", "SA-04")]

    class FakeRetriever:
        def __init__(self, level):
            self.level = level

        def invoke(self, query):
            def doc(src, title):
                return Document(page_content="", metadata={"source": src, "slide_title": title})

            if "stationnarité faible" in query.lower():
                return [doc("UTS.tex", "Stationnarité faible")]
            if "instantanée" in query:
                return [doc("MTS.tex", "Autre"), doc("MTS.tex", "Définition de la Causalité en multivarié")]
            return []

    monkeypatch.setattr(retriever_mod, "get_retriever", FakeRetriever)
    import ela.retrieval

    monkeypatch.setattr(ela.retrieval, "get_retriever", FakeRetriever)
    report = runner.evaluate_retrieval(questions, k=5)
    assert report["n_questions"] == 3
    # UTS-01: 1/1 ; MTS-05: 1/2 ; SA-04: 0/1  → hit@k = 2/4
    assert report["hit_at_k"] == 0.5
    assert report["any_hit_at_k"] == round(2 / 3, 3)
    assert report["mrr"] == round((1 + 0.5 + 0) / 3, 3)
    assert report["by_course"]["Duration_Models"]["any_hit_rate"] == 0.0
