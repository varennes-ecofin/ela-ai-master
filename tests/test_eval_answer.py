"""Tests for the end-to-end evaluation (judge + runner + metrics) with fakes."""

from __future__ import annotations

import json

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from ela.eval.answer import AgentEngineRunner, LegacyEngineRunner, RunResult, evaluate_answers, summarize
from ela.eval.dataset import EvalQuestion
from ela.eval.judge import Judgment, expected_behavior, judge_answer


def _q(id: str, type: str = "definition", points=None) -> EvalQuestion:
    return EvalQuestion(
        id=id,
        level="M1",
        course="Univariate_Time_Series",
        type=type,
        difficulty=1,
        question=f"Question {id} ?",
        expected_sources=[],
        answer_points=points if points is not None else ["point A", "point B"],
    )


def _judge_llm(*verdicts: dict) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([json.dumps(v) for v in verdicts]))


def _verdict(covered: list[bool], refused=False, ack=False, contradiction=False) -> dict:
    return {
        "points": [{"point": f"p{i}", "covered": c, "evidence": ""} for i, c in enumerate(covered)],
        "refused": refused,
        "acknowledges_not_in_course": ack,
        "contradiction": contradiction,
        "comment": "ok",
    }


def test_expected_behavior():
    assert expected_behavior("definition") == "answer"
    assert expected_behavior("out-of-domain") == "refuse"
    assert expected_behavior("not-in-corpus") == "general"


async def test_judge_answer_keeps_teacher_wording():
    llm = _judge_llm(_verdict([True, False]))
    j = await judge_answer(llm, question="q", answer="a", answer_points=["point A", "point B"])
    assert isinstance(j, Judgment)
    assert [p.point for p in j.points] == ["point A", "point B"]
    assert j.fidelity == 0.5


class FakeRunner:
    def __init__(self, results: dict[str, RunResult]):
        self.results = results

    async def run(self, question):
        return self.results[question.id]


async def test_evaluate_answers_metrics_and_phantoms():
    results = {
        "A": RunResult(
            answer="Réponse [S1] et [S3].", latency_s=1.0, n_tool_calls=2, n_searches=2, known_ids=["S1", "S2"]
        ),
        "B": RunResult(answer="Réponse [S1].", latency_s=3.0, n_tool_calls=1, n_searches=1, known_ids=["S1"]),
        "OOD": RunResult(
            answer="Je suis spécialisé en séries temporelles. Ce sujet sort du cadre du cours.", latency_s=0.5
        ),
        "NIC": RunResult(
            answer="Le cours n'en parle pas. [Connaissances générales] KPSS...", latency_s=2.0, n_searches=1
        ),
    }
    questions = [_q("A", "multi-hop"), _q("B"), _q("OOD", "out-of-domain", []), _q("NIC", "not-in-corpus", [])]
    judge = _judge_llm(
        _verdict([True, True]),
        _verdict([True, False]),
        _verdict([], refused=True),
        _verdict([], ack=True),
    )
    seen = []
    report = await evaluate_answers(questions, FakeRunner(results), judge=judge, progress=seen.append)
    m = report["metrics"]
    assert len(seen) == 4 and report["n_questions"] == 4
    assert m["questions_with_phantom"] == 1 and m["citation_valid_rate"] == round(2 / 3, 3)
    assert m["fidelity_mean"] == 0.75 and m["fidelity_judged_n"] == 2
    assert m["ood_refusal_rate"] == 1.0 and m["nic_general_rate"] == 1.0
    assert m["multihop_two_searches_rate"] == 1.0 and m["unwanted_refusals"] == 0
    assert m["latency_median_s"] == 1.5
    assert report["by_course"]["Univariate_Time_Series"]["questions"] == 2
    a = next(r for r in report["questions"] if r["id"] == "A")
    assert a["citations_invalid"] == ["S3"] and a["phantom"] is True


async def test_evaluate_without_judge_uses_heuristics():
    results = {"OOD": RunResult(answer="Ce sujet sort du cadre du cours.", latency_s=0.2)}
    report = await evaluate_answers([_q("OOD", "out-of-domain", [])], FakeRunner(results))
    assert report["metrics"]["ood_refusal_rate"] == 1.0 and report["metrics"]["fidelity_mean"] is None


def test_summarize_empty():
    m = summarize([])
    assert m["n"] == 0 and m["latency_median_s"] is None and m["citation_valid_rate"] is None


async def test_legacy_runner_collects_docs_and_legacy_citations():
    class FakePipeline:
        def __init__(self):
            self.last_docs = []

        async def answer(self, question):
            self.last_docs = [Document(page_content="x", metadata={"source": "UTS.tex", "slide_title": "ACF"})]
            return "Réponse [Source: UTS.tex]. Autre [Source: MTS.tex]."

    runner = LegacyEngineRunner(pipeline_factory=lambda level: FakePipeline())
    result = await runner.run(_q("A"))
    assert result.n_searches == 1 and result.sources == ["UTS.tex — ACF"]
    check = result.citation_check()
    assert check.valid == ["UTS.tex"] and check.invalid == ["MTS.tex"]


async def test_agent_runner_collects_events():
    from ela.agent.citations import SourceRegistry
    from ela.agent.context import TurnContext
    from ela.agent.events import FinalEvent, ToolStartEvent

    class FakeEngine:
        def __init__(self):
            self.last_context = TurnContext(level="M1", registry=SourceRegistry())

        async def ask(self, question, chat_history=None, images=None):
            self.last_context.registry.register(
                Document(page_content="x", metadata={"chunk_id": "c1", "source": "UTS.tex"})
            )
            yield ToolStartEvent("c", "search_course", {"query": "q"}, "🔎 Recherche : « q »")
            yield FinalEvent(
                text="Réponse [S1].", citations_valid=["S1"], n_model_calls=2, n_tool_calls=1, n_searches=1
            )

    runner = AgentEngineRunner(engine_factory=lambda level: FakeEngine())
    result = await runner.run(_q("A"))
    assert result.known_ids == ["S1"] and result.n_model_calls == 2 and result.steps == ["🔎 Recherche : « q »"]
    assert result.citation_check().clean and result.sources == ["S1 UTS.tex"]
