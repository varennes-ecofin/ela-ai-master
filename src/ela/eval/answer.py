"""End-to-end evaluation (``ela-eval answer``).

Each reference question is answered by the conversation engine exactly as a
student would receive it (level-scoped retrieval, prompts, model). The run
records latency, the number of searches / tool calls, the passages handed to
the model and the citations of the answer; an optional LLM judge then grades
the answer against the teacher's ``answer_points``.

Metrics (``report["metrics"]``):

* ``latency_median_s`` / ``latency_p90_s`` — wall-clock time per answer;
* ``searches_mean`` / ``tool_calls_mean`` — agent activity;
* ``citation_valid_rate`` / ``questions_with_phantom`` — every citation in the
  answer must point to a passage really given to the model;
* ``fidelity_mean`` — share of expected points covered (judge);
* ``ood_refusal_rate`` — out-of-domain questions refused;
* ``nic_general_rate`` — not-in-corpus questions answered as general knowledge;
* ``multihop_two_searches_rate`` — multi-hop questions answered with ≥ 2 searches.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from langchain_core.language_models import BaseChatModel

from ela.agent.citations import (
    CitationCheck,
    extract_citations,
    has_general_knowledge_tag,
    verify_citations,
    verify_legacy_citations,
)
from ela.eval.dataset import EvalQuestion
from ela.eval.judge import Judgment, expected_behavior, judge_answer

logger = logging.getLogger(__name__)

REFUSAL_MARKERS = ("sort du cadre du cours", "hors du cadre du cours", "je suis spécialisé en")


@dataclass
class RunResult:
    """What one answer run produced.

    Attributes:
        answer: Final answer text (as shown to the student).
        latency_s: Wall-clock time of the run.
        n_model_calls: Model invocations (when known, else ``None``).
        n_tool_calls: Tool invocations (0 for the fixed pipeline).
        n_searches: Course searches performed.
        sources: Labels of the passages handed to the model.
        source_files: ``source`` metadata of those passages (legacy citation check).
        known_ids: Identifiers (``S1``...) handed to the model (agent citation check).
        steps: Human-readable description of each step (for the report).
        error: Error message when the run failed.
    """

    answer: str
    latency_s: float
    n_model_calls: int | None = None
    n_tool_calls: int = 0
    n_searches: int = 0
    sources: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    known_ids: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    error: str | None = None

    def citation_check(self) -> CitationCheck:
        """Check the citations of the answer against what the model received."""
        if self.known_ids or extract_citations(self.answer):
            return verify_citations(self.answer, self.known_ids)
        return verify_legacy_citations(self.answer, self.source_files)


class AnswerRunner(Protocol):
    """Produces an answer for one reference question."""

    async def run(self, question: EvalQuestion) -> RunResult:
        """Answer ``question`` and describe the run."""
        ...


class LegacyEngineRunner:
    """Run the frozen Phase-2 pipeline (``retrieve → grade → answer``) for the baseline.

    See :mod:`ela.eval.legacy_pipeline`.
    """

    def __init__(self, pipeline_factory: Callable[[str], object] | None = None) -> None:
        self._factory = pipeline_factory or self._default_factory

    @staticmethod
    def _default_factory(level: str):
        from ela.eval.legacy_pipeline import LegacyPipeline

        return LegacyPipeline(level)

    async def run(self, question: EvalQuestion) -> RunResult:
        """Answer ``question`` with a fresh pipeline of the question's level."""
        pipeline = self._factory(question.level)
        start = time.perf_counter()
        error = None
        answer = ""
        try:
            answer = await pipeline.answer(question.question)
        except Exception as exc:
            error = str(exc)
        latency = time.perf_counter() - start
        docs = list(getattr(pipeline, "last_docs", []) or [])
        return RunResult(
            answer=answer,
            latency_s=latency,
            n_model_calls=None,
            n_tool_calls=0,
            n_searches=1 if docs else 0,
            sources=[_doc_label(d) for d in docs],
            source_files=[str((d.metadata or {}).get("source", "")) for d in docs],
            steps=["retrieve_and_grade", "answer"],
            error=error,
        )


class AgentEngineRunner:
    """Run the agent through :class:`ela.engine.ElaEngine` and collect its events."""

    def __init__(self, engine_factory: Callable[[str], object] | None = None) -> None:
        self._factory = engine_factory or self._default_factory

    @staticmethod
    def _default_factory(level: str):
        from ela.engine import ElaEngine

        return ElaEngine(user_level=level)

    async def run(self, question: EvalQuestion) -> RunResult:
        """Answer ``question`` with a fresh engine of the question's level."""
        from ela.agent.events import ErrorEvent, FinalEvent, ToolStartEvent

        engine = self._factory(question.level)
        start = time.perf_counter()
        final: FinalEvent | None = None
        error = None
        steps: list[str] = []
        try:
            async for event in engine.ask(question.question):
                if isinstance(event, ToolStartEvent):
                    steps.append(event.label)
                elif isinstance(event, FinalEvent):
                    final = event
                elif isinstance(event, ErrorEvent):
                    error = event.message
        except Exception as exc:
            error = str(exc)
        latency = time.perf_counter() - start
        ctx = getattr(engine, "last_context", None)
        refs = list(ctx.registry) if ctx is not None else []
        return RunResult(
            answer=final.text if final is not None else "",
            latency_s=latency,
            n_model_calls=final.n_model_calls if final is not None else None,
            n_tool_calls=final.n_tool_calls if final is not None else 0,
            n_searches=final.n_searches if final is not None else (ctx.n_searches if ctx is not None else 0),
            sources=[f"{r.sid} {r.label}" for r in refs],
            source_files=[r.source for r in refs],
            known_ids=[r.sid for r in refs],
            steps=steps,
            error=error,
        )


def _doc_label(doc) -> str:
    meta = doc.metadata or {}
    where = " › ".join(p for p in (meta.get("chapter"), meta.get("section"), meta.get("slide_title")) if p)
    src = meta.get("source", "Inconnu")
    return f"{src} — {where}" if where else str(src)


def looks_like_refusal(answer: str) -> bool:
    """Heuristic refusal detector (used when no judge is available)."""
    low = answer.lower()
    return any(marker in low for marker in REFUSAL_MARKERS)


async def evaluate_answers(
    questions: list[EvalQuestion],
    runner: AnswerRunner,
    *,
    judge: BaseChatModel | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Answer every question, check citations, grade with the judge.

    Args:
        questions: Reference questions (all types).
        runner: Produces an answer per question.
        judge: Judge model, or ``None`` to skip grading.
        progress: Optional callback receiving each per-question record.

    Returns:
        Report dict with ``questions`` (per-question records), ``metrics`` and ``by_course``.
    """
    records: list[dict] = []
    for q in questions:
        result = await runner.run(q)
        check = result.citation_check()
        behavior = expected_behavior(q.type)
        judgment: Judgment | None = None
        if judge is not None and not result.error:
            try:
                judgment = await judge_answer(
                    judge, question=q.question, answer=result.answer, answer_points=q.answer_points
                )
            except Exception as exc:
                logger.warning("Judge failed on %s: %s", q.id, exc)
        refused = judgment.refused if judgment is not None else looks_like_refusal(result.answer)
        general = has_general_knowledge_tag(result.answer) or (
            judgment.acknowledges_not_in_course if judgment is not None else False
        )
        record = {
            "id": q.id,
            "level": q.level,
            "course": q.course,
            "type": q.type,
            "difficulty": q.difficulty,
            "expected_behavior": behavior,
            "question": q.question,
            "answer": result.answer,
            "error": result.error,
            "latency_s": round(result.latency_s, 3),
            "n_model_calls": result.n_model_calls,
            "n_tool_calls": result.n_tool_calls,
            "n_searches": result.n_searches,
            "steps": result.steps,
            "sources": result.sources,
            "citations_valid": check.valid,
            "citations_invalid": check.invalid,
            "phantom": not check.clean,
            "refused": refused,
            "general_knowledge": general,
            "fidelity": judgment.fidelity if judgment is not None else None,
            "judgment": judgment.model_dump() if judgment is not None else None,
        }
        records.append(record)
        if progress is not None:
            progress(record)
        await asyncio.sleep(0)  # let the event loop breathe between questions

    return {
        "n_questions": len(records),
        "metrics": summarize(records),
        "by_course": by_course(records),
        "questions": records,
    }


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize(records: list[dict]) -> dict:
    """Aggregate per-question records into the report metrics."""
    ok = [r for r in records if not r["error"]]
    latencies = [r["latency_s"] for r in ok]
    n_cit = sum(len(r["citations_valid"]) + len(r["citations_invalid"]) for r in ok)
    n_valid = sum(len(r["citations_valid"]) for r in ok)
    judged = [r["fidelity"] for r in ok if r["fidelity"] is not None]
    ood = [r for r in ok if r["expected_behavior"] == "refuse"]
    nic = [r for r in ok if r["expected_behavior"] == "general"]
    answerable = [r for r in ok if r["expected_behavior"] == "answer"]
    multihop = [r for r in ok if r["type"] == "multi-hop"]
    contradictions = [r for r in ok if r["judgment"] and r["judgment"].get("contradiction")]

    def _rate(num: int, den: int) -> float | None:
        return round(num / den, 3) if den else None

    return {
        "n": len(records),
        "n_errors": len(records) - len(ok),
        "latency_median_s": round(statistics.median(latencies), 3) if latencies else None,
        "latency_p90_s": round(_percentile(latencies, 0.9), 3) if latencies else None,
        "latency_mean_s": round(statistics.fmean(latencies), 3) if latencies else None,
        "searches_mean": round(statistics.fmean(r["n_searches"] for r in ok), 2) if ok else None,
        "tool_calls_mean": round(statistics.fmean(r["n_tool_calls"] for r in ok), 2) if ok else None,
        "citation_valid_rate": _rate(n_valid, n_cit),
        "questions_with_phantom": sum(1 for r in ok if r["phantom"]),
        "fidelity_mean": round(statistics.fmean(judged), 3) if judged else None,
        "fidelity_judged_n": len(judged),
        "contradictions": len(contradictions),
        "unwanted_refusals": sum(1 for r in answerable if r["refused"]),
        "ood_refusal_rate": _rate(sum(1 for r in ood if r["refused"]), len(ood)),
        "nic_general_rate": _rate(sum(1 for r in nic if r["general_knowledge"]), len(nic)),
        "multihop_two_searches_rate": _rate(sum(1 for r in multihop if r["n_searches"] >= 2), len(multihop)),
    }


def by_course(records: list[dict]) -> dict:
    """Per-course fidelity and latency (answerable questions only)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if not r["error"] and r["expected_behavior"] == "answer":
            groups[r["course"]].append(r)
    out = {}
    for course, rows in groups.items():
        judged = [r["fidelity"] for r in rows if r["fidelity"] is not None]
        out[course] = {
            "questions": len(rows),
            "fidelity_mean": round(statistics.fmean(judged), 3) if judged else None,
            "latency_median_s": round(statistics.median(r["latency_s"] for r in rows), 3),
            "phantom": sum(1 for r in rows if r["phantom"]),
        }
    return out
