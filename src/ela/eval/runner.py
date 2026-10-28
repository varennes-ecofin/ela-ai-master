"""Evaluation runner (``ela-eval``).

``ela-eval retrieval`` runs every question of the reference set through
the level-scoped retriever and reports retrieval metrics:

* **hit@k** — share of expected slides found in the top-k results,
* **any-hit@k** — share of questions with at least one expected slide in the top-k,
* **MRR** — mean reciprocal rank of the first expected slide.

``ela-eval answer`` runs the whole conversation engine on every question
(see :mod:`ela.eval.answer`): latency, searches, citation control and an
LLM judge grading the answers against the teacher's ``answer_points``.

Results are printed and saved as JSON under ``eval/reports/`` so that two
configurations (before/after) can be compared with ``ela-eval compare``
or from the backoffice. Every run prints one ``[i/n] ID …`` line per
question (progress of the background job) and can write a small summary
of the run with ``--result`` (consumed by the job runner).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from ela.eval.compare import (
    METRIC_SPECS,
    ReportKindError,
    compare_reports,
    format_comparison,
    report_kind,
    report_metrics,
)
from ela.eval.dataset import EvalQuestion, load_dataset
from ela.eval.reports import ReportNameError, report_name


def progress_line(index: int, total: int, question_id: str, detail: str = "") -> str:
    """Format the per-question progress line (``[12/71] UTS-12 …``) parsed by the job runner."""
    return f"[{index}/{total}] {question_id:<8} {detail}".rstrip()


def evaluate_retrieval(
    questions: list[EvalQuestion], k: int, *, progress: Callable[[dict], None] | None = None
) -> dict:
    """Run the retriever on each question and compute per-question hits.

    Args:
        questions: Questions that expect retrieval (``expected_sources`` non empty).
        k: Number of top results considered.
        progress: Optional callback receiving each per-question record.

    Returns:
        A report dict with per-question details and aggregate metrics.
    """
    from ela.retrieval import get_retriever  # heavy imports (models) kept out of module import

    per_question = []
    for q in questions:
        start = time.perf_counter()
        retriever = get_retriever(q.level)
        docs = retriever.invoke(q.question)[:k]
        latency = time.perf_counter() - start
        hits = q.hits(docs)
        rank = q.first_hit_rank(docs)
        # Diagnostics: was an expected slide in the candidate pool before reranking?
        candidates = [h.doc for h in getattr(retriever, "last_hits", [])]
        candidate_rank = q.first_hit_rank(candidates)
        record = {
            "id": q.id,
            "level": q.level,
            "course": q.course,
            "type": q.type,
            "difficulty": q.difficulty,
            "question": q.question,
            "expected": len(hits),
            "found": sum(hits),
            "first_rank": rank,
            "candidate_rank": candidate_rank,
            "n_candidates": len(candidates),
            "latency_s": round(latency, 3),
            "retrieved": [
                {"source": d.metadata.get("source"), "slide_title": d.metadata.get("slide_title")} for d in docs
            ],
        }
        per_question.append(record)
        if progress is not None:
            progress(record)

    total_expected = sum(r["expected"] for r in per_question)
    total_found = sum(r["found"] for r in per_question)
    any_hit = sum(1 for r in per_question if r["found"] > 0)
    mrr = sum(1.0 / r["first_rank"] for r in per_question if r["first_rank"]) / max(len(per_question), 1)
    in_pool = sum(1 for r in per_question if r["candidate_rank"])

    by_course: dict[str, dict] = defaultdict(lambda: {"questions": 0, "any_hit": 0, "expected": 0, "found": 0})
    for r in per_question:
        c = by_course[r["course"]]
        c["questions"] += 1
        c["any_hit"] += int(r["found"] > 0)
        c["expected"] += r["expected"]
        c["found"] += r["found"]

    return {
        "k": k,
        "n_questions": len(per_question),
        "hit_at_k": round(total_found / max(total_expected, 1), 3),
        "any_hit_at_k": round(any_hit / max(len(per_question), 1), 3),
        "mrr": round(mrr, 3),
        "any_hit_candidates": round(in_pool / max(len(per_question), 1), 3),
        "mean_latency_s": round(sum(r["latency_s"] for r in per_question) / max(len(per_question), 1), 3),
        "by_course": {c: {**v, "any_hit_rate": round(v["any_hit"] / v["questions"], 3)} for c, v in by_course.items()},
        "questions": per_question,
    }


def _print_report(report: dict) -> None:
    print(f"\nRetrieval — k={report['k']} — {report['n_questions']} questions")
    print(f"  hit@k      : {report['hit_at_k']:.3f}   (slides attendues retrouvées)")
    print(f"  any-hit@k  : {report['any_hit_at_k']:.3f}   (questions avec au moins une slide attendue)")
    print(f"  MRR        : {report['mrr']:.3f}")
    if "any_hit_candidates" in report:
        print(
            f"  rappel pool: {report['any_hit_candidates']:.3f}   "
            "(slide attendue présente parmi les candidats AVANT reranking)"
        )
    print(f"  latence    : {report['mean_latency_s']:.2f} s / question\n")
    print(f"  {'cours':<28}{'questions':>10}{'any-hit':>10}{'hit@k':>10}")
    for course, v in sorted(report["by_course"].items()):
        hit = v["found"] / max(v["expected"], 1)
        print(f"  {course:<28}{v['questions']:>10}{v['any_hit_rate']:>10.2f}{hit:>10.2f}")
    misses = [r for r in report["questions"] if r["found"] == 0]
    if misses:
        print("\n  Questions sans aucune slide attendue dans le top-k :")
        for r in misses:
            got = ", ".join(f"{d['slide_title']}" for d in r["retrieved"][:3] if d["slide_title"])
            pool = f"candidat #{r['candidate_rank']}" if r.get("candidate_rank") else "absente du pool"
            print(f"   - {r['id']:<8} [{pool}] → {got[:90]}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def run_metadata(kind: str, label: str, dataset_version: int) -> dict[str, Any]:
    """Provenance recorded in every report (model, provider, prompt and dataset versions)."""
    from ela.config import get_settings

    s = get_settings()
    return {
        "kind": kind,
        "label": label,
        "model": s.llm_model_chat,
        "provider": s.llm_provider,
        "prompt_version": s.prompt_version,
        "dataset_version": dataset_version,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def select_questions(
    questions: list[EvalQuestion],
    *,
    course: str | None = None,
    qtype: str | None = None,
    ids: str | None = None,
    limit: int = 0,
) -> list[EvalQuestion]:
    """Apply the ``--course``, ``--type``, ``--ids`` and ``--limit`` filters."""
    selected = list(questions)
    if course:
        selected = [q for q in selected if q.course == course]
    if qtype:
        selected = [q for q in selected if q.type == qtype]
    if ids:
        wanted = {i.strip() for i in ids.split(",") if i.strip()}
        selected = [q for q in selected if q.id in wanted]
    if limit and limit > 0:
        selected = selected[:limit]
    return selected


def reports_directory() -> Path:
    """Return the reports directory from the settings (``EVAL_REPORTS_DIR``)."""
    from ela.config import get_settings

    return Path(get_settings().eval_reports_dir)


def dataset_path(arg: str | None) -> Path:
    """Dataset path: ``--dataset`` when given, else ``EVAL_DATASET`` from the settings."""
    if arg:
        return Path(arg)
    from ela.config import get_settings

    return Path(get_settings().eval_dataset)


def save_report(report: dict, label: str, kind: str, reports_dir: Path | None = None) -> Path:
    """Write the report JSON under the reports directory and return its path.

    Raises:
        ReportNameError: Invalid label.
    """
    directory = reports_dir or reports_directory()
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / report_name(label, kind)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def result_summary(report: dict, path: Path) -> dict[str, Any]:
    """Small JSON written with ``--result`` (job result): report name and key metrics."""
    kind = report_kind(report)
    metrics = report_metrics(report)
    keys = [key for key, _, _ in METRIC_SPECS[kind]]
    return {
        "report": path.name,
        "kind": kind,
        "label": report.get("label"),
        "model": report.get("model"),
        "provider": report.get("provider"),
        "n_questions": report.get("n_questions"),
        "n_errors": metrics.get("n_errors", 0),
        "metrics": {k: metrics.get(k) for k in keys if k in metrics},
    }


def write_result(args: argparse.Namespace, report: dict, path: Path) -> None:
    """Write the ``--result`` summary when requested."""
    if getattr(args, "result", None):
        Path(args.result).write_text(
            json.dumps(result_summary(report, path), ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _check_label(label: str) -> str:
    try:
        report_name(label, "x")
    except ReportNameError as exc:
        sys.exit(f"❌ {exc}")
    return label


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_retrieval(args: argparse.Namespace) -> None:
    """Run the retrieval evaluation and save the report."""
    from ela.log import configure_logging

    configure_logging()
    label = _check_label(args.label)
    ds = load_dataset(dataset_path(args.dataset))
    questions = select_questions(
        [q for q in ds.questions if q.expects_retrieval], course=args.course, qtype=args.type, limit=args.limit
    )
    if not questions:
        sys.exit("❌ Aucune question sélectionnée.")
    total = len(questions)
    print(f"\n{total} questions, k={args.k}\n")

    done: list[str] = []

    def _progress(r: dict) -> None:
        rank = f"rang {r['first_rank']}" if r["first_rank"] else "non trouvée"
        done.append(r["id"])
        print(progress_line(len(done), total, r["id"], f"{r['found']}/{r['expected']}  {rank}"), flush=True)

    report = evaluate_retrieval(questions, k=args.k, progress=_progress)
    report.update(run_metadata("retrieval", label, ds.version))
    out = save_report(report, label, "retrieval")
    _print_report(report)
    print(f"\n  Rapport : {out}")
    write_result(args, report, out)


def _judge_model():
    """Judge chat model: ``EVAL_JUDGE_MODEL`` on ``EVAL_JUDGE_PROVIDER`` (defaults: chat model, chat provider).

    The judge is built directly (no fallback) so that it never follows a
    model override of the evaluated engine.
    """
    from ela.config import get_settings
    from ela.llm import build_chat_model

    s = get_settings()
    effort = s.eval_judge_reasoning_effort
    model = s.eval_judge_model or s.llm_model_chat
    provider = s.eval_judge_provider or s.llm_provider
    return build_chat_model(model, provider, reasoning_effort=effort, streaming=False), model, provider


def _print_answer_report(report: dict) -> None:
    m = report["metrics"]

    def f(v, fmt="{:.3f}"):
        return "—" if v is None else fmt.format(v)

    print(f"\nRéponses — {report['n_questions']} questions ({m['n_errors']} erreurs)")
    print(f"  latence médiane : {f(m['latency_median_s'], '{:.2f}')} s   p90 : {f(m['latency_p90_s'], '{:.2f}')} s")
    print(
        f"  recherches/q    : {f(m['searches_mean'], '{:.2f}')}   "
        f"appels d'outils/q : {f(m['tool_calls_mean'], '{:.2f}')}"
    )
    print(
        f"  citations valides : {f(m['citation_valid_rate'])}   "
        f"questions avec citation fantôme : {m['questions_with_phantom']}"
    )
    print(
        f"  fidélité (juge) : {f(m['fidelity_mean'])} sur {m['fidelity_judged_n']} questions   "
        f"contradictions : {m['contradictions']}"
    )
    print(
        f"  refus hors domaine : {f(m['ood_refusal_rate'], '{:.2f}')}   "
        f"hors corpus étiqueté : {f(m['nic_general_rate'], '{:.2f}')}   "
        f"refus indus : {m['unwanted_refusals']}   "
        f"multi-étapes ≥ 2 recherches : {f(m['multihop_two_searches_rate'], '{:.2f}')}"
    )
    print(f"\n  {'cours':<28}{'questions':>10}{'fidélité':>10}{'latence':>10}{'fantômes':>10}")
    for course, v in sorted(report["by_course"].items()):
        print(
            f"  {course:<28}{v['questions']:>10}{f(v['fidelity_mean'], '{:.2f}'):>10}"
            f"{f(v['latency_median_s'], '{:.2f}'):>10}{v['phantom']:>10}"
        )
    flagged = [
        r
        for r in report["questions"]
        if r["phantom"] or r["error"] or (r["fidelity"] is not None and r["fidelity"] < 0.5)
    ]
    if flagged:
        print("\n  Questions à regarder (citation fantôme, erreur ou fidélité < 0,5) :")
        for r in flagged:
            phantom = "citation fantôme " + ", ".join(r["citations_invalid"]) if r["phantom"] else ""
            why = "erreur" if r["error"] else phantom
            fid = "" if r["fidelity"] is None else f"fidélité {r['fidelity']:.2f}"
            print(f"   - {r['id']:<8} {why} {fid}".rstrip())


def cmd_answer(args: argparse.Namespace) -> None:
    """Run the end-to-end evaluation and save the report."""
    import asyncio

    from ela.eval.answer import AgentEngineRunner, evaluate_answers
    from ela.log import configure_logging

    configure_logging()
    label = _check_label(args.label)
    ds = load_dataset(dataset_path(args.dataset))
    questions = select_questions(
        list(ds.questions), course=args.course, qtype=args.type, ids=args.ids, limit=args.limit
    )
    if not questions:
        sys.exit("❌ Aucune question sélectionnée.")

    judge = judge_model = judge_provider = None
    if not args.no_judge:
        judge, judge_model, judge_provider = _judge_model()
    total = len(questions)
    done: list[str] = []

    def _progress(r: dict) -> None:
        fid = "—" if r["fidelity"] is None else f"{r['fidelity']:.2f}"
        flag = " ⚠ fantôme" if r["phantom"] else ""
        err = f" ❌ {r['error']}" if r["error"] else ""
        done.append(r["id"])
        detail = f"{r['latency_s']:>6.2f}s  recherches={r['n_searches']}  fidélité={fid}{flag}{err}"
        print(progress_line(len(done), total, r["id"], detail), flush=True)

    judge_label = "désactivé" if judge is None else f"{judge_model} ({judge_provider})"
    print(f"\n{total} questions, juge {judge_label}\n")
    report = asyncio.run(evaluate_answers(questions, AgentEngineRunner(), judge=judge, progress=_progress))
    from ela.tracing import flush

    flush()
    report.update(run_metadata("answer", label, ds.version))
    report["judge_model"] = judge_model
    report["judge_provider"] = judge_provider
    out = save_report(report, label, "answer")
    _print_answer_report(report)
    print(f"\n  Rapport : {out}")
    write_result(args, report, out)


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two saved reports (retrieval or answer)."""
    a = json.loads(Path(args.before).read_text(encoding="utf-8"))
    b = json.loads(Path(args.after).read_text(encoding="utf-8"))
    try:
        print(format_comparison(compare_reports(a, b)))
    except ReportKindError as exc:
        sys.exit(f"❌ {exc}")


def main() -> None:
    """Console entry point (``ela-eval``)."""
    parser = argparse.ArgumentParser(description="ELA AI — évaluation du RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ret = sub.add_parser("retrieval", help="Mesurer hit@k / MRR du retriever sur le jeu de questions")
    p_ret.add_argument("--dataset", default=None, help="Jeu de questions (défaut : EVAL_DATASET)")
    p_ret.add_argument("--k", type=int, default=5)
    p_ret.add_argument("--label", default="run", help="Étiquette du rapport (ex : before, bge-m3)")
    p_ret.add_argument("--course", default=None, help="Limiter à un cours")
    p_ret.add_argument("--type", default=None, help="Limiter à un type de question (definition, formula...)")
    p_ret.add_argument("--limit", type=int, default=0, help="N premières questions seulement")
    p_ret.add_argument("--result", default=None, help="Écrire un résumé JSON du run (utilisé par le backoffice)")
    p_ret.set_defaults(func=cmd_retrieval)

    p_ans = sub.add_parser("answer", help="Évaluation de bout en bout : réponses, citations, latence, juge LLM")
    p_ans.add_argument("--dataset", default=None, help="Jeu de questions (défaut : EVAL_DATASET)")
    p_ans.add_argument("--label", default="run", help="Étiquette du rapport (ex : before, agent)")
    p_ans.add_argument("--course", default=None, help="Limiter à un cours")
    p_ans.add_argument("--type", default=None, help="Limiter à un type de question (multi-hop, out-of-domain...)")
    p_ans.add_argument("--ids", default=None, help="Identifiants séparés par des virgules (UTS-01,MTS-05)")
    p_ans.add_argument("--limit", type=int, default=0, help="N premières questions seulement")
    p_ans.add_argument("--no-judge", action="store_true", help="Ne pas appeler le juge LLM")
    p_ans.add_argument("--result", default=None, help="Écrire un résumé JSON du run (utilisé par le backoffice)")
    p_ans.set_defaults(func=cmd_answer)

    p_cmp = sub.add_parser("compare", help="Comparer deux rapports JSON (retrieval ou answer)")
    p_cmp.add_argument("before")
    p_cmp.add_argument("after")
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    try:
        args.func(args)
    except FileNotFoundError as exc:
        sys.exit(f"❌ {exc}")


if __name__ == "__main__":
    main()
