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
configurations (before/after) can be compared with ``ela-eval compare``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ela.eval.dataset import DEFAULT_PATH, EvalQuestion, load_dataset

REPORTS_DIR = Path("eval") / "reports"


def evaluate_retrieval(questions: list[EvalQuestion], k: int) -> dict:
    """Run the retriever on each question and compute per-question hits.

    Args:
        questions: Questions that expect retrieval (``expected_sources`` non empty).
        k: Number of top results considered.

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
        per_question.append(
            {
                "id": q.id,
                "level": q.level,
                "course": q.course,
                "type": q.type,
                "difficulty": q.difficulty,
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
        )

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


def cmd_retrieval(args: argparse.Namespace) -> None:
    """Run the retrieval evaluation and save the report."""
    from ela.log import configure_logging

    configure_logging()
    ds = load_dataset(args.dataset)
    questions = [q for q in ds.questions if q.expects_retrieval]
    if args.course:
        questions = [q for q in questions if q.course == args.course]
    report = evaluate_retrieval(questions, k=args.k)
    report["label"] = args.label
    report["dataset_version"] = ds.version
    report["timestamp"] = datetime.now().isoformat(timespec="seconds")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{args.label}-retrieval.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_report(report)
    print(f"\n  Rapport : {out}")


def get_settings_prompt_version() -> str:
    """Prompt version recorded in the report."""
    from ela.config import get_settings

    return get_settings().prompt_version


def _judge_model():
    """Judge chat model from ``EVAL_JUDGE_MODEL`` (defaults to the chat model)."""
    from ela.config import get_settings
    from ela.llm import build_chat_model, get_chat_model

    s = get_settings()
    effort = s.eval_judge_reasoning_effort
    if s.eval_judge_model:
        return build_chat_model(s.eval_judge_model, reasoning_effort=effort, streaming=False)
    return get_chat_model("chat", reasoning_effort=effort, streaming=False)


def _answer_runner(engine: str):
    """Runner for the agent (default) or the frozen Phase-2 pipeline (baseline)."""
    from ela.eval.answer import AgentEngineRunner, LegacyEngineRunner

    return LegacyEngineRunner() if engine == "legacy" else AgentEngineRunner()


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

    from ela.eval.answer import evaluate_answers
    from ela.log import configure_logging

    configure_logging()
    ds = load_dataset(args.dataset)
    questions = list(ds.questions)
    if args.course:
        questions = [q for q in questions if q.course == args.course]
    if args.type:
        questions = [q for q in questions if q.type == args.type]
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        questions = [q for q in questions if q.id in wanted]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        sys.exit("❌ Aucune question sélectionnée.")

    judge = None if args.no_judge else _judge_model()

    def _progress(r: dict) -> None:
        fid = "—" if r["fidelity"] is None else f"{r['fidelity']:.2f}"
        flag = " ⚠ fantôme" if r["phantom"] else ""
        err = f" ❌ {r['error']}" if r["error"] else ""
        print(f"  {r['id']:<8} {r['latency_s']:>6.2f}s  recherches={r['n_searches']}  fidélité={fid}{flag}{err}")

    print(f"\n{len(questions)} questions, moteur {args.engine}, juge {'désactivé' if judge is None else 'activé'}\n")
    report = asyncio.run(evaluate_answers(questions, _answer_runner(args.engine), judge=judge, progress=_progress))
    from ela.tracing import flush

    flush()
    report["label"] = args.label
    report["engine"] = args.engine
    report["prompt_version"] = get_settings_prompt_version()
    report["dataset_version"] = ds.version
    report["timestamp"] = datetime.now().isoformat(timespec="seconds")
    report["kind"] = "answer"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{args.label}-answer.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_answer_report(report)
    print(f"\n  Rapport : {out}")


_RETRIEVAL_KEYS = ("hit_at_k", "any_hit_at_k", "mrr", "any_hit_candidates", "mean_latency_s")
_ANSWER_KEYS = (
    "latency_median_s",
    "latency_p90_s",
    "searches_mean",
    "tool_calls_mean",
    "citation_valid_rate",
    "questions_with_phantom",
    "fidelity_mean",
    "contradictions",
    "unwanted_refusals",
    "ood_refusal_rate",
    "nic_general_rate",
    "multihop_two_searches_rate",
)


def _metrics(report: dict) -> tuple[dict, tuple[str, ...]]:
    if report.get("kind") == "answer" or "metrics" in report:
        return report["metrics"], _ANSWER_KEYS
    return report, _RETRIEVAL_KEYS


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two saved reports (retrieval or answer)."""
    a = json.loads(Path(args.before).read_text(encoding="utf-8"))
    b = json.loads(Path(args.after).read_text(encoding="utf-8"))
    la, lb = a.get("label", "before"), b.get("label", "after")
    ma, keys = _metrics(a)
    mb, keys_b = _metrics(b)
    if keys != keys_b:
        sys.exit("❌ Les deux rapports ne sont pas du même type (retrieval / answer).")
    print(f"\n{'métrique':<28}{la:>18}{lb:>18}{'delta':>10}")
    for key in keys:
        va, vb = ma.get(key), mb.get(key)
        if va is not None and vb is not None:
            print(f"{key:<28}{va:>18.3f}{vb:>18.3f}{vb - va:>+10.3f}")
        elif vb is not None:
            print(f"{key:<28}{'—':>18}{vb:>18.3f}")
    qa = {r["id"]: r for r in a["questions"]}
    qb = {r["id"]: r for r in b["questions"]}
    if keys is _RETRIEVAL_KEYS:
        gained = [i for i in qb if qb[i]["found"] > 0 and qa.get(i, {}).get("found", 0) == 0]
        lost = [i for i in qa if qa[i]["found"] > 0 and qb.get(i, {}).get("found", 0) == 0]
        print(f"\nQuestions gagnées : {gained}\nQuestions perdues : {lost}")
    else:
        pairs = {i: (_fid(qa.get(i)), _fid(qb[i])) for i in qb}
        better = [i for i, (fa, fb) in pairs.items() if fa is not None and fb is not None and fb > fa]
        worse = [i for i, (fa, fb) in pairs.items() if fa is not None and fb is not None and fb < fa]
        print(f"\nFidélité en hausse : {better}\nFidélité en baisse : {worse}")


def _fid(record: dict | None) -> float | None:
    return None if not record else record.get("fidelity")


def main() -> None:
    """Console entry point (``ela-eval``)."""
    parser = argparse.ArgumentParser(description="ELA AI — évaluation du RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ret = sub.add_parser("retrieval", help="Mesurer hit@k / MRR du retriever sur le jeu de questions")
    p_ret.add_argument("--dataset", default=str(DEFAULT_PATH))
    p_ret.add_argument("--k", type=int, default=5)
    p_ret.add_argument("--label", default="run", help="Étiquette du rapport (ex : before, bge-m3)")
    p_ret.add_argument("--course", default=None, help="Limiter à un cours")
    p_ret.set_defaults(func=cmd_retrieval)

    p_ans = sub.add_parser("answer", help="Évaluation de bout en bout : réponses, citations, latence, juge LLM")
    p_ans.add_argument("--dataset", default=str(DEFAULT_PATH))
    p_ans.add_argument("--label", default="run", help="Étiquette du rapport (ex : before, agent)")
    p_ans.add_argument("--course", default=None, help="Limiter à un cours")
    p_ans.add_argument("--type", default=None, help="Limiter à un type de question (multi-hop, out-of-domain...)")
    p_ans.add_argument("--ids", default=None, help="Identifiants séparés par des virgules (UTS-01,MTS-05)")
    p_ans.add_argument("--limit", type=int, default=0, help="N premières questions seulement")
    p_ans.add_argument("--no-judge", action="store_true", help="Ne pas appeler le juge LLM")
    p_ans.add_argument(
        "--engine",
        choices=["agent", "legacy"],
        default="agent",
        help="agent (Phase 3) ou legacy (chaîne fixe de la Phase 2, pour la mesure « avant »)",
    )
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
