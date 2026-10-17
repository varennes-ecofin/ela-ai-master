"""Evaluation runner (``ela-eval``).

``ela-eval retrieval`` runs every question of the reference set through
the level-scoped retriever and reports retrieval metrics:

* **hit@k** — share of expected slides found in the top-k results,
* **any-hit@k** — share of questions with at least one expected slide in the top-k,
* **MRR** — mean reciprocal rank of the first expected slide.

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


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two saved retrieval reports."""
    a = json.loads(Path(args.before).read_text(encoding="utf-8"))
    b = json.loads(Path(args.after).read_text(encoding="utf-8"))
    la, lb = a.get("label", "before"), b.get("label", "after")
    print(f"\n{'métrique':<20}{la:>18}{lb:>18}{'delta':>10}")
    for key in ("hit_at_k", "any_hit_at_k", "mrr", "any_hit_candidates", "mean_latency_s"):
        if key in a and key in b:
            print(f"{key:<20}{a[key]:>18.3f}{b[key]:>18.3f}{b[key] - a[key]:>+10.3f}")
        elif key in b:
            print(f"{key:<20}{'—':>18}{b[key]:>18.3f}")
    qa = {r["id"]: r for r in a["questions"]}
    qb = {r["id"]: r for r in b["questions"]}
    gained = [i for i in qb if qb[i]["found"] > 0 and qa.get(i, {}).get("found", 0) == 0]
    lost = [i for i in qa if qa[i]["found"] > 0 and qb.get(i, {}).get("found", 0) == 0]
    print(f"\nQuestions gagnées : {gained}\nQuestions perdues : {lost}")


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

    p_cmp = sub.add_parser("compare", help="Comparer deux rapports JSON")
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
