"""Comparison of two evaluation reports (``ela-eval compare`` and the backoffice page).

Both reports must be of the same kind (``retrieval``, ``answer`` or, later,
``code``). The comparison lists the key metrics side by side with their
delta, then the questions whose outcome changed: for ``retrieval``, the
questions gained / lost (an expected slide found or not in the top-k); for
``answer``, the questions whose fidelity rose / fell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

KIND_RETRIEVAL = "retrieval"
KIND_ANSWER = "answer"
KIND_CODE = "code"
KINDS = (KIND_RETRIEVAL, KIND_ANSWER, KIND_CODE)

#: Key metrics of each kind, in display order: ``(key, label, higher_is_better)``.
METRIC_SPECS: dict[str, tuple[tuple[str, str, bool], ...]] = {
    KIND_RETRIEVAL: (
        ("hit_at_k", "hit@k", True),
        ("any_hit_at_k", "any-hit@k", True),
        ("mrr", "MRR", True),
        ("any_hit_candidates", "rappel du pool", True),
        ("mean_latency_s", "latence moyenne (s)", False),
    ),
    KIND_ANSWER: (
        ("latency_median_s", "latence médiane (s)", False),
        ("latency_p90_s", "latence p90 (s)", False),
        ("searches_mean", "recherches / question", True),
        ("tool_calls_mean", "appels d'outils / question", True),
        ("citation_valid_rate", "citations valides", True),
        ("questions_with_phantom", "questions avec citation fantôme", False),
        ("fidelity_mean", "fidélité (juge)", True),
        ("contradictions", "contradictions", False),
        ("unwanted_refusals", "refus indus", False),
        ("ood_refusal_rate", "refus hors domaine", True),
        ("nic_general_rate", "hors corpus étiqueté", True),
        ("multihop_two_searches_rate", "multi-étapes ≥ 2 recherches", True),
    ),
    KIND_CODE: (
        ("generated_rate", "script généré", True),
        ("syntax_ok_rate", "syntaxe correcte du premier coup", True),
        ("run_ok_first_rate", "exécution réussie du premier coup", True),
        ("run_ok_rate", "exécution réussie (après correction)", True),
        ("fixes_mean", "corrections / sujet", False),
        ("output_rate", "sortie chiffrée", True),
        ("figure_rate", "figure attendue produite", True),
        ("terms_rate", "termes attendus présents", True),
        ("timeouts", "délais dépassés", False),
        ("latency_generation_median_s", "latence génération médiane (s)", False),
        ("latency_run_median_s", "latence exécution médiane (s)", False),
    ),
}

#: Headline metrics shown in listings: ``(key, short label)``.
HEADLINE: dict[str, tuple[tuple[str, str], ...]] = {
    KIND_RETRIEVAL: (("any_hit_at_k", "any-hit@k"), ("mrr", "MRR"), ("mean_latency_s", "latence (s)")),
    KIND_ANSWER: (
        ("fidelity_mean", "fidélité"),
        ("questions_with_phantom", "fantômes"),
        ("latency_median_s", "latence (s)"),
    ),
    KIND_CODE: (("run_ok_rate", "exécution OK"), ("fixes_mean", "corrections"), ("output_rate", "sortie")),
}


class ReportKindError(ValueError):
    """The two reports are not of the same kind."""


def report_kind(report: dict[str, Any]) -> str:
    """Return the kind of a report (``retrieval`` / ``answer`` / ``code``).

    Reports written before lot 5.5 carry no ``kind``: a retrieval report is
    recognised by its ``hit_at_k`` key, an answer report by its ``metrics``.

    Args:
        report: Parsed JSON report.

    Returns:
        The kind.

    Raises:
        ReportKindError: When the structure matches no known kind.
    """
    kind = report.get("kind")
    if kind in KINDS:
        return str(kind)
    if "hit_at_k" in report:
        return KIND_RETRIEVAL
    if "metrics" in report:
        return KIND_ANSWER
    raise ReportKindError("rapport d'un type inconnu")


def report_metrics(report: dict[str, Any]) -> dict[str, Any]:
    """Return the flat metrics mapping of a report, whatever its kind."""
    if report_kind(report) == KIND_RETRIEVAL:
        return report
    return report.get("metrics") or {}


@dataclass(frozen=True)
class MetricRow:
    """One line of the comparison table."""

    key: str
    label: str
    before: float | None
    after: float | None
    higher_is_better: bool

    @property
    def delta(self) -> float | None:
        """``after - before`` when both values exist."""
        if self.before is None or self.after is None:
            return None
        return round(self.after - self.before, 3)

    @property
    def trend(self) -> str:
        """``better`` / ``worse`` / ``same`` / ``n/a`` for styling."""
        d = self.delta
        if d is None:
            return "n/a"
        if abs(d) < 1e-9:
            return "same"
        return "better" if (d > 0) == self.higher_is_better else "worse"


@dataclass(frozen=True)
class Comparison:
    """Outcome of :func:`compare_reports`."""

    kind: str
    label_before: str
    label_after: str
    rows: list[MetricRow]
    gained: list[str] = field(default_factory=list)  # questions improved
    lost: list[str] = field(default_factory=list)  # questions degraded
    only_before: list[str] = field(default_factory=list)  # ids present in one report only
    only_after: list[str] = field(default_factory=list)

    @property
    def change_labels(self) -> tuple[str, str]:
        """Human labels of ``gained`` / ``lost`` for the kind."""
        if self.kind == KIND_RETRIEVAL:
            return ("Questions gagnées", "Questions perdues")
        if self.kind == KIND_CODE:
            return ("Sujets désormais exécutés", "Sujets désormais en échec")
        return ("Fidélité en hausse", "Fidélité en baisse")


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _question_outcomes(kind: str, records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Per-question scalar used to detect gains and losses."""
    out: dict[str, float | None] = {}
    for r in records:
        qid = str(r.get("id"))
        if kind == KIND_RETRIEVAL:
            out[qid] = 1.0 if (r.get("found") or 0) > 0 else 0.0
        elif kind == KIND_CODE:
            out[qid] = 1.0 if r.get("run_ok") else 0.0
        else:
            out[qid] = _num(r.get("fidelity"))
    return out


def compare_reports(before: dict[str, Any], after: dict[str, Any]) -> Comparison:
    """Compare two reports of the same kind.

    Args:
        before: Reference report (parsed JSON).
        after: Report to compare against the reference.

    Returns:
        The comparison (metric rows, gained / lost questions).

    Raises:
        ReportKindError: When the reports are not of the same kind.
    """
    kind_a, kind_b = report_kind(before), report_kind(after)
    if kind_a != kind_b:
        raise ReportKindError(f"les deux rapports ne sont pas du même type ({kind_a} / {kind_b})")
    ma, mb = report_metrics(before), report_metrics(after)
    rows = [
        MetricRow(key, label, _num(ma.get(key)), _num(mb.get(key)), hib)
        for key, label, hib in METRIC_SPECS[kind_a]
        if key in ma or key in mb
    ]
    qa = _question_outcomes(kind_a, before.get("questions") or [])
    qb = _question_outcomes(kind_a, after.get("questions") or [])
    common = [i for i in qb if i in qa]
    gained = [i for i in common if qa[i] is not None and qb[i] is not None and qb[i] > qa[i]]
    lost = [i for i in common if qa[i] is not None and qb[i] is not None and qb[i] < qa[i]]
    return Comparison(
        kind=kind_a,
        label_before=str(before.get("label") or "avant"),
        label_after=str(after.get("label") or "après"),
        rows=rows,
        gained=gained,
        lost=lost,
        only_before=[i for i in qa if i not in qb],
        only_after=[i for i in qb if i not in qa],
    )


def format_comparison(cmp: Comparison) -> str:
    """Plain-text rendering used by the command line."""
    lines = [f"\n{'métrique':<34}{cmp.label_before:>18}{cmp.label_after:>18}{'delta':>10}"]
    for row in cmp.rows:
        if row.before is not None and row.after is not None:
            lines.append(f"{row.label:<34}{row.before:>18.3f}{row.after:>18.3f}{row.delta:>+10.3f}")
        elif row.after is not None:
            lines.append(f"{row.label:<34}{'—':>18}{row.after:>18.3f}")
        elif row.before is not None:
            lines.append(f"{row.label:<34}{row.before:>18.3f}{'—':>18}")
    up, down = cmp.change_labels
    lines.append(f"\n{up} : {cmp.gained}\n{down} : {cmp.lost}")
    if cmp.only_before or cmp.only_after:
        lines.append(f"Seulement dans {cmp.label_before} : {cmp.only_before}")
        lines.append(f"Seulement dans {cmp.label_after} : {cmp.only_after}")
    return "\n".join(lines)
