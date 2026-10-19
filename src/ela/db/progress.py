"""Student report (``/bilan``): what was worked, how it went, what to do next.

Built on the fly from ``student_activity``, ``quiz_results`` and
``exercise_attempts`` — no derived table to keep in sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ela.db.activity import activity_counts, chapters_worked
from ela.db.exercises import exercise_summary
from ela.db.quiz import QuizStats, user_stats

WEAK_RATE = 0.6  # a topic is weak below this success rate (with at least 2 answers)


@dataclass(frozen=True)
class Suggestion:
    """A next step offered to the student (rendered as a button)."""

    kind: str  # "quiz" | "exercise"
    topic: str
    label: str


@dataclass
class StudentReport:
    """Aggregated view of one student's work."""

    identifier: str
    activity: dict
    chapters: list[dict]
    quiz: QuizStats
    exercises: list[dict]
    suggestions: list[Suggestion] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        """``True`` when the student has no recorded activity."""
        return not (self.activity.get("questions") or self.quiz.n_answers or self.exercises)


def build_suggestions(report: StudentReport, *, limit: int = 3) -> list[Suggestion]:
    """Weak quiz topics first, then unfinished exercises, then chapters asked about but never quizzed."""
    out: list[Suggestion] = []
    quizzed = {t["topic"].lower() for t in report.quiz.by_topic}
    for t in report.quiz.by_topic:
        if t["n"] >= 2 and t["rate"] < WEAK_RATE:
            out.append(Suggestion("quiz", t["topic"], f"🎓 Quiz : {t['topic']}"))
    for ex in report.exercises:
        if not ex.get("completed"):
            out.append(Suggestion("exercise", ex["topic"], f"✏️ Reprendre : {ex['topic']}"))
    for ch in report.chapters:
        title = ch.get("chapter") or ch.get("course") or ""
        if title and not ch.get("quizzes") and title.lower() not in quizzed:
            out.append(Suggestion("quiz", title, f"🎓 Me tester : {title}"))
    seen: set[tuple[str, str]] = set()
    unique = []
    for s in out:
        key = (s.kind, s.topic.lower())
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique[:limit]


async def student_report(identifier: str) -> StudentReport:
    """Collect the report of one student."""
    report = StudentReport(
        identifier=identifier,
        activity=await activity_counts(identifier),
        chapters=await chapters_worked(identifier),
        quiz=await user_stats(identifier),
        exercises=await exercise_summary(identifier),
    )
    report.suggestions = build_suggestions(report)
    return report


def render_report(report: StudentReport) -> str:
    """Markdown shown in the chat for ``/bilan``."""
    if report.empty:
        return (
            "📈 **Mon bilan**\n\nRien à montrer pour l'instant : posez des questions, faites un quiz "
            "(`/start_quiz`) ou un exercice (`/exercise`) et revenez ici."
        )
    a = report.activity
    lines = ["📈 **Mon bilan**", ""]
    lines.append(
        f"**Questions posées** : {a['questions']} (dans {a['threads']} conversation(s))"
        + (f", dont {a['general']} hors du cours" if a.get("general") else "")
        + "."
    )
    if report.chapters:
        lines.append("")
        lines.append("**Chapitres travaillés**")
        for ch in report.chapters[:8]:
            where = " › ".join(p for p in (ch.get("course"), ch.get("chapter")) if p)
            counts = []
            if ch.get("questions"):
                counts.append(f"{ch['questions']} question(s)")
            if ch.get("quizzes"):
                counts.append(f"{ch['quizzes']} quiz")
            lines.append(f"- {where} — {', '.join(counts) or 'travaillé'}")
    q = report.quiz
    lines.append("")
    if q.n_answers:
        lines.append(
            f"**Quiz** : {q.n_quizzes} quiz, {q.n_correct}/{q.n_answers} bonnes réponses "
            f"({round(100 * (q.rate or 0))} %)."
        )
        weak = [t for t in q.by_topic if t["n"] >= 2 and t["rate"] < WEAK_RATE]
        strong = [t for t in q.by_topic if t["n"] >= 2 and t["rate"] >= 0.8]
        if weak:
            lines.append("- Points faibles : " + ", ".join(f"{t['topic']} ({round(100 * t['rate'])} %)" for t in weak))
        if strong:
            lines.append("- Points forts : " + ", ".join(f"{t['topic']} ({round(100 * t['rate'])} %)" for t in strong))
    else:
        lines.append("**Quiz** : aucun pour l'instant (`/start_quiz`).")
    lines.append("")
    if report.exercises:
        done = sum(1 for e in report.exercises if e.get("completed"))
        lines.append(f"**Exercices** : {len(report.exercises)} commencé(s), {done} terminé(s).")
        for ex in report.exercises[:5]:
            status = "✅ terminé" if ex.get("completed") else f"⏳ {ex['attempts']} tentative(s)"
            lines.append(f"- {ex.get('title') or ex['topic']} — {status}")
    else:
        lines.append("**Exercices** : aucun pour l'instant (`/exercise`).")
    suggestions = report.suggestions or build_suggestions(report)
    if suggestions:
        lines.append("")
        lines.append("**Et maintenant ?** " + " · ".join(s.label for s in suggestions))
    return "\n".join(lines)
