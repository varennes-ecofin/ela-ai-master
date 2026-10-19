"""Pydantic schemas of the pedagogical objects returned by the tools."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Difficulty = Literal["facile", "moyen", "difficile"]


class QuizQuestion(BaseModel):
    """One multiple-choice question.

    LaTeX (``$...$``) is allowed everywhere: the options are rendered in the
    message body (KaTeX) and the answer buttons only carry a letter.
    """

    question: str = Field(description="Énoncé de la question (LaTeX $...$ autorisé).")
    options: list[str] = Field(
        min_length=2,
        max_length=4,
        description="2 à 4 propositions courtes, sans lettre de préfixe (LaTeX $...$ autorisé).",
    )
    correct_index: int = Field(ge=0, description="Indice (0-based) de la bonne proposition.")
    explanation: str = Field(description="Explication de la bonne réponse (LaTeX $...$ autorisé).")

    @model_validator(mode="after")
    def _index_in_range(self) -> QuizQuestion:
        if self.correct_index >= len(self.options):
            raise ValueError("correct_index out of range")
        return self


class Quiz(BaseModel):
    """A quiz grounded in the course."""

    topic: str
    questions: list[QuizQuestion] = Field(min_length=1)
    sources: list[str] = Field(default_factory=list, description="Identifiants S1, S2... des extraits utilisés.")
    quiz_id: str = Field(default_factory=lambda: uuid.uuid4().hex, description="Identifiant du quiz (persistance).")
    difficulty: Difficulty = "moyen"
    source_labels: list[str] = Field(default_factory=list, description="Cours › chapitre › slide des extraits.")
    course: str | None = Field(default=None, description="Cours dominant des extraits.")
    chapter: str | None = Field(default=None, description="Chapitre dominant des extraits.")

    def to_legacy(self) -> list[dict]:
        """Shape used by the Chainlit quiz state machine (Phase 1/2)."""
        return [q.model_dump() for q in self.questions]


class Exercise(BaseModel):
    """An exercise with progressive hints and a worked solution."""

    exercise_id: str = Field(default_factory=lambda: uuid.uuid4().hex, description="Identifiant (persistance).")
    topic: str = Field(default="", description="Sujet demandé par l'étudiant.")
    title: str = Field(description="Titre court de l'exercice.")
    statement: str = Field(description="Énoncé complet (LaTeX $...$ autorisé), données incluses.")
    hints: list[str] = Field(min_length=1, max_length=3, description="1 à 3 indices, du plus vague au plus précis.")
    solution: str = Field(description="Corrigé détaillé, étape par étape (LaTeX $...$ autorisé).")
    sources: list[str] = Field(default_factory=list, description="Identifiants S1, S2... des extraits utilisés.")


class QuizPayload(BaseModel):
    """What the model returns when asked for a quiz (sources are added by the tool)."""

    questions: list[QuizQuestion] = Field(min_length=1)


class ExercisePayload(BaseModel):
    """What the model returns when asked for an exercise (sources are added by the tool)."""

    title: str
    statement: str
    hints: list[str] = Field(min_length=1, max_length=3)
    solution: str


Verdict = Literal["correct", "partiel", "incorrect", "hors_sujet"]


class StepFeedback(BaseModel):
    """Socratic feedback on one step written by the student (never reveals the next steps)."""

    verdict: Verdict = Field(description="correct | partiel | incorrect | hors_sujet")
    feedback: str = Field(description="2 à 4 phrases : ce qui est juste, ce qui ne l'est pas, sans donner la suite.")
    guiding_question: str = Field(
        default="",
        description="Une question qui oriente l'étudiant vers l'étape suivante (vide si l'exercice est terminé).",
    )
    exercise_completed: bool = Field(
        default=False, description="Vrai si l'étudiant a atteint le résultat final attendu."
    )


class ReviewSection(BaseModel):
    """One section of a review sheet (follows a section or a group of slides of the chapter)."""

    heading: str = Field(description="Titre court de la section.")
    key_points: list[str] = Field(min_length=1, max_length=8, description="Idées essentielles, une phrase chacune.")
    definitions: list[str] = Field(default_factory=list, description="« Terme : définition » (LaTeX $...$ autorisé).")
    formulas: list[str] = Field(
        default_factory=list, description="Formules essentielles en LaTeX $...$, avec leur signification en un mot."
    )
    pitfalls: list[str] = Field(
        default_factory=list, description="Pièges, conditions d'application, erreurs classiques."
    )


class ReviewSheetPayload(BaseModel):
    """What the model returns when asked for a review sheet."""

    title: str = Field(description="Titre de la fiche (le thème du chapitre).")
    objectives: list[str] = Field(
        min_length=1, max_length=6, description="Ce qu'il faut savoir faire après ce chapitre."
    )
    sections: list[ReviewSection] = Field(min_length=1)
    takeaways: list[str] = Field(min_length=1, max_length=8, description="À retenir absolument.")


class ReviewSheet(ReviewSheetPayload):
    """A review sheet of one chapter, built from its slides (shared between students)."""

    sheet_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    course: str = ""
    chapter: str = ""
    level: str = ""
    slides: list[str] = Field(default_factory=list, description="Titres des slides du chapitre.")

    def to_markdown(self) -> str:
        """Markdown of the sheet (KaTeX-friendly), as shown in the chat and downloaded."""
        lines = [f"# 📄 {self.title}", "", f"*{self.course} › {self.chapter}*", "", "## Objectifs", ""]
        lines += [f"- {o}" for o in self.objectives]
        for sec in self.sections:
            lines += ["", f"## {sec.heading}", ""]
            lines += [f"- {p}" for p in sec.key_points]
            if sec.definitions:
                lines += ["", "**Définitions**", ""] + [f"- {d}" for d in sec.definitions]
            if sec.formulas:
                lines += ["", "**Formules**", ""] + [f"- {f}" for f in sec.formulas]
            if sec.pitfalls:
                lines += ["", "**Pièges et conditions**", ""] + [f"- ⚠️ {p}" for p in sec.pitfalls]
        lines += ["", "## À retenir", ""] + [f"- {t}" for t in self.takeaways]
        if self.slides:
            lines += ["", f"*Slides du chapitre : {' · '.join(self.slides)}*"]
        return "\n".join(lines)
