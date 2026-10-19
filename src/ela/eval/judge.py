"""LLM judge for the end-to-end evaluation.

For each reference question, the judge receives the student's question, the
answer produced by ELA and the ``answer_points`` listed in
``eval/questions.yaml``. It decides which points are covered, whether the
answer refused (expected for out-of-domain questions) and whether it
acknowledges that the course does not cover the topic (expected for
not-in-corpus questions).

The judge is deliberately *not* the retrieval metric: it measures the
fidelity of the final answer, independently of which slides were found.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ela.structured import ainvoke_structured

ExpectedBehavior = Literal["answer", "refuse", "general"]


class PointVerdict(BaseModel):
    """Whether one expected point is covered by the answer."""

    point: str = Field(description="Le point attendu, recopié.")
    covered: bool = Field(description="Vrai si la réponse contient ce point (même formulé autrement).")
    evidence: str = Field(default="", description="Courte citation de la réponse qui le justifie, sinon vide.")


class Judgment(BaseModel):
    """Verdict of the judge on one answer."""

    points: list[PointVerdict] = Field(default_factory=list)
    refused: bool = Field(description="Vrai si la réponse refuse de traiter la question (hors domaine).")
    acknowledges_not_in_course: bool = Field(
        description="Vrai si la réponse indique que le cours ne couvre pas (ou pas directement) le sujet."
    )
    contradiction: bool = Field(
        default=False,
        description="Vrai si la réponse contredit un des points attendus ou contient une erreur manifeste.",
    )
    comment: str = Field(default="", description="Une phrase de synthèse.")

    @property
    def fidelity(self) -> float | None:
        """Share of expected points covered (``None`` when there is no point)."""
        if not self.points:
            return None
        return sum(1 for p in self.points if p.covered) / len(self.points)


JUDGE_SYSTEM_PROMPT = """Tu es un évaluateur rigoureux de réponses d'un assistant pédagogique en économétrie.

On te donne une QUESTION d'étudiant, la RÉPONSE de l'assistant et une liste de POINTS ATTENDUS
(rédigés par l'enseignant à partir de son cours). Pour chaque point, décide s'il est couvert par la
réponse : le fond compte, pas la formulation (une formule équivalente ou une explication en mots
suffit). Sois exigeant sur les erreurs : un point énoncé de façon fausse n'est pas couvert et
`contradiction` doit être vrai.

Indique aussi :
- `refused` : la réponse décline explicitement la question (sujet hors du cadre du cours) ;
- `acknowledges_not_in_course` : la réponse dit que le cours ne traite pas (ou pas directement) le sujet,
  typiquement avec l'étiquette [Connaissances générales].

Réponds uniquement dans le format structuré demandé."""


def expected_behavior(question_type: str) -> ExpectedBehavior:
    """Behaviour expected for a question type of the reference set."""
    if question_type == "out-of-domain":
        return "refuse"
    if question_type == "not-in-corpus":
        return "general"
    return "answer"


async def judge_answer(
    llm: BaseChatModel,
    *,
    question: str,
    answer: str,
    answer_points: list[str],
) -> Judgment:
    """Ask the judge model to grade one answer.

    Args:
        llm: Judge chat model.
        question: Student question.
        answer: Answer produced by ELA.
        answer_points: Expected points (may be empty for refusal questions).

    Returns:
        The structured judgment.
    """
    points_block = "\n".join(f"- {p}" for p in answer_points) if answer_points else "(aucun point attendu)"
    user = f"QUESTION :\n{question}\n\nPOINTS ATTENDUS :\n{points_block}\n\nRÉPONSE DE L'ASSISTANT :\n{answer}"
    judgment = await ainvoke_structured(
        llm, Judgment, [SystemMessage(content=JUDGE_SYSTEM_PROMPT), HumanMessage(content=user)]
    )
    # Keep the teacher's wording even if the model paraphrased the points.
    if answer_points and len(judgment.points) == len(answer_points):
        for verdict, point in zip(judgment.points, answer_points, strict=True):
            verdict.point = point
    return judgment
