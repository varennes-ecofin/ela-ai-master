"""Reference question set for evaluating the retrieval and the answers.

The dataset lives in ``eval/questions.yaml`` at the project root. Each
question lists the slides that retrieval must surface and the points a
good answer must contain (used by the LLM judge in a later step).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from langchain_core.documents import Document
from pydantic import BaseModel, Field, model_validator

QuestionType = Literal["definition", "formula", "procedure", "reasoning", "multi-hop", "out-of-domain", "not-in-corpus"]

DEFAULT_PATH = Path("eval") / "questions.yaml"


class ExpectedSource(BaseModel):
    """A slide (or section) that retrieval is expected to return."""

    source: str
    slide_title: str | None = None
    section: str | None = None

    @model_validator(mode="after")
    def _one_locator(self) -> ExpectedSource:
        if not (self.slide_title or self.section):
            raise ValueError("expected source needs slide_title or section")
        return self

    def matches(self, doc: Document) -> bool:
        """Whether a retrieved document corresponds to this expected slide.

        Args:
            doc: Retrieved LangChain document (metadata ``source``, ``slide_title``, ``section``).

        Returns:
            ``True`` when the source file matches and the title/section substring is found.
        """
        meta = doc.metadata or {}
        if meta.get("source") != self.source:
            return False
        if self.slide_title:
            return self.slide_title.lower() in (meta.get("slide_title") or "").lower()
        return (self.section or "").lower() in (meta.get("section") or "").lower()


class EvalQuestion(BaseModel):
    """One reference question."""

    id: str
    level: str
    course: str
    type: QuestionType
    difficulty: int = Field(ge=1, le=3)
    question: str
    expected_sources: list[ExpectedSource] = Field(default_factory=list)
    answer_points: list[str] = Field(default_factory=list)
    notes: str | None = None

    @property
    def expects_retrieval(self) -> bool:
        """Questions whose quality is measured by retrieval hits."""
        return bool(self.expected_sources)

    def hits(self, docs: list[Document]) -> list[bool]:
        """For each expected source, whether at least one retrieved doc matches."""
        return [any(exp.matches(d) for d in docs) for exp in self.expected_sources]

    def first_hit_rank(self, docs: list[Document]) -> int | None:
        """1-based rank of the first retrieved doc matching any expected source."""
        for rank, d in enumerate(docs, start=1):
            if any(exp.matches(d) for exp in self.expected_sources):
                return rank
        return None


class EvalDataset(BaseModel):
    """The whole question set."""

    version: int
    questions: list[EvalQuestion]

    @model_validator(mode="after")
    def _unique_ids(self) -> EvalDataset:
        ids = [q.id for q in self.questions]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate question ids: {sorted(dupes)}")
        return self

    def by_type(self, *types: QuestionType) -> list[EvalQuestion]:
        """Questions of the given types."""
        return [q for q in self.questions if q.type in types]


def load_dataset(path: str | Path = DEFAULT_PATH) -> EvalDataset:
    """Load and validate the YAML question set.

    Args:
        path: Path of the YAML file.

    Returns:
        The validated dataset.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return EvalDataset.model_validate(data)
