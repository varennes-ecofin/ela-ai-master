"""Frozen Phase-2 pipeline (``retrieve → grade → answer``) for the evaluation baseline.

This is the pre-agent chain kept **only** for ``ela-eval answer --engine
legacy``: it lets the agent be compared with what students had before,
on the same questions and the same store. It is not used by the chat
application and can be deleted once the comparison is no longer needed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ela import prompts
from ela.engine import chat_llm, chunk_text, format_context
from ela.retrieval import ScopedRetriever, get_retriever

logger = logging.getLogger(__name__)


@dataclass
class GradeResult:
    """Output of the relevance grader."""

    docs_relevant: bool
    query_in_domain: bool
    reasoning: str
    action: Literal["answer_rag", "refuse_domain", "retry_query"]
    suggested_query: str = ""


GRADER_SYSTEM_PROMPT = """Tu es un module de contrôle qualité pour un assistant pédagogique en économétrie.

TÂCHE : Évalue en JSON si les documents récupérés sont pertinents pour la question posée, et si la question est dans le domaine autorisé.

DOMAINE AUTORISÉ d'ELA :
- Séries temporelles (ARIMA, VAR, VECM, stationnarité, cointégration, racine unitaire, bruit blanc, AR, MA, ARMA, processus non-causaux, etc.)
- Économétrie financière (ARCH, GARCH, volatilité, rendements, gestion des risques)
- Économétrie générale si couverte dans les documents du cours (MCO, tests statistiques, etc.)
- Code Python/R appliqué à ces sujets

DOMAINE INTERDIT (sauf si présent dans les documents) :
- Micro-économétrie (Panel, Logit/Probit, Tobit, IV)
- Machine Learning généraliste (Classification, Clustering, NLP, Vision)
- Culture générale, politique, histoire

RÈGLES D'ÉVALUATION :
1. docs_relevant = true si AU MOINS UN document contient des informations directement liées à la question (même partiellement).
2. query_in_domain = true si la question porte sur le domaine autorisé OU si les documents contiennent effectivement du contenu pertinent (le cours prime sur les règles théoriques).
3. Si docs_relevant = false MAIS query_in_domain = true → action = "retry_query", et propose une reformulation de recherche dans suggested_query.
4. Si query_in_domain = false → action = "refuse_domain".
5. Sinon → action = "answer_rag".

IMPORTANT : Sois TOLÉRANT sur la pertinence. Si un document mentionne le concept demandé, même indirectement, considère-le comme pertinent. Le but est d'éviter les faux négatifs (dire que le RAG n'a rien alors qu'il a la réponse).

FORMAT DE SORTIE (JSON strict, rien d'autre) :
{"docs_relevant": bool, "query_in_domain": bool, "reasoning": "...", "action": "answer_rag|refuse_domain|retry_query", "suggested_query": "..."}"""


class RAGGuard:
    """Relevance grader of the Phase-2 pipeline (one model call per question)."""

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    async def grade(self, question: str, documents: list[Document]) -> GradeResult:
        """Grade the retrieved documents (fail-open on errors)."""
        parts = []
        for i, doc in enumerate(documents[:5]):
            preview = doc.page_content[:300].replace("\n", " ")
            parts.append(
                f"[Doc {i + 1} | {doc.metadata.get('source', '?')} | {doc.metadata.get('section', '?')}] {preview}"
            )
        summary = "\n".join(parts) or "(Aucun document récupéré)"
        messages = [
            SystemMessage(content=GRADER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"QUESTION DE L'ÉTUDIANT :\n{question}\n\nDOCUMENTS RÉCUPÉRÉS ({len(documents)} documents) :\n{summary}"
            ),
        ]
        try:
            response = await self.llm.ainvoke(messages)
            cleaned = chunk_text(response.content).strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            data = json.loads(cleaned)
            return GradeResult(
                docs_relevant=data.get("docs_relevant", True),
                query_in_domain=data.get("query_in_domain", True),
                reasoning=data.get("reasoning", ""),
                action=data.get("action", "answer_rag"),
                suggested_query=data.get("suggested_query", ""),
            )
        except Exception as exc:
            logger.warning("RAG Guard error (fail-open): %s", exc)
            return GradeResult(True, True, f"Grading failed ({exc})", "answer_rag")


class LegacyPipeline:
    """The Phase-2 chain for one level.

    Args:
        user_level: Academic level.
        retriever: Retriever override (tests).
        llm: Model override (tests).
    """

    def __init__(self, user_level: str, *, retriever: ScopedRetriever | None = None, llm: BaseChatModel | None = None):
        self.user_level = user_level
        self._retriever = retriever
        self._llm = llm
        self.last_docs: list[Document] = []

    @property
    def retriever(self) -> ScopedRetriever:
        """Level-scoped retriever."""
        if self._retriever is None:
            self._retriever = get_retriever(self.user_level)
        return self._retriever

    @property
    def llm(self) -> BaseChatModel:
        """Text model."""
        return self._llm if self._llm is not None else chat_llm()

    async def answer(self, question: str) -> str:
        """Answer ``question`` exactly as the Phase-2 chat did (no streaming)."""
        guard = RAGGuard(self.llm)
        docs = self.retriever.invoke(question)
        grade = await guard.grade(question, docs)
        if grade.action == "retry_query" and grade.suggested_query:
            docs_retry = self.retriever.invoke(grade.suggested_query)
            grade_retry = await guard.grade(question, docs_retry)
            if grade_retry.docs_relevant:
                docs, grade = docs_retry, grade_retry
        self.last_docs = list(docs)
        if grade.action == "refuse_domain":
            return f"{prompts.REFUSAL_MESSAGE}\n\n_({grade.reasoning})_"
        instruction = (
            prompts.SOURCE_INSTRUCTION_RELEVANT if grade.docs_relevant else prompts.SOURCE_INSTRUCTION_FALLBACK
        )
        system_prompt = (
            f"{prompts.ELA_BASE_INSTRUCTIONS}"
            f"{prompts.level_instructions(self.user_level)}\n\n"
            f"{instruction}\n\n"
            f"CONTEXTE DU COURS (RAG) :\n{format_context(docs)}"
        )
        response = await self.llm.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
        return chunk_text(response.content)
