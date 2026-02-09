# rag_guard.py
"""
RAG Guard - Lightweight relevance grader for ELA AI.

Intercepts between retrieval and generation to:
1. Grade whether retrieved documents are relevant to the query.
2. Classify whether the query falls within ELA's authorized domain.
3. Route to the appropriate response strategy (RAG / refuse / retry).

This replaces the need for a full LangGraph CRAG implementation
while solving the core problem: ELA falling back to general knowledge
when the RAG actually contains the answer.
"""

import json
from typing import List, Literal
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq


# --- Grading result ---

@dataclass
class GradeResult:
    """Output of the RAG guard grading step."""
    docs_relevant: bool          # Are retrieved docs relevant to the query?
    query_in_domain: bool        # Is the query within ELA's authorized scope?
    reasoning: str               # Short explanation (for debugging/logging)
    action: Literal["answer_rag", "refuse_domain", "retry_query"]
    suggested_query: str = ""    # Alternative search query if action == retry_query


# --- Grading prompt (kept minimal to reduce latency) ---

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
    """Lightweight grader that validates retrieval quality before generation."""

    def __init__(self, llm: ChatGroq, max_retries: int = 1):
        """
        Initialize the RAG guard.

        Args:
            llm: The ChatGroq LLM instance (shared with ELA to avoid extra init).
            max_retries: Max number of query reformulation attempts.
        """
        self.llm = llm
        self.max_retries = max_retries

    async def grade(self, question: str, documents: List[Document]) -> GradeResult:
        """
        Grade the relevance of retrieved documents against the user query.

        Args:
            question: The user's original question.
            documents: List of documents retrieved from the RAG pipeline.

        Returns:
            GradeResult with routing decision.
        """
        # Format documents for the grader (compact, just enough for assessment)
        docs_summary = self._format_docs_for_grading(documents)

        user_content = (
            f"QUESTION DE L'ÉTUDIANT :\n{question}\n\n"
            f"DOCUMENTS RÉCUPÉRÉS ({len(documents)} documents) :\n{docs_summary}"
        )

        messages = [
            SystemMessage(content=GRADER_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            result = self._parse_grade_response(response.content)
            return result
        except Exception as e:
            # Fail-open: if grading fails, proceed with RAG answer
            print(f"⚠️ RAG Guard error (fail-open): {e}")
            return GradeResult(
                docs_relevant=True,
                query_in_domain=True,
                reasoning=f"Grading failed ({e}), proceeding with RAG.",
                action="answer_rag",
            )

    def _format_docs_for_grading(self, documents: List[Document]) -> str:
        """
        Format documents into a compact string for grading.
        Only includes first ~300 chars per doc to keep the grader fast.

        Args:
            documents: Retrieved documents.

        Returns:
            Formatted string summary.
        """
        if not documents:
            return "(Aucun document récupéré)"

        parts = []
        for i, doc in enumerate(documents[:5]):  # Cap at 5 docs for speed
            source = doc.metadata.get("source", "?")
            section = doc.metadata.get("section", "?")
            # Truncate content for grading (we don't need the full text)
            content_preview = doc.page_content[:300].replace("\n", " ")
            parts.append(f"[Doc {i+1} | {source} | {section}] {content_preview}")
        return "\n".join(parts)

    def _parse_grade_response(self, content: str) -> GradeResult:
        """
        Parse the LLM's JSON response into a GradeResult.

        Args:
            content: Raw LLM response string.

        Returns:
            Parsed GradeResult.
        """
        # Clean markdown fences if present
        cleaned = content.strip()
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