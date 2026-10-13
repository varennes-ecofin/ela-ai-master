"""ELA conversation engine (one lightweight instance per chat session).

Heavy components (embeddings, vector store, BM25 indexes, reranker, LLM
clients) are shared process-wide; :class:`ElaEngine` only holds the
user's level and delegates to them. Phase 3 replaces the fixed
``retrieve → grade → answer`` pipeline with an agent, keeping this
public interface (``ask``, ``generate_quiz``, ``generate_code``).
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import AsyncIterator, Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ela import prompts
from ela.guard import GradeResult, RAGGuard
from ela.llm import encode_image_block, get_chat_model
from ela.retrieval import ScopedRetriever, get_retriever

logger = logging.getLogger(__name__)

# An uploaded image: (path on disk, MIME type reported by the UI)
ImageInput = tuple[str, str | None]

_llm_lock = threading.Lock()
_chat_llm: BaseChatModel | None = None
_vision_llm: BaseChatModel | None = None


def chat_llm() -> BaseChatModel:
    """Shared text model (with fallback), built on first use."""
    global _chat_llm
    if _chat_llm is None:
        with _llm_lock:
            if _chat_llm is None:
                _chat_llm = get_chat_model("chat")
    return _chat_llm


def vision_llm() -> BaseChatModel:
    """Shared multimodal model, built on first use."""
    global _vision_llm
    if _vision_llm is None:
        with _llm_lock:
            if _vision_llm is None:
                _vision_llm = get_chat_model("vision")
    return _vision_llm


def reset_llms() -> None:
    """Forget the cached clients (tests / provider change)."""
    global _chat_llm, _vision_llm
    with _llm_lock:
        _chat_llm = None
        _vision_llm = None


def chunk_text(content) -> str:
    """Extract plain text from a streamed message chunk.

    Args:
        content: ``AIMessageChunk.content`` (str or list of content blocks).

    Returns:
        The textual part of the chunk (possibly empty).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def format_context(docs: Sequence[Document]) -> str:
    """Format retrieved documents as the RAG context block.

    Args:
        docs: Retrieved documents.

    Returns:
        One ``>> [Source: file]`` section per document.
    """
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "Inconnu")
        parts.append(f">> [Source: {src}]\n{doc.page_content.replace(chr(10), ' ')}")
    return "\n\n".join(parts)


class ElaEngine:
    """Per-session façade over the shared RAG components.

    Args:
        user_level: Academic level of the current user (``M1``, ``M2``, ``ALL``).
        retriever: Retriever override (tests); defaults to the shared one.
        llm: Text model override (tests); defaults to the shared one.
        vision: Vision model override (tests); defaults to the shared one.
    """

    def __init__(
        self,
        user_level: str = "ALL",
        *,
        retriever: ScopedRetriever | None = None,
        llm: BaseChatModel | None = None,
        vision: BaseChatModel | None = None,
    ) -> None:
        self.user_level = user_level
        self._retriever = retriever
        self._llm = llm
        self._vision = vision
        self._guard: RAGGuard | None = None

    # --- lazy accessors ------------------------------------------------

    @property
    def retriever(self) -> ScopedRetriever:
        """Level-scoped retriever (shared, built on first access)."""
        if self._retriever is None:
            self._retriever = get_retriever(self.user_level)
        return self._retriever

    @property
    def llm(self) -> BaseChatModel:
        """Text model."""
        return self._llm if self._llm is not None else chat_llm()

    @property
    def vision(self) -> BaseChatModel:
        """Multimodal model."""
        return self._vision if self._vision is not None else vision_llm()

    @property
    def guard(self) -> RAGGuard:
        """Relevance grader bound to the text model."""
        if self._guard is None:
            self._guard = RAGGuard(llm=self.llm, max_retries=1)
        return self._guard

    # --- retrieval -----------------------------------------------------

    async def retrieve_and_grade(self, question: str) -> tuple[list[Document], GradeResult]:
        """Retrieve documents, grade them, retry once with a reformulation.

        Args:
            question: The user's question.

        Returns:
            ``(documents, grade)``.
        """
        docs = self.retriever.invoke(question)
        grade = await self.guard.grade(question, docs)
        if grade.action == "retry_query" and grade.suggested_query:
            logger.info("RAG Guard retry: %r", grade.suggested_query)
            docs_retry = self.retriever.invoke(grade.suggested_query)
            grade_retry = await self.guard.grade(question, docs_retry)
            if grade_retry.docs_relevant:
                return docs_retry, grade_retry
        return docs, grade

    # --- main entry point ----------------------------------------------

    async def ask(
        self,
        question: str,
        chat_history: list[BaseMessage] | None = None,
        images: Sequence[ImageInput] | None = None,
    ) -> AsyncIterator[str]:
        """Answer a question with RAG guard and streaming.

        Args:
            question: The user's question.
            chat_history: Bounded history (see ``ela.history``).
            images: Optional ``(path, mime)`` tuples; routes to the vision model.

        Yields:
            Response text chunks.
        """
        chat_history = chat_history or []
        images = list(images or [])

        try:
            docs, grade = await self.retrieve_and_grade(question)

            if grade.action == "refuse_domain":
                yield f"{prompts.REFUSAL_MESSAGE}\n\n_({grade.reasoning})_"
                return

            source_instruction = (
                prompts.SOURCE_INSTRUCTION_RELEVANT if grade.docs_relevant else prompts.SOURCE_INSTRUCTION_FALLBACK
            )
            system_prompt = (
                f"{prompts.ELA_BASE_INSTRUCTIONS}"
                f"{prompts.level_instructions(self.user_level)}\n\n"
                f"{source_instruction}\n\n"
                f"CONTEXTE DU COURS (RAG) :\n{format_context(docs)}"
            )

            content_blocks: list = [{"type": "text", "text": question}]
            for path, mime in images:
                content_blocks.append(encode_image_block(path, mime))

            messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
            messages.extend(chat_history)
            messages.append(HumanMessage(content=content_blocks))

            model = self.vision if images else self.llm
            if images:
                logger.info("%d image(s) attached → vision model", len(images))

            async for chunk in model.astream(messages):
                text = chunk_text(chunk.content)
                if text:
                    yield text
        except Exception as exc:  # surfaced in the chat, logged with traceback
            logger.exception("ask() failed")
            yield f"❌ Erreur ELA : {exc}"

    # --- quiz ----------------------------------------------------------

    async def generate_quiz(self, topic: str, num_questions: int = 3) -> list[dict]:
        """Generate quiz questions grounded in the course.

        Args:
            topic: Quiz topic.
            num_questions: Number of questions.

        Returns:
            List of question dicts; ``[{"error": "domain", ...}]`` when out of
            scope; empty list when nothing relevant was found or on failure.
        """
        docs, grade = await self.retrieve_and_grade(topic)
        if grade.action == "refuse_domain":
            return [{"error": "domain", "message": grade.reasoning}]
        if not grade.docs_relevant:
            return []

        messages = [
            SystemMessage(content=prompts.quiz_prompt(topic, num_questions, format_context(docs))),
            HumanMessage(content=f"Génère le quiz sur {topic} sans LaTeX."),
        ]
        try:
            response = await self.llm.ainvoke(messages)
            content = chunk_text(response.content).strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
            data = json.loads(content)
            return data if isinstance(data, list) else []
        except Exception:
            logger.exception("Quiz generation failed")
            return []

    # --- code workshop -------------------------------------------------

    async def generate_code(self, topic: str, language: str = "Python") -> str:
        """Generate a runnable example script grounded in the course theory.

        Args:
            topic: Model or concept to implement.
            language: ``Python`` or ``R``.

        Returns:
            The generated answer (intro + code block + interpretation).
        """
        docs = self.retriever.invoke(topic)
        system_prompt = prompts.code_prompt(
            topic, language, prompts.level_instructions(self.user_level), format_context(docs)
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Écris le script pour {topic} en {language}."),
        ]
        try:
            response = await self.llm.ainvoke(messages)
            return chunk_text(response.content)
        except Exception as exc:
            logger.exception("Code generation failed")
            return f"❌ Erreur de génération de code : {exc}"
