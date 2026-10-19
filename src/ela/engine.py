"""ELA conversation engine (one lightweight instance per chat session).

Heavy components (embeddings, vector store, models, the compiled agent) are
shared process-wide; :class:`ElaEngine` holds the user's level and the
per-session history summary, and turns one student message into a stream
of :mod:`ela.agent.events`:

1. images, if any, are transcribed by the vision model and appended to the
   message (the text agent never needs a multimodal model);
2. the agent (``create_agent``) decides when and how to search, with its
   tools and middleware;
3. the final answer is checked against the passages really handed to the
   model (phantom citations removed) and normalised for KaTeX.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator, Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ela import prompts
from ela.agent.citations import verify_citations
from ela.agent.context import TurnContext
from ela.agent.events import (
    AgentEvent,
    CodeRunEvent,
    ErrorEvent,
    ExerciseEvent,
    FinalEvent,
    QuizEvent,
    ResetEvent,
    ReviewSheetEvent,
    SourcesEvent,
    TokenEvent,
    ToolEndEvent,
    ToolStartEvent,
    tool_failed,
    tool_label,
    tool_summary,
)
from ela.agent.vision import ImageInput, transcribe_images, with_transcription
from ela.config import get_settings
from ela.history import ChatTurn, HistorySummarizer, split_history, to_messages
from ela.llm import get_chat_model
from ela.mathfmt import normalize_math
from ela.pedagogy.exercise import check_step as _check_step
from ela.pedagogy.exercise import generate_exercise as _generate_exercise
from ela.pedagogy.quiz import generate_quiz as _generate_quiz
from ela.pedagogy.review import ReviewLookupError, review_sheet_for
from ela.pedagogy.schemas import Difficulty, Exercise, Quiz, ReviewSheet, StepFeedback
from ela.retrieval import ScopedRetriever, get_retriever
from ela.sandbox import service as sandbox
from ela.sandbox.client import RunResult
from ela.structured import with_overrides
from ela.tracing import run_config

logger = logging.getLogger(__name__)

LIMIT_MESSAGE = (
    "⚠️ Je n'ai pas réussi à conclure dans le nombre d'étapes autorisé. "
    "Reformulez la question de façon plus précise (une notion à la fois)."
)
EMPTY_MESSAGE = "⚠️ Je n'ai pas pu produire de réponse. Merci de reformuler votre question."


class GenerationError(RuntimeError):
    """A quiz or exercise could not be generated (model or provider failure, not a missing topic)."""


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
    """Forget the cached clients and the shared agent (tests / provider change)."""
    global _chat_llm, _vision_llm
    with _llm_lock:
        _chat_llm = None
        _vision_llm = None
    from ela.agent.agent import reset_agent

    reset_agent()


def chunk_text(content) -> str:
    """Extract plain text from a message or streamed chunk content.

    Args:
        content: ``AIMessage.content`` (str or list of content blocks).

    Returns:
        The textual part (possibly empty); reasoning blocks are ignored.
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
    """Format retrieved documents as a context block (code workshop).

    Args:
        docs: Retrieved documents.

    Returns:
        One ``>> [Source: file — chapter > section > slide]`` section per document.
    """
    parts = []
    for doc in docs:
        meta = doc.metadata or {}
        src = meta.get("source", "Inconnu")
        where = " > ".join(b for b in (meta.get("chapter"), meta.get("section"), meta.get("slide_title")) if b)
        label = f"{src} — {where}" if where else src
        parts.append(f">> [Source: {label}]\n{doc.page_content}")
    return "\n\n".join(parts)


class ElaEngine:
    """Per-session façade over the shared agent and RAG components.

    Args:
        user_level: Academic level of the current user (``M1``, ``M2``, ``ALL``).
        retriever: Retriever override (tests); defaults to the shared one.
        llm: Text model override (tests; quiz/exercise/summary generation).
        vision: Vision model override (tests).
        agent: Compiled agent override (tests); defaults to the shared one.
    """

    def __init__(
        self,
        user_level: str = "ALL",
        *,
        retriever: ScopedRetriever | None = None,
        llm: BaseChatModel | None = None,
        vision: BaseChatModel | None = None,
        agent=None,
    ) -> None:
        self.user_level = user_level
        self._retriever = retriever
        self._llm = llm
        self._vision = vision
        self._agent = agent
        self._summarizer = HistorySummarizer(llm)
        #: Context of the last turn (sources, searches, quiz/exercise) for the UI and the evaluation.
        self.last_context: TurnContext | None = None

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
    def agent(self):
        """Compiled agent (shared unless overridden)."""
        if self._agent is None:
            from ela.agent.agent import get_agent

            self._agent = get_agent()
        return self._agent

    @property
    def summary(self) -> str:
        """Current summary of the older part of the conversation."""
        return self._summarizer.summary

    # --- history ---------------------------------------------------------

    async def _prepare_history(self, chat_history) -> tuple[list[BaseMessage], str | None]:
        """Bound the history and update the summary of what falls out of the window."""
        if not chat_history:
            return [], self._summarizer.summary or None
        if isinstance(chat_history[0], BaseMessage):  # already bounded by the caller
            return list(chat_history), self._summarizer.summary or None
        dropped, kept = split_history(list(chat_history))
        summary = None
        if dropped and get_settings().agent_summary_enabled:
            summary = await self._summarizer.update(dropped) or None
        return to_messages(kept), summary

    # --- main entry point ----------------------------------------------

    async def ask(
        self,
        question: str,
        chat_history: Sequence[ChatTurn] | Sequence[BaseMessage] | None = None,
        images: Sequence[ImageInput] | None = None,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        quiz_difficulty: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Answer a student message with the agent, streaming events.

        Args:
            question: The student's message.
            chat_history: Previous turns of the thread (``ChatTurn`` objects, bounded
                and summarised here) or already-bounded LangChain messages.
            images: Optional ``(path, mime)`` tuples, transcribed by the vision model.
            session_id: Chat thread id (tracing).
            user_id: Student identifier (tracing, only if enabled).
            quiz_difficulty: Adaptive quiz difficulty of the session (``generate_quiz`` tool).

        Yields:
            :mod:`ela.agent.events` events, ending with ``FinalEvent`` or ``ErrorEvent``.
        """
        try:
            history, summary = await self._prepare_history(chat_history)
            text = question
            if images:
                async for event in self._transcribe(list(images), question):
                    if isinstance(event, str):
                        text = with_transcription(question, event)
                    else:
                        yield event

            ctx = TurnContext(
                level=self.user_level,
                retriever=self._retriever,
                llm=self._llm,
                summary=summary,
                quiz_difficulty=quiz_difficulty,
                user_id=user_id,
                thread_id=session_id,
            )
            self.last_context = ctx
            config = run_config(
                run_name="ela.ask",
                session_id=session_id,
                user_id=user_id,
                tags=["chat"],
                metadata={"level": self.user_level, "images": len(images or [])},
            )
            async for event in self._run(ctx, [*history, HumanMessage(content=text)], config):
                yield event
        except Exception as exc:  # surfaced in the chat, logged with traceback
            logger.exception("ask() failed")
            yield ErrorEvent(f"❌ Erreur ELA : {exc}")

    async def _transcribe(self, images: list[ImageInput], question: str) -> AsyncIterator[AgentEvent | str]:
        """Transcribe images with the vision model, as a visible pseudo-tool step."""
        yield ToolStartEvent("image", "read_image", {"n": len(images)}, tool_label("read_image", {}))
        try:
            transcription = await transcribe_images(self.vision, images, question)
        except Exception as exc:
            logger.exception("Image transcription failed")
            yield ToolEndEvent("image", "read_image", f"échec ({exc.__class__.__name__})", ok=False)
            return
        logger.info("%d image(s) transcribed (%d chars)", len(images), len(transcription))
        yield ToolEndEvent("image", "read_image", f"{len(transcription)} caractères transcrits")
        yield transcription

    async def _run(self, ctx: TurnContext, messages: list[BaseMessage], config) -> AsyncIterator[AgentEvent]:
        """Stream one agent run and post-process the final answer."""
        pending: dict[str, tuple[str, dict]] = {}
        streamed = False
        final_text: str | None = None
        n_model = 0
        n_tools = 0
        limit = False

        async for mode, payload in self.agent.astream(
            {"messages": messages}, context=ctx, stream_mode=["messages", "updates"], config=config
        ):
            if mode == "messages":
                chunk, meta = payload
                if meta.get("langgraph_node") == "model" and isinstance(chunk, AIMessage | AIMessageChunk):
                    token = chunk_text(chunk.content)
                    if token:
                        streamed = True
                        yield TokenEvent(token)
                continue
            for node, update in (payload or {}).items():
                msgs = (update or {}).get("messages") or []
                if node == "model":
                    n_model += 1
                    for m in msgs:
                        if not isinstance(m, AIMessage):
                            continue
                        if m.tool_calls:
                            if streamed:
                                yield ResetEvent()
                                streamed = False
                            for tc in m.tool_calls:
                                call_id = tc.get("id") or f"call_{len(pending) + 1}"
                                pending[call_id] = (tc["name"], tc.get("args") or {})
                                yield ToolStartEvent(
                                    call_id,
                                    tc["name"],
                                    tc.get("args") or {},
                                    tool_label(tc["name"], tc.get("args") or {}),
                                )
                        else:
                            final_text = chunk_text(m.content)
                    continue
                for m in msgs:
                    if isinstance(m, ToolMessage):
                        name, _args = pending.pop(m.tool_call_id, (m.name or "tool", {}))
                        n_tools += 1
                        result = chunk_text(m.content) if not isinstance(m.content, str) else m.content
                        ok = m.status != "error" and not tool_failed(result)
                        yield ToolEndEvent(m.tool_call_id, name, tool_summary(name, result), ok=ok)
                    elif isinstance(m, AIMessage) and node.startswith("ModelCallLimit"):
                        limit = True

        if final_text is None or not final_text.strip():
            final_text = LIMIT_MESSAGE if limit else EMPTY_MESSAGE
        check = verify_citations(final_text, ctx.registry.ids())
        if check.invalid:
            logger.warning("Phantom citations removed: %s", check.invalid)
        text = normalize_math(check.text)
        if ctx.quiz is not None:
            yield QuizEvent(ctx.quiz)
        if ctx.exercise is not None:
            yield ExerciseEvent(ctx.exercise)
        if ctx.review_sheet is not None:
            yield ReviewSheetEvent(ctx.review_sheet)
        for code, result in ctx.code_runs:
            yield CodeRunEvent(code, result)
        yield SourcesEvent(cited=ctx.registry.cited(text), all=list(ctx.registry))
        yield FinalEvent(
            text=text,
            citations_valid=check.valid,
            citations_invalid=check.invalid,
            n_model_calls=n_model,
            n_tool_calls=n_tools,
            n_searches=ctx.n_searches,
            limit_reached=limit,
        )

    async def ask_text(self, question: str, **kwargs) -> str:
        """Answer and return only the final text (tests, scripts)."""
        final = ""
        async for event in self.ask(question, **kwargs):
            if isinstance(event, FinalEvent):
                final = event.text
            elif isinstance(event, ErrorEvent):
                final = event.message
        return final

    # --- pedagogical objects (direct calls from the UI commands) ----------

    async def generate_quiz(
        self,
        topic: str,
        num_questions: int = 3,
        difficulty: Difficulty = "moyen",
        *,
        focus: list[str] | None = None,
        course: str | None = None,
        chapter: str | None = None,
    ) -> Quiz | None:
        """Generate a quiz grounded in the course (``/start_quiz`` command, targeted retry, suggestions).

        Args:
            topic: Quiz topic.
            num_questions: Number of questions.
            difficulty: ``facile`` / ``moyen`` / ``difficile``.
            focus: Previously failed questions to re-target with new wording.
            course: Restrict the passages to one course folder.
            chapter: Restrict the passages to one chapter title.

        Returns:
            The quiz, or ``None`` when the course has nothing on the topic.

        Raises:
            GenerationError: When the model did not produce a usable quiz.
        """
        ctx = TurnContext(level=self.user_level, retriever=self._retriever, llm=self._llm)
        self.last_context = ctx
        try:
            quiz = await _generate_quiz(
                self.llm,
                self.retriever,
                topic,
                n=num_questions,
                difficulty=difficulty,
                level=self.user_level,
                registry=ctx.registry,
                focus=focus,
                course=course,
                chapter=chapter,
            )
        except Exception as exc:
            logger.exception("Quiz generation failed")
            raise GenerationError("quiz") from exc
        ctx.quiz = quiz
        return quiz

    async def generate_exercise(self, topic: str) -> Exercise | None:
        """Generate an exercise grounded in the course.

        Args:
            topic: Exercise topic.

        Returns:
            The exercise, or ``None`` when the course has nothing on the topic.

        Raises:
            GenerationError: When the model did not produce a usable exercise.
        """
        ctx = TurnContext(level=self.user_level, retriever=self._retriever, llm=self._llm)
        self.last_context = ctx
        try:
            exercise = await _generate_exercise(
                self.llm, self.retriever, topic, level=self.user_level, registry=ctx.registry
            )
        except Exception as exc:
            logger.exception("Exercise generation failed")
            raise GenerationError("exercice") from exc
        ctx.exercise = exercise
        return exercise

    async def generate_review_sheet(self, course: str, chapter: str) -> ReviewSheet:
        """Review sheet of a chapter (``/fiche`` command, suggestions), cached per chapter.

        Args:
            course: Course name, acronym or file name.
            chapter: Chapter title, number or fragment.

        Returns:
            The sheet.

        Raises:
            ReviewLookupError: Unknown course or chapter (message for the student).
            GenerationError: When the model did not produce a usable sheet.
        """
        try:
            return await review_sheet_for(self.llm, self.user_level, course, chapter)
        except ReviewLookupError:
            raise
        except Exception as exc:
            logger.exception("Review sheet generation failed")
            raise GenerationError("fiche") from exc

    async def run_code(
        self,
        code: str,
        *,
        user_id: str | None,
        thread_id: str | None = None,
        origin: str = "workshop",
        topic: str | None = None,
        language: str = "Python",
    ) -> RunResult:
        """Execute a script of the Atelier Code in the sandbox (quota and persistence included).

        Raises:
            QuotaExceededError: Daily quota reached.
            SandboxUnavailableError: Sandbox not configured or unreachable.
        """
        return await sandbox.execute(
            code, user_id=user_id, thread_id=thread_id, origin=origin, topic=topic, language=language.lower()
        )

    async def interpret_run(
        self, code: str, result: RunResult, *, topic: str = "", language: str = "Python"
    ) -> sandbox.Interpretation:
        """Interpret the real output of a successful run against the course.

        Raises:
            GenerationError: When the model call failed.
        """
        ctx = TurnContext(level=self.user_level, retriever=self._retriever, llm=self._llm)
        self.last_context = ctx
        try:
            return await sandbox.interpret_run(
                self.code_llm,
                self.retriever,
                code,
                result,
                topic=topic,
                level=self.user_level,
                language=language,
                registry=ctx.registry,
            )
        except Exception as exc:
            logger.exception("Run interpretation failed")
            raise GenerationError("interprétation") from exc

    async def fix_code(
        self,
        code: str,
        error: str,
        *,
        topic: str = "",
        language: str = "Python",
        history: list[str] | None = None,
    ) -> str:
        """Ask the model to repair a failing script (one pass).

        Args:
            code: Failing script.
            error: Error text (or hint) to fix.
            topic: Subject of the script.
            language: Script language.
            history: Errors of the previous attempts (regression guard).

        Raises:
            GenerationError: When the model returned no code block.
        """
        try:
            return await sandbox.fix_code(self.code_llm, code, error, topic=topic, language=language, history=history)
        except Exception as exc:
            logger.exception("Code fix failed")
            raise GenerationError("correction") from exc

    async def check_step(
        self,
        exercise: Exercise,
        attempt: str,
        *,
        revealed_hints: int = 0,
        previous: list[tuple[str, str]] | None = None,
    ) -> StepFeedback | None:
        """Socratic feedback on a student's step (``None`` on failure).

        Args:
            exercise: Exercise being solved.
            attempt: The student's text.
            revealed_hints: Hints already shown.
            previous: Earlier ``(attempt, verdict)`` pairs.
        """
        try:
            return await _check_step(
                self.llm, exercise, attempt, level=self.user_level, revealed_hints=revealed_hints, previous=previous
            )
        except Exception:
            logger.exception("check_step failed")
            return None

    # --- code workshop -------------------------------------------------

    @property
    def code_llm(self) -> BaseChatModel:
        """Model used by the Atelier Code (bigger budget, higher reasoning effort than the chat)."""
        s = get_settings()
        effort = None if s.code_reasoning_effort == "omit" else s.code_reasoning_effort
        return with_overrides(self.llm, max_tokens=s.code_max_tokens, reasoning_effort=effort)

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
            response = await self.code_llm.ainvoke(messages)
            return chunk_text(response.content)
        except Exception as exc:
            logger.exception("Code generation failed")
            return f"❌ Erreur de génération de code : {exc}"
