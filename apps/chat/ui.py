"""Chainlit rendering of the agent events (steps, sources panel, quiz, exercise).

Kept apart from ``app.py`` (routing, auth, session) so that the mapping
*event → UI* stays in one place:

* ``ToolStartEvent`` / ``ToolEndEvent`` → one ``cl.Step`` per tool call
  (« 🔎 Recherche : « test ADF » → 5 extrait(s) »);
* ``TokenEvent`` → streamed answer; ``ResetEvent`` clears what was streamed;
* ``SourcesEvent`` → one side element per cited passage, named ``S1``,
  ``S2``... so that the ``[S1]`` references of the answer become links to
  the panel (course › chapter › slide, extract rendered with KaTeX);
* ``QuizEvent`` / ``ExerciseEvent`` → interactive messages with buttons (quiz
  options are rendered in the message with KaTeX, the buttons carry letters);
* ``ReviewSheetEvent`` → the sheet as Markdown with a downloadable ``.md`` file;
* after an ordinary answer, follow-up buttons (quiz, exercise, sheet, code)
  derived from the cited passages.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime

import chainlit as cl

from ela.agent.citations import SourceRef, has_general_knowledge_tag
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
)
from ela.agent.outline import courses_elsewhere, get_outlines
from ela.config import get_settings
from ela.db.activity import QuestionRecord, record_question
from ela.db.exercises import ExerciseAttempt, record_attempt
from ela.db.progress import render_report, student_report
from ela.db.quiz import QuizAnswer, next_difficulty, record_answer
from ela.engine import ElaEngine, GenerationError
from ela.eval.answer import looks_like_refusal
from ela.history import ChatTurn
from ela.mathfmt import normalize_math
from ela.pedagogy.relevance import normalize
from ela.pedagogy.review import ReviewLookupError
from ela.pedagogy.schemas import Exercise, Quiz, ReviewSheet
from ela.sandbox import service as sandbox
from ela.sandbox.client import RunResult, SandboxUnavailableError
from ela.sandbox.client import enabled as sandbox_enabled

logger = logging.getLogger("ela.chat.ui")

AUTHOR = "ELA AI 🤖"
_LETTERS = ["A", "B", "C", "D"]
_OPTION_PREFIX = re.compile(r"^[A-D0-9][)\.]\s*")


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Sources panel
# ---------------------------------------------------------------------------


def source_markdown(ref: SourceRef) -> str:
    """Markdown shown in the side panel for one passage."""
    meta = [f"Fichier : `{ref.source}`"]
    if ref.score is not None:
        meta.append(f"score {ref.score:.3f}")
    if ref.rerank_score is not None:
        meta.append(f"rerank {ref.rerank_score:.3f}")
    if ref.query:
        meta.append(f"requête « {ref.query} »")
    header = f"**{ref.breadcrumb}**\n\n_{' · '.join(meta)}_\n\n---\n\n"
    return header + normalize_math(strip_context_header(ref.content))


_CONTEXT_HEADER = re.compile(r"\A(?:[A-ZÉ_ ]{4,}:[^\n]*\n)+CONTENU:\s*\n", re.MULTILINE)


def strip_context_header(content: str) -> str:
    """Drop the ``COURS:/CHAPITRE:/.../CONTENU:`` header the ingestion prepends to each chunk.

    The panel already shows the breadcrumb, so only the body is rendered.
    """
    return _CONTEXT_HEADER.sub("", content, count=1)


def source_elements(refs: Sequence[SourceRef]) -> list[cl.Text]:
    """Side elements of the cited passages (``name`` = citation label)."""
    return [cl.Text(name=ref.sid, content=source_markdown(ref), display="side") for ref in refs]


# ---------------------------------------------------------------------------
# One student turn
# ---------------------------------------------------------------------------


async def run_turn(
    engine: ElaEngine,
    question: str,
    history: list[ChatTurn],
    images: list[tuple[str, str | None]],
    *,
    session_id: str | None,
    user_id: str | None,
) -> FinalEvent | ErrorEvent | None:
    """Stream one agent turn into the chat.

    Args:
        engine: Session engine.
        question: Student message.
        history: Previous turns of the thread (bounded by the engine).
        images: Attached images ``(path, mime)``.
        session_id: Thread id (tracing).
        user_id: Student identifier (tracing, if enabled).

    Returns:
        The final (or error) event, for the caller (quiz / exercise follow-up).
    """
    msg = cl.Message(content="", author=AUTHOR)
    await msg.send()
    steps: dict[str, cl.Step] = {}
    quiz: Quiz | None = None
    exercise: Exercise | None = None
    sheet: ReviewSheet | None = None
    code_runs: list[CodeRunEvent] = []
    elements: list[cl.Text] = []
    outcome: FinalEvent | ErrorEvent | None = None
    sources_seen: SourcesEvent | None = None
    quiz_calls: dict[str, dict] = {}  # call_id → generate_quiz arguments
    failed_quiz: dict | None = None

    async for event in engine.ask(
        question,
        chat_history=history,
        images=images,
        session_id=session_id,
        user_id=user_id,
        quiz_difficulty=current_difficulty(),
    ):
        outcome = await _render_event(event, msg, steps, elements)
        if isinstance(event, SourcesEvent):
            sources_seen = event
        elif isinstance(event, QuizEvent):
            quiz = event.quiz
        elif isinstance(event, ExerciseEvent):
            exercise = event.exercise
        elif isinstance(event, ReviewSheetEvent):
            sheet = event.sheet
        elif isinstance(event, CodeRunEvent):
            code_runs.append(event)
        elif isinstance(event, ToolStartEvent) and event.name == "generate_quiz":
            quiz_calls[event.call_id] = event.args
        elif isinstance(event, ToolEndEvent) and not event.ok and event.call_id in quiz_calls:
            failed_quiz = quiz_calls[event.call_id]

    for step in steps.values():  # tool calls never reported as finished (error paths)
        await _close_step(step, "interrompu", ok=False)
    if isinstance(outcome, FinalEvent) and user_id:
        await _record_activity(
            engine, question, outcome, sources_seen, user_id=user_id, thread_id=session_id, sheet=sheet
        )
    if isinstance(outcome, FinalEvent):
        actions: list[cl.Action] = []
        if quiz is None and failed_quiz is not None:
            # The model's quiz tool failed: offer a direct retry with the same parameters
            # (a new agent turn would forget the requested number of questions).
            actions.append(
                quiz_retry_action(
                    str(failed_quiz.get("topic") or question),
                    int(failed_quiz.get("n") or 3),
                    str(failed_quiz.get("difficulty") or current_difficulty()),
                )
            )
        elif quiz is None and exercise is None and sheet is None:
            actions = followup_actions(question, outcome, sources_seen)
        if actions:
            msg.actions = actions
            await msg.update()
    if quiz is not None:
        await start_quiz(quiz, user_id=user_id, thread_id=session_id)
    if exercise is not None:
        await show_exercise(exercise, user_id=user_id, thread_id=session_id)
    if sheet is not None:
        await show_review_sheet(sheet, user_id=user_id, thread_id=session_id)
    for run in code_runs:
        await show_code_output(run.result, code=run.code, topic=question, language="Python")
    return outcome


async def _record_activity(
    engine: ElaEngine,
    question: str,
    final: FinalEvent,
    sources: SourcesEvent | None,
    *,
    user_id: str,
    thread_id: str | None,
    sheet: ReviewSheet | None = None,
) -> None:
    """Persist the answered question for the student report and the backoffice."""
    cited = list(sources.cited) if sources is not None else []
    course = _dominant(r.course for r in cited) or _dominant(r.course for r in (sources.all if sources else []))
    chapter = _dominant(r.chapter for r in cited)
    if sheet is not None and not cited:  # a review sheet cites no passage: the chapter is the sheet's
        course, chapter = sheet.course, sheet.chapter
    await record_question(
        QuestionRecord(
            user_identifier=user_id,
            thread_id=thread_id,
            level=getattr(engine, "user_level", None),
            question=question,
            course=course,
            chapter=chapter,
            n_searches=final.n_searches,
            n_sources=len(cited),
            cited=[r.breadcrumb for r in cited],
            general_knowledge=has_general_knowledge_tag(final.text),
            refused=looks_like_refusal(final.text),
        )
    )


def _dominant(values) -> str | None:
    counts: dict[str, int] = {}
    for v in values:
        if v:
            counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get) if counts else None


_CODE_WORDS = re.compile(r"\b(python|code|script|implémenter|implementer|programme|statsmodels|numpy|pandas)\b", re.I)


def _short(text: str, n: int = 40) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def followup_actions(question: str, final: FinalEvent, sources: SourcesEvent | None) -> list[cl.Action]:
    """Buttons offered after an answer, derived from the cited passages (no model call).

    « Me tester » on the notion searched (quiz restricted to the dominant
    chapter), an exercise on that notion, the review sheet of the chapter,
    and a Python example when the question is about code. Nothing on a
    refusal, on an answer without any cited passage, or during a quiz.

    Args:
        question: Student message.
        final: Final event of the turn.
        sources: Passages of the turn.

    Returns:
        Up to four ``report_suggestion`` actions.
    """
    if sources is None or not sources.cited or looks_like_refusal(final.text):
        return []
    if cl.user_session.get("quiz_mode") == "active" or cl.user_session.get("exercise_mode"):
        return []
    cited = list(sources.cited)
    course = _dominant(r.course for r in cited)
    chapter = _dominant(r.chapter for r in cited)
    notion = next((r.query for r in cited if r.query), None) or _short(question, 60)
    actions = [
        cl.Action(
            name="report_suggestion",
            payload={"kind": "quiz", "topic": notion, "course": course, "chapter": chapter},
            label=f"🎓 Me tester : {_short(notion)}",
        ),
        cl.Action(
            name="report_suggestion",
            payload={"kind": "exercise", "topic": notion},
            label=f"✏️ Exercice : {_short(notion)}",
        ),
    ]
    if course and chapter:
        actions.append(
            cl.Action(
                name="report_suggestion",
                payload={"kind": "sheet", "course": course, "chapter": chapter},
                label=f"📄 Fiche : {_short(chapter)}",
            )
        )
    if _CODE_WORDS.search(question):
        actions.append(
            cl.Action(
                name="report_suggestion",
                payload={"kind": "code", "topic": notion, "language": "Python"},
                label="💻 Exemple Python",
            )
        )
    return actions


async def _render_event(
    event: AgentEvent, msg: cl.Message, steps: dict[str, cl.Step], elements: list[cl.Text]
) -> FinalEvent | ErrorEvent | None:
    if isinstance(event, TokenEvent):
        await msg.stream_token(event.text)
    elif isinstance(event, ResetEvent):
        msg.content = ""
        await msg.update()
    elif isinstance(event, ToolStartEvent):
        step = cl.Step(name=event.label, type="tool", show_input=False)
        step.start = _now()
        await step.send()
        steps[event.call_id] = step
    elif isinstance(event, ToolEndEvent):
        step = steps.pop(event.call_id, None)
        if step is not None:
            await _close_step(step, event.summary, ok=event.ok)
    elif isinstance(event, SourcesEvent):
        elements.extend(source_elements(event.cited))
    elif isinstance(event, FinalEvent):
        msg.content = event.text
        msg.elements = elements
        await msg.update()
        return event
    elif isinstance(event, ErrorEvent):
        msg.content = event.message
        await msg.update()
        return event
    return None


async def _close_step(step: cl.Step, summary: str, *, ok: bool) -> None:
    step.output = summary if ok else f"⚠️ {summary}"
    step.end = _now()
    await step.update()


# ---------------------------------------------------------------------------
# Quiz
# ---------------------------------------------------------------------------


async def start_quiz(quiz: Quiz, *, user_id: str | None = None, thread_id: str | None = None) -> None:
    """Store the quiz in the session and ask its first question.

    Args:
        quiz: Generated quiz.
        user_id: Student login (persistence of the answers).
        thread_id: Chat thread id (persistence).
    """
    cl.user_session.set("quiz", quiz.model_dump())
    cl.user_session.set("quiz_data", quiz.to_legacy())
    cl.user_session.set("quiz_index", 0)
    cl.user_session.set("quiz_score", 0)
    cl.user_session.set("quiz_errors", [])
    cl.user_session.set("quiz_mode", "active")
    cl.user_session.set("quiz_user", user_id)
    cl.user_session.set("quiz_thread", thread_id)
    await ask_next_question()


_N_QUESTIONS = re.compile(r"\b(\d{1,2})\s*questions?\b", re.IGNORECASE)


def parse_quiz_request(text: str, default_n: int = 3) -> tuple[str, int]:
    """Split « stationnarité, 5 questions » into ``("stationnarité", 5)`` (1–10 questions)."""
    m = _N_QUESTIONS.search(text)
    if not m:
        return text.strip(), default_n
    n = max(1, min(int(m.group(1)), 10))
    topic = (text[: m.start()] + text[m.end() :]).strip(" ,;:-–—\t")
    return topic or text.strip(), n


def current_difficulty() -> str:
    """Difficulty of the next quiz for this session (adapted to the last score)."""
    return cl.user_session.get("quiz_difficulty") or "moyen"


async def ask_next_question() -> None:
    """Display the current quiz question (options in the message, letter buttons)."""
    quiz_data = cl.user_session.get("quiz_data") or []
    index = cl.user_session.get("quiz_index") or 0

    if index >= len(quiz_data):
        await _finish_quiz(len(quiz_data))
        return

    q = quiz_data[index]
    options = [_OPTION_PREFIX.sub("", o).strip() for o in q["options"][:4]]
    # Options are rendered in the message (Markdown + KaTeX); the buttons only carry the letter.
    listing = "\n".join(f"**{_LETTERS[i]})** {normalize_math(opt)}" for i, opt in enumerate(options))
    actions = [
        cl.Action(name="quiz_answer", payload={"value": str(i)}, label=_LETTERS[i], tooltip="Cliquez pour choisir")
        for i in range(len(options))
    ]
    header = f"✅ **Question {index + 1}/{len(quiz_data)}**"
    if index == 0:  # the real size and difficulty of the quiz, whatever the agent announced
        quiz = cl.user_session.get("quiz") or {}
        header = (
            f"🎓 **Quiz : {quiz.get('topic', '')}** — {len(quiz_data)} question(s), "
            f"difficulté « {quiz.get('difficulty', current_difficulty())} »\n\n" + header
        )
    await cl.Message(
        content=f"{header}\n\n{normalize_math(q['question'])}\n\n{listing}",
        actions=actions,
        author=AUTHOR,
    ).send()


async def _finish_quiz(n_questions: int) -> None:
    """Score message, adaptive difficulty, targeted-retry button."""
    score = cl.user_session.get("quiz_score") or 0
    errors = cl.user_session.get("quiz_errors") or []
    quiz = cl.user_session.get("quiz") or {}
    difficulty = quiz.get("difficulty") or current_difficulty()
    nxt = next_difficulty(difficulty, score, n_questions)
    cl.user_session.set("quiz_difficulty", nxt)
    cl.user_session.set("quiz_mode", None)

    lines = [f"🏆 **Quiz terminé !**\nVotre score : {score}/{n_questions} (difficulté « {difficulty} »)."]
    if nxt != difficulty:
        lines.append(f"Le prochain quiz sera de difficulté « {nxt} ».")
    actions = []
    if errors:
        lines.append(f"{len(errors)} question(s) à retravailler.")
        actions.append(cl.Action(name="quiz_retry_errors", payload={}, label="🔁 Refaire un quiz sur mes erreurs"))
    lines.append("Posez une autre question, demandez un exercice ou un nouveau quiz.")
    await cl.Message(content="\n".join(lines), actions=actions, author=AUTHOR).send()


async def answer_quiz(action: cl.Action) -> None:
    """Grade the clicked option, persist it and move on."""
    quiz_data = cl.user_session.get("quiz_data") or []
    index = cl.user_session.get("quiz_index") or 0
    score = cl.user_session.get("quiz_score") or 0
    if index >= len(quiz_data):
        return
    current = quiz_data[index]
    user_idx = int(action.payload["value"])
    correct_idx = current["correct_index"]
    explanation = normalize_math(current["explanation"])
    actions: list[cl.Action] = []
    if user_idx == correct_idx:
        score += 1
        cl.user_session.set("quiz_score", score)
        feedback = f"✅ **Correct !**\n\n{explanation}"
    else:
        clean_correct = normalize_math(_OPTION_PREFIX.sub("", current["options"][correct_idx]).strip())
        feedback = (
            f"❌ **Incorrect.**\nLa bonne réponse était **{_LETTERS[correct_idx]}** : {clean_correct}\n\n{explanation}"
        )
        errors = list(cl.user_session.get("quiz_errors") or [])
        errors.append(index)
        cl.user_session.set("quiz_errors", errors)
        actions.append(cl.Action(name="quiz_explain", payload={"index": index}, label="💬 M'expliquer davantage"))
    await action.remove()
    await _persist_answer(index, user_idx)
    await cl.Message(content=feedback, actions=actions, author=AUTHOR).send()
    cl.user_session.set("quiz_index", index + 1)
    await ask_next_question()


async def _persist_answer(index: int, answer_index: int) -> None:
    """Record the answer in ``quiz_results`` (silently skipped without a logged-in user)."""
    user = cl.user_session.get("quiz_user")
    quiz = cl.user_session.get("quiz") or {}
    if not user or not quiz:
        return
    q = quiz["questions"][index]
    await record_answer(
        QuizAnswer(
            user_identifier=user,
            thread_id=cl.user_session.get("quiz_thread"),
            quiz_id=quiz["quiz_id"],
            topic=quiz["topic"],
            question_index=index,
            question=q["question"],
            options=q["options"],
            correct_index=q["correct_index"],
            answer_index=answer_index,
            difficulty=quiz.get("difficulty"),
            level=cl.user_session.get("quiz_level"),
            course=quiz.get("course"),
            chapter=quiz.get("chapter"),
            sources=quiz.get("source_labels") or [],
        )
    )


def explain_prompt(index: int) -> str | None:
    """Student-side message asking the agent to explain a failed question."""
    quiz = cl.user_session.get("quiz") or {}
    questions = quiz.get("questions") or []
    if index >= len(questions):
        return None
    q = questions[index]
    correct = _OPTION_PREFIX.sub("", q["options"][q["correct_index"]]).strip()
    return (
        f"Dans le quiz sur « {quiz.get('topic', '')} », je me suis trompé à la question : « {q['question']} ». "
        f"La bonne réponse est « {correct} ». Explique-moi pourquoi, en t'appuyant sur le cours, "
        "et pourquoi les autres propositions sont fausses."
    )


async def explain_question(
    action: cl.Action, engine: ElaEngine, history: list[ChatTurn], *, session_id: str | None, user_id: str | None
) -> None:
    """Ask the agent to explain a failed question (targeted follow-up)."""
    await action.remove()
    prompt = explain_prompt(int(action.payload.get("index", -1)))
    if prompt is None:
        return
    await run_turn(engine, prompt, history, [], session_id=session_id, user_id=user_id)


async def retry_errors(action: cl.Action, engine: ElaEngine, *, user_id: str | None, thread_id: str | None) -> None:
    """Generate a new quiz targeting the questions failed in the last one."""
    await action.remove()
    quiz = cl.user_session.get("quiz") or {}
    errors = cl.user_session.get("quiz_errors") or []
    questions = quiz.get("questions") or []
    focus = [questions[i]["question"] for i in errors if i < len(questions)]
    if not focus:
        return
    msg_wait = cl.Message(
        content=f"🔁 Nouveau quiz sur **{quiz.get('topic', '')}**, ciblé sur vos erreurs...", author=AUTHOR
    )
    await msg_wait.send()
    try:
        new_quiz = await engine.generate_quiz(
            quiz.get("topic", ""), num_questions=max(3, len(focus)), difficulty=current_difficulty(), focus=focus
        )
    except GenerationError:
        new_quiz = None
    if new_quiz is None:
        msg_wait.content = "⚠️ Je n'ai pas réussi à générer un nouveau quiz ; réessayez avec `/start_quiz`."
        await msg_wait.update()
        return
    msg_wait.content = (
        f"🔁 Quiz de {len(new_quiz.questions)} questions ciblé sur vos erreurs (difficulté « {new_quiz.difficulty} »)."
    )
    await msg_wait.update()
    await start_quiz(new_quiz, user_id=user_id, thread_id=thread_id)


# ---------------------------------------------------------------------------
# Exercise
# ---------------------------------------------------------------------------


VERDICT_LABELS = {
    "correct": "✅ Étape correcte",
    "partiel": "🟡 Presque",
    "incorrect": "❌ Pas tout à fait",
    "hors_sujet": "🤔 Hors sujet",
}


def _exercise_actions(n_hints_left: int, *, completed: bool = False) -> list[cl.Action]:
    actions = []
    if not completed:
        actions.append(cl.Action(name="exercise_step", payload={}, label="✍️ Vérifier mon étape"))
        if n_hints_left > 0:
            actions.append(cl.Action(name="exercise_hint", payload={}, label=f"💡 Indice ({n_hints_left} restant(s))"))
    actions.append(cl.Action(name="exercise_solution", payload={}, label="📝 Solution"))
    return actions


def _hints_left() -> int:
    data = cl.user_session.get("exercise") or {}
    return len(data.get("hints") or []) - (cl.user_session.get("exercise_hint_index") or 0)


async def show_exercise(exercise: Exercise, *, user_id: str | None = None, thread_id: str | None = None) -> None:
    """Display the statement with the step / hint / solution buttons.

    Args:
        exercise: Generated exercise.
        user_id: Student login (persistence of the attempts).
        thread_id: Chat thread id (persistence).
    """
    cl.user_session.set("exercise", exercise.model_dump())
    cl.user_session.set("exercise_hint_index", 0)
    cl.user_session.set("exercise_attempts", [])
    cl.user_session.set("exercise_completed", False)
    cl.user_session.set("exercise_user", user_id)
    cl.user_session.set("exercise_thread", thread_id)
    cl.user_session.set("exercise_mode", None)
    await cl.Message(
        content=f"✏️ **Exercice : {exercise.title}**\n\n{normalize_math(exercise.statement)}\n\n"
        "_Résolvez-le étape par étape : « Vérifier mon étape » pour soumettre votre raisonnement, "
        "« Indice » si vous bloquez, « Solution » pour le corrigé._",
        actions=_exercise_actions(len(exercise.hints)),
        author=AUTHOR,
    ).send()


async def reveal_hint(action: cl.Action) -> None:
    """Send the next hint."""
    data = cl.user_session.get("exercise")
    if not data:
        return
    index = cl.user_session.get("exercise_hint_index") or 0
    hints = data["hints"]
    await action.remove()
    if index >= len(hints):
        return
    cl.user_session.set("exercise_hint_index", index + 1)
    left = len(hints) - index - 1
    await cl.Message(
        content=f"💡 **Indice {index + 1}/{len(hints)}** : {normalize_math(hints[index])}",
        actions=_exercise_actions(left),
        author=AUTHOR,
    ).send()


async def reveal_solution(action: cl.Action) -> None:
    """Send the worked solution and close the exercise."""
    data = cl.user_session.get("exercise")
    await action.remove()
    if not data:
        return
    cl.user_session.set("exercise", None)
    cl.user_session.set("exercise_mode", None)
    await cl.Message(content=f"📝 **Corrigé**\n\n{normalize_math(data['solution'])}", author=AUTHOR).send()


async def ask_for_step(action: cl.Action) -> None:
    """Switch the next student message to a step submission."""
    await action.remove()
    if not cl.user_session.get("exercise"):
        return
    cl.user_session.set("exercise_mode", "awaiting_step")
    await cl.Message(
        content="✍️ Écrivez votre étape (raisonnement, formule ou calcul) ; je vous dis si elle tient "
        "et je vous oriente vers la suite, sans donner la solution. Tapez `stop` pour revenir au chat.",
        author=AUTHOR,
    ).send()


async def check_student_step(engine: ElaEngine, attempt: str) -> None:
    """Evaluate the submitted step (Socratic feedback) and offer the next move."""
    cl.user_session.set("exercise_mode", None)
    data = cl.user_session.get("exercise")
    if not data:
        return
    if attempt.strip().lower() in {"stop", "annuler", "quitter"}:
        await cl.Message(
            content="D'accord, l'exercice reste ouvert : utilisez les boutons quand vous voulez reprendre.",
            actions=_exercise_actions(_hints_left()),
            author=AUTHOR,
        ).send()
        return
    exercise = Exercise.model_validate(data)
    attempts = list(cl.user_session.get("exercise_attempts") or [])
    revealed = cl.user_session.get("exercise_hint_index") or 0
    thinking = cl.Message(content="🔎 Je regarde votre étape...", author=AUTHOR)
    await thinking.send()
    feedback = await engine.check_step(
        exercise, attempt, revealed_hints=revealed, previous=[(a["attempt"], a["verdict"]) for a in attempts]
    )
    if feedback is None:
        thinking.content = "⚠️ Je n'ai pas pu évaluer cette étape ; réessayez ou demandez un indice."
        thinking.actions = _exercise_actions(_hints_left())
        await thinking.update()
        return
    attempts.append({"attempt": attempt, "verdict": feedback.verdict})
    cl.user_session.set("exercise_attempts", attempts)
    await _persist_attempt(exercise, attempt, feedback, len(attempts), revealed)

    label = VERDICT_LABELS.get(feedback.verdict, feedback.verdict)
    parts = [f"**{label}**", normalize_math(feedback.feedback)]
    if feedback.exercise_completed:
        parts.append("🎉 **Exercice terminé !** Vous avez atteint le résultat attendu.")
        cl.user_session.set("exercise_completed", True)  # the solution stays available for comparison
    elif feedback.guiding_question:
        parts.append(f"➡️ {normalize_math(feedback.guiding_question)}")
    thinking.content = "\n\n".join(parts)
    thinking.actions = _exercise_actions(_hints_left(), completed=feedback.exercise_completed)
    await thinking.update()  # Message.update() also sends the actions not yet bound to the message


async def _persist_attempt(exercise: Exercise, attempt: str, feedback, index: int, revealed: int) -> None:
    user = cl.user_session.get("exercise_user")
    if not user:
        return
    await record_attempt(
        ExerciseAttempt(
            user_identifier=user,
            thread_id=cl.user_session.get("exercise_thread"),
            exercise_id=exercise.exercise_id,
            topic=exercise.topic or exercise.title,
            title=exercise.title,
            level=cl.user_session.get("quiz_level"),
            attempt_index=index,
            attempt=attempt,
            verdict=feedback.verdict,
            completed=feedback.exercise_completed,
            hints_revealed=revealed,
        )
    )


# ---------------------------------------------------------------------------
# Student report (/bilan)
# ---------------------------------------------------------------------------


async def show_report(user_id: str | None) -> None:
    """Send the student's report with one button per suggestion."""
    if not user_id:
        await cl.Message(content="📈 Le bilan nécessite un compte étudiant connecté.", author=AUTHOR).send()
        return
    report = await student_report(user_id)
    actions = [
        cl.Action(name="report_suggestion", payload={"kind": s.kind, "topic": s.topic}, label=s.label)
        for s in report.suggestions
    ]
    await cl.Message(content=render_report(report), actions=actions, author=AUTHOR).send()


async def follow_suggestion(
    action: cl.Action, engine: ElaEngine, *, user_id: str | None, thread_id: str | None
) -> None:
    """Start the quiz or exercise suggested by the report."""
    await action.remove()
    kind, topic = action.payload.get("kind"), action.payload.get("topic", "")
    if not topic and kind != "sheet":
        return
    if kind == "exercise":
        await launch_exercise(engine, topic, user_id=user_id, thread_id=thread_id)
        return
    if kind == "sheet":
        await launch_review_sheet(
            engine,
            action.payload.get("course", ""),
            action.payload.get("chapter", ""),
            user_id=user_id,
            thread_id=thread_id,
        )
        return
    if kind == "code":
        await launch_code(engine, topic, action.payload.get("language") or "Python")
        return
    await launch_quiz(
        engine,
        topic,
        n=3,
        difficulty=current_difficulty(),
        user_id=user_id,
        thread_id=thread_id,
        course=action.payload.get("course"),
        chapter=action.payload.get("chapter"),
    )


# ---------------------------------------------------------------------------
# Direct generation (commands and buttons, outside the agent)
# ---------------------------------------------------------------------------


def nothing_on_topic(level: str, topic: str) -> str:
    """Student-facing message when the visible course has nothing on ``topic``."""
    elsewhere = courses_elsewhere(topic, level)
    if elsewhere:
        where = ", ".join(f"**{course}** (niveau {lvl})" for course, lvl in elsewhere)
        return (
            f"⚠️ « {topic} » n'est pas au programme des cours accessibles à votre niveau ({level}) : "
            f"ce sujet est traité dans {where}. Choisissez un sujet de vos cours."
        )
    return (
        f"⚠️ Je n'ai rien trouvé sur « {topic} » dans vos cours. "
        "Essayez avec les termes du cours (par exemple un titre de chapitre ou de section)."
    )


def quiz_retry_action(topic: str, n: int, difficulty: str) -> cl.Action:
    """Button regenerating a quiz with the same parameters (no agent turn)."""
    return cl.Action(
        name="quiz_retry",
        payload={"topic": topic, "n": n, "difficulty": difficulty},
        label="🔁 Réessayer le quiz",
    )


async def launch_quiz(
    engine: ElaEngine,
    topic: str,
    *,
    n: int,
    difficulty: str,
    user_id: str | None,
    thread_id: str | None,
    intro: str | None = None,
    course: str | None = None,
    chapter: str | None = None,
) -> bool:
    """Generate a quiz directly and ask its first question.

    Args:
        engine: Session engine.
        topic: Quiz topic.
        n: Number of questions requested.
        difficulty: Difficulty of the quiz.
        user_id: Student login (persistence).
        thread_id: Chat thread id (persistence).
        intro: Waiting message (default: « Génération d'un quiz sur … »).
        course: Restrict the passages to one course (suggestions).
        chapter: Restrict the passages to one chapter (suggestions).

    Returns:
        ``True`` when a quiz was started.
    """
    msg_wait = cl.Message(content=intro or f"🔍 Génération d'un quiz sur **{topic}**...", author=AUTHOR)
    await msg_wait.send()
    try:
        quiz = await engine.generate_quiz(topic, num_questions=n, difficulty=difficulty, course=course, chapter=chapter)
    except GenerationError:
        msg_wait.content = "⚠️ La génération du quiz a échoué (le modèle n'a pas produit de questions exploitables)."
        msg_wait.actions = [quiz_retry_action(topic, n, difficulty)]
        await msg_wait.update()
        return False
    if quiz is None:
        msg_wait.content = nothing_on_topic(engine.user_level, topic)
        await msg_wait.update()
        return False
    got = len(quiz.questions)
    short = f" — {got} sur les {n} demandées ont pu être générées" if got < n else ""
    msg_wait.content = f"🎓 Quiz de {got} questions sur **{topic}** (difficulté « {quiz.difficulty} »){short}."
    await msg_wait.update()
    await start_quiz(quiz, user_id=user_id, thread_id=thread_id)
    return True


async def retry_quiz(action: cl.Action, engine: ElaEngine, *, user_id: str | None, thread_id: str | None) -> None:
    """Regenerate the quiz whose generation failed (same topic, count and difficulty)."""
    await action.remove()
    p = action.payload or {}
    topic = p.get("topic", "")
    if not topic:
        return
    await launch_quiz(
        engine,
        topic,
        n=int(p.get("n") or 3),
        difficulty=p.get("difficulty") or current_difficulty(),
        user_id=user_id,
        thread_id=thread_id,
        intro=f"🔁 Nouvelle tentative : quiz sur **{topic}**...",
    )


async def launch_exercise(engine: ElaEngine, topic: str, *, user_id: str | None, thread_id: str | None) -> bool:
    """Generate an exercise directly and display it.

    Returns:
        ``True`` when an exercise was shown.
    """
    msg_wait = cl.Message(
        content=f"🔍 Analyse de vos cours sur **{topic}** et rédaction de l'exercice...", author=AUTHOR
    )
    await msg_wait.send()
    try:
        exercise = await engine.generate_exercise(topic)
    except GenerationError:
        msg_wait.content = "⚠️ La rédaction de l'exercice a échoué. Réessayez avec `/exercise`."
        await msg_wait.update()
        return False
    if exercise is None:
        msg_wait.content = nothing_on_topic(engine.user_level, topic)
        await msg_wait.update()
        return False
    msg_wait.content = f"✏️ Exercice prêt (extraits {', '.join(exercise.sources)})."
    await msg_wait.update()
    await show_exercise(exercise, user_id=user_id, thread_id=thread_id)
    return True


# ---------------------------------------------------------------------------
# Review sheets (/fiche)
# ---------------------------------------------------------------------------


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", normalize(text)).strip("-")[:60] or "fiche"


async def show_review_sheet(sheet: ReviewSheet, *, user_id: str | None = None, thread_id: str | None = None) -> None:
    """Display a review sheet (Markdown + KaTeX) with its downloadable ``.md`` file and follow-up buttons.

    Args:
        sheet: The sheet.
        user_id: Student login (the consultation is logged for the report).
        thread_id: Chat thread id.
    """
    markdown = sheet.to_markdown()
    file = cl.File(
        name=f"fiche_{_slug(sheet.course)}_{_slug(sheet.chapter)}.md",
        content=markdown.encode("utf-8"),
        mime="text/markdown",
        display="inline",
    )
    actions = [
        cl.Action(
            name="report_suggestion",
            payload={"kind": "quiz", "topic": sheet.chapter, "course": sheet.course, "chapter": sheet.chapter},
            label="🎓 Me tester sur ce chapitre",
        ),
        cl.Action(
            name="report_suggestion",
            payload={"kind": "exercise", "topic": sheet.title},
            label="✏️ Un exercice",
        ),
    ]
    await cl.Message(content=normalize_math(markdown), elements=[file], actions=actions, author=AUTHOR).send()
    if user_id:
        await record_question(
            QuestionRecord(
                user_identifier=user_id,
                thread_id=thread_id,
                level=sheet.level or None,
                question=f"Fiche de révision : {sheet.course} › {sheet.chapter}",
                course=sheet.course,
                chapter=sheet.chapter,
                n_sources=len(sheet.slides),
            )
        )


async def launch_review_sheet(
    engine: ElaEngine, course: str, chapter: str, *, user_id: str | None, thread_id: str | None
) -> bool:
    """Generate (or fetch from the cache) and display the sheet of a chapter.

    Returns:
        ``True`` when a sheet was shown.
    """
    cl.user_session.set("sheet_mode", None)
    msg_wait = cl.Message(content=f"📄 Préparation de la fiche **{course} › {chapter}**...", author=AUTHOR)
    await msg_wait.send()
    try:
        sheet = await engine.generate_review_sheet(course, chapter)
    except ReviewLookupError as exc:
        msg_wait.content = f"⚠️ {exc}"
        await msg_wait.update()
        return False
    except GenerationError:
        msg_wait.content = "⚠️ La génération de la fiche a échoué. Réessayez avec `/fiche`."
        await msg_wait.update()
        return False
    msg_wait.content = f"📄 Fiche **{sheet.title}** ({len(sheet.slides)} slides résumées)."
    await msg_wait.update()
    await show_review_sheet(sheet, user_id=user_id, thread_id=thread_id)
    return True


def _visible_outlines(engine: ElaEngine) -> dict:
    try:
        return get_outlines(getattr(engine, "user_level", "ALL"))
    except Exception:  # store unreachable
        logger.warning("Course outlines unavailable", exc_info=True)
        return {}


async def start_sheet_picker(engine: ElaEngine) -> None:
    """``/fiche``: list the visible courses as buttons (or ask for the chapter when there is one course)."""
    outlines = _visible_outlines(engine)
    if not outlines:
        await cl.Message(content="⚠️ Aucun cours n'est disponible pour votre niveau.", author=AUTHOR).send()
        return
    cl.user_session.set("sheet_mode", "waiting")
    if len(outlines) == 1:
        course = next(iter(outlines))
        await _ask_chapter(outlines[course])
        return
    actions = [
        cl.Action(name="sheet_course", payload={"course": course}, label=f"📚 {course.replace('_', ' ')}")
        for course in outlines
    ]
    await cl.Message(
        content="📄 **Fiche de révision**\nDe quel cours ? (ou écrivez *cours, chapitre* — ex. *UTS, chapitre 2*)",
        actions=actions,
        author=AUTHOR,
    ).send()


async def _ask_chapter(outline) -> None:
    actions = [
        cl.Action(
            name="sheet_chapter",
            payload={"course": outline.course, "chapter": ch.title},
            label=f"{i}. {_short(ch.title, 48)}",
        )
        for i, ch in enumerate(outline.chapters, start=1)
    ]
    await cl.Message(
        content=f"📄 Quel chapitre de **{outline.course.replace('_', ' ')}** ?", actions=actions, author=AUTHOR
    ).send()


async def pick_sheet_course(action: cl.Action, engine: ElaEngine) -> None:
    """Course button of the picker → chapter buttons."""
    await action.remove()
    outlines = _visible_outlines(engine)
    outline = outlines.get(action.payload.get("course", ""))
    if outline is None:
        await cl.Message(content="⚠️ Cours introuvable.", author=AUTHOR).send()
        return
    await _ask_chapter(outline)


async def pick_sheet_chapter(
    action: cl.Action, engine: ElaEngine, *, user_id: str | None, thread_id: str | None
) -> None:
    """Chapter button of the picker → the sheet."""
    await action.remove()
    await launch_review_sheet(
        engine,
        action.payload.get("course", ""),
        action.payload.get("chapter", ""),
        user_id=user_id,
        thread_id=thread_id,
    )


def parse_sheet_request(text: str, outlines: dict) -> tuple[str, str] | None:
    """Split « UTS, chapitre 2 » / « racines unitaires (UTS) » into ``(course, chapter)``.

    The course is recognised by any of its aliases in the text; with a single
    visible course it may be omitted. ``None`` when no course is identified.
    """
    norm = normalize(text)
    course = None
    for name, outline in outlines.items():
        aliases = sorted((a for a in outline.aliases if len(a) >= 3), key=len, reverse=True)
        hit = next((a for a in aliases if re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", norm)), None)
        if hit:
            course = name
            norm = re.sub(rf"(?<![a-z0-9]){re.escape(hit)}(?![a-z0-9])", " ", norm)
            break
    if course is None:
        if len(outlines) != 1:
            return None
        course = next(iter(outlines))
    chapter = re.sub(r"\s+", " ", norm).strip(" ,;:-–—()\t")
    return (course, chapter) if chapter else None


async def handle_sheet_request(engine: ElaEngine, text: str, *, user_id: str | None, thread_id: str | None) -> None:
    """Text typed while the picker waits (« UTS, chapitre 2 »)."""
    parsed = parse_sheet_request(text, _visible_outlines(engine))
    if parsed is None:
        await cl.Message(
            content="⚠️ Indiquez le cours et le chapitre (ex. *UTS, chapitre 2*) ou utilisez les boutons.", author=AUTHOR
        ).send()
        return
    await launch_review_sheet(engine, parsed[0], parsed[1], user_id=user_id, thread_id=thread_id)


# ---------------------------------------------------------------------------
# Code workshop (direct)
# ---------------------------------------------------------------------------


async def launch_code(engine: ElaEngine, topic: str, language: str = "Python") -> None:
    """Generate the example script for ``topic``, show it and offer to run it (Python, sandbox configured)."""
    msg_load = cl.Message(content=f"💻 Génération du script **{language}** pour **{topic}**...", author=AUTHOR)
    await msg_load.send()
    answer = await engine.generate_code(topic, language)
    msg_load.content = normalize_math(answer)
    code = sandbox.extract_code(answer, language.lower())
    if code and _runnable(language):
        code = await _ensure_parses(engine, code, topic=topic, language=language)
        remember_code(code, topic=topic, language=language)
        if sandbox.extract_code(msg_load.content, language.lower()) != code:  # fixed before display
            msg_load.content += f"\n\n🔧 **Script vérifié et corrigé (syntaxe)**\n\n```{language.lower()}\n{code}\n```"
        msg_load.actions = [run_code_action()]
    await msg_load.update()
    cl.user_session.set("code_mode", None)


async def _ensure_parses(engine: ElaEngine, code: str, *, topic: str, language: str, attempts: int = 2) -> str:
    """Fix syntax errors / truncation before showing a script (no execution, no quota)."""
    history: list[str] = []
    for _ in range(attempts):
        error = sandbox.syntax_error(code, language)
        if error is None:
            return code
        logger.info("Generated script does not parse; fixing before display: %s", error.splitlines()[0])
        try:
            code = await engine.fix_code(code, error, topic=topic, language=language, history=history)
        except GenerationError:
            return code
        history.append(error)
    return code


# ---------------------------------------------------------------------------
# Code execution (sandbox)
# ---------------------------------------------------------------------------


def _runnable(language: str) -> bool:
    return language.lower() == "python" and sandbox_enabled()


def run_code_action() -> cl.Action:
    """« ▶️ Exécuter » button (the script itself is kept in the session)."""
    return cl.Action(name="code_run", payload={}, label="▶️ Exécuter")


def remember_code(code: str, *, topic: str, language: str) -> None:
    """Keep the last script of the session for the run / fix buttons (new script: attempts reset)."""
    cl.user_session.set("code_last", {"code": code, "topic": topic, "language": language, "attempts": []})


def _note_attempt(last: dict, error: str) -> None:
    """Record a failed attempt of the current script (regression guard for the next fix)."""
    attempts = list(last.get("attempts") or [])
    attempts.append(error)
    last["attempts"] = attempts[-6:]
    cl.user_session.set("code_last", last)


async def show_code_output(result: RunResult, *, code: str, topic: str, language: str = "Python") -> None:
    """Display a run (status, stdout, stderr tail, figures) with the fix / re-run buttons.

    Args:
        result: Run result.
        code: Script that was run (kept for « Corriger »).
        topic: Subject (persistence, fix prompt).
        language: Script language.
    """
    images = [
        cl.Image(name=f"figure_{i}", content=png, display="inline", size="large")
        for i, png in enumerate(result.images, start=1)
    ]
    actions: list[cl.Action] = []
    if not result.ok and _runnable(language):
        last = cl.user_session.get("code_last") or {}
        if last.get("code") != code:
            remember_code(code, topic=topic, language=language)
        cl.user_session.set("code_last_error", result.stderr)
        actions.append(cl.Action(name="code_fix", payload={}, label="🔧 Corriger le script"))
    await cl.Message(content=sandbox.format_result(result), elements=images, actions=actions, author=AUTHOR).send()


async def run_last_code(action: cl.Action, engine: ElaEngine, *, user_id: str | None, thread_id: str | None) -> None:
    """« ▶️ Exécuter » : run the last script of the session in the sandbox."""
    await action.remove()
    last = cl.user_session.get("code_last") or {}
    code = last.get("code")
    if not code:
        await cl.Message(content="⚠️ Aucun script à exécuter dans cette session.", author=AUTHOR).send()
        return
    remaining = await sandbox.quota_remaining(user_id)
    step = cl.Step(name="▶️ Exécution du script dans le bac à sable", type="tool", show_input=False)
    step.start = _now()
    await step.send()
    try:
        result = await engine.run_code(
            code,
            user_id=user_id,
            thread_id=thread_id,
            origin=last.get("origin") or "workshop",
            topic=last.get("topic"),
            language=last.get("language") or "Python",
        )
    except sandbox.QuotaExceededError as exc:
        await _close_step(step, f"quota journalier atteint ({exc.limit})", ok=False)
        await cl.Message(
            content=f"⚠️ Vous avez atteint votre quota de **{exc.limit} exécutions par jour**. Réessayez demain.",
            author=AUTHOR,
        ).send()
        return
    except SandboxUnavailableError:
        await _close_step(step, "bac à sable indisponible", ok=False)
        await cl.Message(content="⚠️ Le bac à sable d'exécution est indisponible pour le moment.", author=AUTHOR).send()
        return
    summary = f"{result.status} en {result.elapsed_ms / 1000:.1f} s"
    if remaining is not None:
        remaining -= 1
        summary += f" — {max(0, remaining)} exécution(s) restante(s) aujourd'hui"
    await _close_step(step, summary, ok=result.ok)
    topic, language = last.get("topic") or "", last.get("language") or "Python"
    if result.status == "error":
        _note_attempt(last, result.stderr)
        if get_settings().code_auto_fix and not last.get("auto_fixed") and (remaining is None or remaining > 0):
            code, result = await _auto_fix_and_rerun(
                engine, code, result, last, user_id=user_id, thread_id=thread_id, topic=topic, language=language
            )
    await show_code_output(result, code=code, topic=topic, language=language)
    if result.ok:
        await interpret_run(engine, code, result, topic=topic, language=language)


async def _auto_fix_and_rerun(
    engine: ElaEngine,
    code: str,
    result: RunResult,
    last: dict,
    *,
    user_id: str | None,
    thread_id: str | None,
    topic: str,
    language: str,
) -> tuple[str, RunResult]:
    """One automatic fix + re-run after a failed execution (shown as a step; the error stays visible on failure)."""
    step = cl.Step(name="🔧 Correction automatique et nouvelle exécution", type="tool", show_input=False)
    step.start = _now()
    await step.send()
    history = [e for e in (last.get("attempts") or [])[:-1]]
    try:
        fixed = await engine.fix_code(code, result.stderr, topic=topic, language=language, history=history)
    except GenerationError:
        await _close_step(step, "correction impossible", ok=False)
        return code, result
    last.update({"code": fixed, "origin": "fix", "auto_fixed": True})
    cl.user_session.set("code_last", last)
    try:
        rerun = await engine.run_code(
            fixed, user_id=user_id, thread_id=thread_id, origin="fix", topic=topic, language=language
        )
    except (sandbox.QuotaExceededError, SandboxUnavailableError) as exc:
        await _close_step(step, f"ré-exécution impossible ({exc.__class__.__name__})", ok=False)
        return fixed, result
    if rerun.status == "error":
        _note_attempt(last, rerun.stderr)
    await _close_step(step, f"{rerun.status} en {rerun.elapsed_ms / 1000:.1f} s", ok=rerun.ok)
    await cl.Message(
        content=f"🔧 **Script corrigé automatiquement**\n\n```{language.lower()}\n{fixed}\n```", author=AUTHOR
    ).send()
    return fixed, rerun


async def interpret_run(engine: ElaEngine, code: str, result: RunResult, *, topic: str, language: str) -> None:
    """Interpret the real output of a run: results table, commentary with course citations.

    When the script printed no number, says so and offers « 🔧 Compléter le script ».
    """
    step = cl.Step(name="🧠 Interprétation des résultats", type="tool", show_input=False)
    step.start = _now()
    await step.send()
    try:
        interp = await engine.interpret_run(code, result, topic=topic, language=language)
    except GenerationError:
        await _close_step(step, "échec de l'interprétation", ok=False)
        await cl.Message(
            content="⚠️ Je n'ai pas réussi à interpréter la sortie. Vous pouvez me poser une question dessus.",
            author=AUTHOR,
        ).send()
        return
    if interp.needs_output:
        await _close_step(step, "aucun résultat chiffré imprimé", ok=False)
        remember_code(code, topic=topic, language=language)
        cl.user_session.set("code_last_error", sandbox.NO_OUTPUT_HINT)
        actions = (
            [cl.Action(name="code_fix", payload={}, label="🔧 Compléter le script")] if _runnable(language) else []
        )
        await cl.Message(content=interp.text, actions=actions, author=AUTHOR).send()
        return
    await _close_step(step, f"{len(interp.cited)} extrait(s) du cours cité(s)", ok=True)
    await cl.Message(content=interp.text, elements=source_elements(interp.cited), author=AUTHOR).send()


async def fix_last_code(action: cl.Action, engine: ElaEngine) -> None:
    """« 🔧 Corriger » : ask the model to repair the failing script, show it with « ▶️ Exécuter »."""
    await action.remove()
    last = cl.user_session.get("code_last") or {}
    code, error = last.get("code"), cl.user_session.get("code_last_error") or ""
    if not code:
        await cl.Message(content="⚠️ Aucun script à corriger dans cette session.", author=AUTHOR).send()
        return
    language = last.get("language") or "Python"
    completing = error == sandbox.NO_OUTPUT_HINT
    msg = cl.Message(
        content="🔧 Complément du script..." if completing else "🔧 Correction du script...", author=AUTHOR
    )
    await msg.send()
    history = [e for e in (last.get("attempts") or []) if e != error]
    try:
        fixed = await engine.fix_code(code, error, topic=last.get("topic") or "", language=language, history=history)
    except GenerationError:
        msg.content = "⚠️ Je n'ai pas réussi à corriger le script. Vous pouvez le modifier et me le renvoyer."
        await msg.update()
        return
    cl.user_session.set("code_last", {**last, "code": fixed, "origin": "fix", "auto_fixed": False})
    title = "Script complété" if completing else "Script corrigé"
    msg.content = f"🔧 **{title}**\n\n```{language.lower()}\n{fixed}\n```"
    msg.actions = [run_code_action()]
    await msg.update()
