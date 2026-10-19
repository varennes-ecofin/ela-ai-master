"""ELA AI — Chainlit chat application.

Run from the project root (``.chainlit/``, ``public/`` and ``chainlit.md``
live there)::

    chainlit run apps/chat/app.py -w

Responsibilities of this module: authentication, session setup, message
routing (chat / quiz / exercise / code workshop). The rendering of the
agent events lives in ``ui.py``; everything else in the ``ela`` package
(agent, retrieval, database, configuration).
"""

from __future__ import annotations

import asyncio
import logging
import sys

# --- Windows: asyncpg requires the selector event loop ---
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import chainlit as cl
import ui
from chainlit.types import ThreadDict
from data_layer import LocalSQLAlchemyDataLayer, LocalStorageClient

from ela.config import get_settings
from ela.db import threads as threads_db
from ela.db import users as users_db
from ela.db.users import UserProfile
from ela.engine import ElaEngine
from ela.history import to_turns
from ela.llm import SUPPORTED_IMAGE_MIMES
from ela.log import configure_logging
from ela.retrieval import resources

configure_logging()
logger = logging.getLogger("ela.chat")
settings = get_settings()

# Pre-load embeddings / store / reranker so the first student does not wait.
if settings.warmup_on_start:
    resources.warmup(block=False)


# ============================================================
# DATA LAYER
# ============================================================
@cl.data_layer
def get_data_layer():
    """Chainlit persistence: PostgreSQL + local file storage."""
    return LocalSQLAlchemyDataLayer(
        conninfo=settings.async_database_url,
        storage_provider=LocalStorageClient(),
    )


# ============================================================
# AUTHENTICATION
# ============================================================
@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    """Verify credentials against the ``users`` table (bcrypt)."""
    profile = await users_db.authenticate(username, password)
    if profile is None:
        return None
    return cl.User(identifier=profile.identifier, metadata={"role": profile.role, "level": profile.level})


# ============================================================
# SESSION
# ============================================================
async def _init_session() -> None:
    """Load the user profile and attach an engine to the session."""
    user = cl.context.session.user
    profile: UserProfile | None = None
    if user:
        profile = await users_db.get_profile(user.identifier)
    cl.user_session.set("user_profile", profile)

    level = profile.level if profile else "ALL"
    cl.user_session.set("quiz_level", level)
    cl.user_session.set("ela_engine", ElaEngine(user_level=level))
    logger.info("Session ready: user=%s level=%s", user.identifier if user else "-", level)


@cl.on_chat_start
async def on_chat_start():
    await _init_session()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    logger.info("Resuming thread %s", thread["id"])
    await _init_session()


def _engine() -> ElaEngine:
    return cl.user_session.get("ela_engine")


def _user_id() -> str | None:
    user = cl.context.session.user
    return user.identifier if user else None


# ============================================================
# STARTERS & AUTHOR NAMES
# ============================================================
@cl.author_rename
def rename(orig_author: str):
    return {"Chatbot": "ELA", "Assistant": "ELA 🤖 "}.get(orig_author, orig_author)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(label="Générer un quiz", message="/start_quiz", icon="/public/quiz.svg"),
        cl.Starter(label="Faire un exercice", message="/exercise", icon="/public/quiz.svg"),
        cl.Starter(label="Fiche de révision", message="/fiche", icon="/public/quiz.svg"),
        cl.Starter(label="Mon bilan", message="/bilan", icon="/public/quiz.svg"),
        cl.Starter(label="Atelier Code", message="/code_workshop", icon="/public/terminal.svg"),
    ]


# ============================================================
# MESSAGE ROUTING
# ============================================================
@cl.on_message
async def main(message: cl.Message):
    engine = _engine()
    profile: UserProfile | None = cl.user_session.get("user_profile")

    # 0. Daily quota
    if profile is not None:
        quota = await users_db.check_quota(profile)
        if not quota.allowed:
            await cl.Message(
                content=(
                    f"⚠️ **Quota journalier atteint** ({quota.used}/{quota.limit} messages).\n\n"
                    "Votre quota sera réinitialisé demain. "
                    "Contactez votre superviseur si vous avez besoin de messages supplémentaires."
                )
            ).send()
            return

    # 1. Special commands (a command cancels a pending « /fiche » picker)
    if message.content.startswith("/"):
        cl.user_session.set("sheet_mode", None)
    if message.content == "/start_quiz":
        await threads_db.rename_thread(cl.context.session.thread_id, "🎓 Nouveau Quiz")
        await cl.Message(
            content="🎓 **Mode Quiz activé !**\nSur quel concept du cours voulez-vous vous tester ? "
            "(ex: *stationnarité, test ADF, modèle GARCH...*) — précisez le nombre de questions si vous voulez "
            "(ex: *test ADF, 5 questions* ; 3 par défaut)"
        ).send()
        cl.user_session.set("quiz_mode", "waiting_topic")
        return

    if message.content == "/bilan":
        await ui.show_report(_user_id())
        return

    if message.content == "/fiche":
        await threads_db.rename_thread(cl.context.session.thread_id, "📄 Fiche de révision")
        await ui.start_sheet_picker(engine)
        return

    if message.content == "/exercise":
        await threads_db.rename_thread(cl.context.session.thread_id, "✏️ Exercice")
        await cl.Message(
            content="✏️ **Mode Exercice activé !**\nSur quelle notion du cours voulez-vous un exercice ? "
            "(ex: *prévision d'un AR(1), test de cointégration, VaR...*)"
        ).send()
        cl.user_session.set("exercise_mode", "waiting_topic")
        return

    if message.content == "/code_workshop":
        await threads_db.rename_thread(cl.context.session.thread_id, "💻 Atelier Code")
        actions = [
            cl.Action(name="code_lang", label="Python", payload={"value": "Python"}),
            cl.Action(name="code_lang", label="R", payload={"value": "R"}),
        ]
        await cl.Message(
            content="💻 **Bienvenue dans l'Atelier Code !**\n\n"
            "Je peux générer pour vous des exemples pratiques basés sur vos cours.\n"
            "Quel langage souhaitez-vous utiliser ?",
            actions=actions,
        ).send()
        return

    # 2. Quiz / exercise state machines
    quiz_mode = cl.user_session.get("quiz_mode")
    if quiz_mode == "waiting_topic":
        await _start_quiz(engine, message.content)
        return
    if quiz_mode == "active":
        await cl.Message(content="💡 Utilisez les boutons ci-dessus pour répondre !").send()
        return
    exercise_mode = cl.user_session.get("exercise_mode")
    if exercise_mode == "waiting_topic":
        await _start_exercise(engine, message.content)
        return
    if exercise_mode == "awaiting_step":
        await ui.check_student_step(engine, message.content)
        return
    if cl.user_session.get("sheet_mode") == "waiting":
        await ui.handle_sheet_request(
            engine, message.content, user_id=_user_id(), thread_id=cl.context.session.thread_id
        )
        return

    # 3. Code workshop
    if cl.user_session.get("code_mode") == "waiting_topic":
        await _run_code_workshop(engine, message.content)
        return

    # 4. Images (several accepted, real MIME forwarded)
    images: list[tuple[str, str | None]] = []
    for element in message.elements or []:
        mime = (getattr(element, "mime", None) or "").lower()
        path = getattr(element, "path", None)
        if mime in SUPPORTED_IMAGE_MIMES and path:
            images.append((path, mime))
        else:
            await cl.Message(
                content=(
                    f"⚠️ Fichier ignoré : **{getattr(element, 'name', '?')}**. "
                    "Je n'accepte que les images (.png, .jpg, .jpeg, .webp, .gif)."
                )
            ).send()
            return
    if len(images) > settings.max_images_per_message:
        await cl.Message(
            content=f"⚠️ Merci de joindre au plus {settings.max_images_per_message} images par message."
        ).send()
        return

    # 5. Agent turn (steps, sources panel, streaming) — history is bounded by the engine
    history = to_turns(cl.chat_context.get(), exclude_id=message.id)
    await ui.run_turn(
        engine,
        message.content,
        history,
        images,
        session_id=cl.context.session.thread_id,
        user_id=_user_id(),
    )


# ============================================================
# QUIZ
# ============================================================
async def _start_quiz(engine: ElaEngine, request: str) -> None:
    """Generate the quiz for the requested topic (« sujet, 5 questions ») and ask the first question."""
    topic, n_questions = ui.parse_quiz_request(request)
    safe_topic = (topic[:25] + "..") if len(topic) > 25 else topic
    await threads_db.rename_thread(cl.context.session.thread_id, f"🎓 Quiz : {safe_topic}")
    cl.user_session.set("quiz_mode", None)
    await ui.launch_quiz(
        engine,
        topic,
        n=n_questions,
        difficulty=ui.current_difficulty(),
        user_id=_user_id(),
        thread_id=cl.context.session.thread_id,
        intro=f"🔍 Analyse de vos cours sur **{topic}** et génération des questions...",
    )


@cl.action_callback("quiz_explain")
async def on_quiz_explain(action: cl.Action):
    history = to_turns(cl.chat_context.get())
    await ui.explain_question(action, _engine(), history, session_id=cl.context.session.thread_id, user_id=_user_id())


@cl.action_callback("quiz_retry_errors")
async def on_quiz_retry_errors(action: cl.Action):
    await ui.retry_errors(action, _engine(), user_id=_user_id(), thread_id=cl.context.session.thread_id)


@cl.action_callback("quiz_retry")
async def on_quiz_retry(action: cl.Action):
    await ui.retry_quiz(action, _engine(), user_id=_user_id(), thread_id=cl.context.session.thread_id)


@cl.action_callback("quiz_answer")
async def on_quiz_answer(action: cl.Action):
    await ui.answer_quiz(action)


# ============================================================
# EXERCISE
# ============================================================
async def _start_exercise(engine: ElaEngine, topic: str) -> None:
    """Generate an exercise for ``topic`` and display it."""
    safe_topic = (topic[:25] + "..") if len(topic) > 25 else topic
    await threads_db.rename_thread(cl.context.session.thread_id, f"✏️ Exercice : {safe_topic}")
    cl.user_session.set("exercise_mode", None)
    await ui.launch_exercise(engine, topic, user_id=_user_id(), thread_id=cl.context.session.thread_id)


@cl.action_callback("report_suggestion")
async def on_report_suggestion(action: cl.Action):
    await ui.follow_suggestion(action, _engine(), user_id=_user_id(), thread_id=cl.context.session.thread_id)


@cl.action_callback("sheet_course")
async def on_sheet_course(action: cl.Action):
    await ui.pick_sheet_course(action, _engine())


@cl.action_callback("sheet_chapter")
async def on_sheet_chapter(action: cl.Action):
    await ui.pick_sheet_chapter(action, _engine(), user_id=_user_id(), thread_id=cl.context.session.thread_id)


@cl.action_callback("code_run")
async def on_code_run(action: cl.Action):
    await ui.run_last_code(action, _engine(), user_id=_user_id(), thread_id=cl.context.session.thread_id)


@cl.action_callback("code_fix")
async def on_code_fix(action: cl.Action):
    await ui.fix_last_code(action, _engine())


@cl.action_callback("exercise_step")
async def on_exercise_step(action: cl.Action):
    await ui.ask_for_step(action)


@cl.action_callback("exercise_hint")
async def on_exercise_hint(action: cl.Action):
    await ui.reveal_hint(action)


@cl.action_callback("exercise_solution")
async def on_exercise_solution(action: cl.Action):
    await ui.reveal_solution(action)


# ============================================================
# CODE WORKSHOP
# ============================================================
@cl.action_callback("code_lang")
async def on_code_lang(action: cl.Action):
    lang = action.payload["value"]
    cl.user_session.set("code_lang_choice", lang)
    cl.user_session.set("code_mode", "waiting_topic")
    await action.remove()
    await cl.Message(
        content=f"C'est noté pour **{lang}** !\n\n"
        "Quel modèle ou concept voulez-vous implémenter ? "
        "(ex: *MCO, VAR, ARCH, Test de Student...*)"
    ).send()


async def _run_code_workshop(engine: ElaEngine, topic: str) -> None:
    """Generate the example script for ``topic`` in the chosen language."""
    language = cl.user_session.get("code_lang_choice") or "Python"
    await threads_db.rename_thread(cl.context.session.thread_id, f"💻 Code : {topic}")
    await ui.launch_code(engine, topic, language)
