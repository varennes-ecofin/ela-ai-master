"""ELA AI — Chainlit chat application.

Run from the project root (``.chainlit/``, ``public/`` and ``chainlit.md``
live there)::

    chainlit run apps/chat/app.py -w

Responsibilities of this module: authentication, session setup, message
routing (chat / quiz / code workshop) and UI. Everything else lives in the
``ela`` package (retrieval, engine, database, configuration).
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys

# --- Windows: asyncpg requires the selector event loop ---
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import chainlit as cl
from chainlit.types import ThreadDict
from data_layer import LocalSQLAlchemyDataLayer, LocalStorageClient

from ela.config import get_settings
from ela.db import threads as threads_db
from ela.db import users as users_db
from ela.db.users import UserProfile
from ela.engine import ElaEngine
from ela.history import bounded_history, to_turns
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

    # 1. Special commands
    if message.content == "/start_quiz":
        await threads_db.rename_thread(cl.context.session.thread_id, "🎓 Nouveau Quiz")
        await cl.Message(
            content="🎓 **Mode Quiz activé !**\nSur quel concept du cours "
            "voulez-vous vous tester ? (ex: *MCO, Séries Temporelles, "
            "Tests de racine unitaire...*)"
        ).send()
        cl.user_session.set("quiz_mode", "waiting_topic")
        return

    if message.content == "/code_workshop":
        await threads_db.rename_thread(cl.context.session.thread_id, "💻 Atelier Code")
        actions = [
            cl.Action(name="code_lang", value="Python", label="Python", payload={"value": "Python"}),
            cl.Action(name="code_lang", value="R", label="R", payload={"value": "R"}),
        ]
        await cl.Message(
            content="💻 **Bienvenue dans l'Atelier Code !**\n\n"
            "Je peux générer pour vous des exemples pratiques basés sur vos cours.\n"
            "Quel langage souhaitez-vous utiliser ?",
            actions=actions,
        ).send()
        return

    # 2. Quiz state machine
    quiz_mode = cl.user_session.get("quiz_mode")
    if quiz_mode == "waiting_topic":
        await _start_quiz(engine, message.content)
        return
    if quiz_mode == "active":
        await cl.Message(content="💡 Utilisez les boutons ci-dessus pour répondre !").send()
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

    # 5. Bounded history + streaming answer
    history = bounded_history(to_turns(cl.chat_context.get(), exclude_id=message.id))

    msg = cl.Message(content="", author="ELA AI 🤖")
    await msg.send()
    async for token in engine.ask(question=message.content, chat_history=history, images=images):
        await msg.stream_token(token)
    await msg.update()


# ============================================================
# QUIZ
# ============================================================
_LETTERS = ["A", "B", "C", "D"]
_OPTION_PREFIX = re.compile(r"^[A-D0-9][\)\.]\s*")


async def _start_quiz(engine: ElaEngine, topic: str) -> None:
    """Generate the quiz for ``topic`` and ask the first question."""
    safe_topic = (topic[:25] + "..") if len(topic) > 25 else topic
    await threads_db.rename_thread(cl.context.session.thread_id, f"🎓 Quiz : {safe_topic}")

    msg_wait = cl.Message(content=f"🔍 Analyse de vos cours sur **{topic}** et génération des questions...")
    await msg_wait.send()

    quiz_data = await engine.generate_quiz(topic, num_questions=3)

    if quiz_data and isinstance(quiz_data[0], dict) and quiz_data[0].get("error") == "domain":
        msg_wait.content = "⚠️ Ce sujet sort du cadre du cours (séries temporelles, économétrie financière)."
        await msg_wait.update()
        cl.user_session.set("quiz_mode", None)
        return
    if not quiz_data:
        msg_wait.content = "⚠️ Je n'ai pas trouvé assez d'informations dans le cours pour ce sujet."
        await msg_wait.update()
        cl.user_session.set("quiz_mode", None)
        return

    cl.user_session.set("quiz_data", quiz_data)
    cl.user_session.set("quiz_index", 0)
    cl.user_session.set("quiz_score", 0)
    cl.user_session.set("quiz_mode", "active")
    await _ask_next_question()


async def _ask_next_question() -> None:
    """Display the current question with one action button per option."""
    quiz_data = cl.user_session.get("quiz_data")
    index = cl.user_session.get("quiz_index")

    if index >= len(quiz_data):
        score = cl.user_session.get("quiz_score")
        await cl.Message(
            content=f"🏆 **Quiz terminé !**\nVotre score : {score}/{len(quiz_data)}\n\n"
            "Posez une autre question ou tapez `/start_quiz` pour recommencer."
        ).send()
        cl.user_session.set("quiz_mode", None)
        return

    q = quiz_data[index]
    actions = []
    for i, raw_option in enumerate(q["options"][:4]):
        clean_option = _OPTION_PREFIX.sub("", raw_option).strip()
        actions.append(
            cl.Action(
                name="quiz_answer",
                payload={"value": str(i)},
                label=f"{_LETTERS[i]}) {clean_option}",
                description="Cliquez pour choisir",
            )
        )
    await cl.Message(
        content=f"✅ **Question {index + 1}/{len(quiz_data)}**\n\n{q['question']}",
        actions=actions,
    ).send()


@cl.action_callback("quiz_answer")
async def on_quiz_answer(action: cl.Action):
    """Grade the clicked answer and move to the next question."""
    quiz_data = cl.user_session.get("quiz_data")
    index = cl.user_session.get("quiz_index")
    score = cl.user_session.get("quiz_score")

    user_idx = int(action.payload["value"])
    current_q = quiz_data[index]
    correct_idx = current_q["correct_index"]

    if user_idx == correct_idx:
        score += 1
        cl.user_session.set("quiz_score", score)
        feedback = f"✅ **Correct !**\n_{current_q['explanation']}_"
    else:
        clean_correct = _OPTION_PREFIX.sub("", current_q["options"][correct_idx]).strip()
        feedback = (
            f"❌ **Incorrect.**\nLa bonne réponse était "
            f"**{_LETTERS[correct_idx]}** : {clean_correct}.\n\n"
            f"_{current_q['explanation']}_"
        )

    await action.remove()
    await cl.Message(content=feedback).send()
    cl.user_session.set("quiz_index", index + 1)
    await asyncio.sleep(1)
    await _ask_next_question()


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

    msg_load = cl.Message(content=f"Génération du script **{language}** pour **{topic}**...")
    await msg_load.send()
    msg_load.content = await engine.generate_code(topic, language)
    await msg_load.update()
    cl.user_session.set("code_mode", None)
