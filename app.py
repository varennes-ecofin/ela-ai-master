# app.py
import os
import re
import sys
import json
import asyncio
import aiofiles
from datetime import datetime
from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from chainlit.element import Element

# --- FIX WINDOWS (OBLIGATOIRE POUR POSTGRES/ASYNCPG) ---
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dotenv import load_dotenv
import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.data.utils import queue_until_user_message
from chainlit.types import ThreadDict

from chainlit.server import app as fastapi_app
from starlette.staticfiles import StaticFiles
from urllib.parse import quote

from langchain_core.messages import HumanMessage, AIMessage

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import bcrypt

# Import de votre logique RAG
from main_ela import ELA_Bot

# chainlit run app.py -w 

load_dotenv()

# ============================================================
# SECTION 1 : MONTAGE STATIQUE DES FICHIERS LOCAUX
# ============================================================
# Chainlit est bâti sur FastAPI/Starlette. On monte le dossier
# de stockage comme route HTTP pour que le navigateur puisse
# charger les images via /local_files/{object_key}.
# ============================================================


STORAGE_BASE_PATH = os.path.abspath(".files_ela")
LOCAL_FILES_ROUTE = "/local_files"  # Préfixe HTTP pour servir les fichiers

os.makedirs(STORAGE_BASE_PATH, exist_ok=True)
fastapi_app.mount(LOCAL_FILES_ROUTE, StaticFiles(directory=STORAGE_BASE_PATH), name="local_files")

# On force cette route en PREMIÈRE position
# Cela empêche le routeur "catch-all" de Chainlit d'intercepter la requête
# et de renvoyer du HTML à la place de l'image.
try:
    # On récupère la route qu'on vient d'ajouter (la dernière de la liste)
    local_files_route = fastapi_app.router.routes.pop()
    # On la réinsère à l'index 0 (priorité maximale)
    fastapi_app.router.routes.insert(0, local_files_route)
    print("✅ Route /local_files déplacée en priorité haute.")
except IndexError:
    print("⚠️ Erreur lors du réordonnancement des routes.")


# ============================================================
# SECTION 2 : STOCKAGE LOCAL (Custom Storage Client)
# ============================================================
class LocalStorageClient:
    """
    Local storage compatible with Chainlit's DataLayer.
    
    Files are stored on disk under STORAGE_BASE_PATH and served
    via the /local_files/ static route mounted on FastAPI.
    Cloud providers (S3, Azure) return presigned HTTPS URLs;
    we return relative HTTP paths that the static mount resolves.
    """

    def __init__(self, base_path: str = ".files_ela"):
        self.base_path = os.path.abspath(base_path)
        os.makedirs(self.base_path, exist_ok=True)

    async def upload_file(
        self,
        object_key: str,
        data: bytes,
        mime: str = "application/octet-stream",
        overwrite: bool = True,
    ) -> dict:
        """Write file to disk and return an HTTP-accessible URL."""
        path_parts = object_key.split("/")
        file_path = os.path.join(self.base_path, *path_parts)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(data)

        # URL relative pour le frontend (servie par StaticFiles)
        url_key = object_key.replace("\\", "/")
        return {
            "object_key": object_key,
            "url": f"{LOCAL_FILES_ROUTE}/{quote(url_key, safe='/')}",
        }

    async def get_read_url(self, object_key: str) -> str:
        """Return an HTTP URL for the frontend to load the file."""
        return f"{LOCAL_FILES_ROUTE}/{object_key}"

    def resolve_to_disk_path(self, object_key: str) -> str:
        """Resolve an object_key to its absolute filesystem path."""
        path_parts = object_key.split("/")
        return os.path.join(self.base_path, *path_parts)

    async def delete_file(self, object_key: str):
        """Delete the physical file and clean up empty parent directories."""
        # file_path = os.path.join(self.base_path, object_key)
        file_path = self.resolve_to_disk_path(object_key)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ Fichier supprimé : {file_path}")

                # Nettoyage des dossiers vides (remonte 2 niveaux max)
                directory = os.path.dirname(file_path)
                for _ in range(2):
                    try:
                        os.rmdir(directory)
                        print(f"📂 Dossier vide supprimé : {directory}")
                        directory = os.path.dirname(directory)
                        if os.path.abspath(directory) == self.base_path:
                            break
                    except OSError:
                        break
        except Exception as e:
            print(f"⚠️ Erreur lors de la suppression : {e}")


# ============================================================
# SECTION 3 : DATA LAYER CUSTOM
# ============================================================
# Chainlit's SQLAlchemyDataLayer.create_element() fetches the
# element URL via aiohttp when element.path is not set.
# Relative URLs like /local_files/... crash aiohttp.
# This subclass intercepts local URLs and reads from disk.
# ============================================================
class LocalSQLAlchemyDataLayer(SQLAlchemyDataLayer):
    """
    Extended SQLAlchemyDataLayer that handles local file URLs.

    Overrides create_element() to read files from disk when the
    element URL points to our local static route, avoiding the
    aiohttp InvalidUrlClientError on relative URLs.
    """

    @queue_until_user_message()
    async def create_element(self, element: "Element"):
        """Persist an element, reading local files from disk instead of HTTP."""
        from chainlit.logger import logger
        from chainlit.element import ElementDict

        if self.show_logger:
            logger.info(f"SQLAlchemy: create_element, element_id = {element.id}")

        if not self.storage_provider:
            logger.warning(
                "SQLAlchemy: create_element error. "
                "No blob_storage_client is configured!"
            )
            return
        if not element.for_id:
            return

        content: Optional[Union[bytes, str]] = None

        if element.path:
            # --- CAS NORMAL : fichier temporaire uploadé par l'utilisateur ---
            async with aiofiles.open(element.path, "rb") as f:
                content = await f.read()

        elif element.url:
            # --- FIX : intercepter les URLs locales ---
            if element.url.startswith(LOCAL_FILES_ROUTE + "/"):
                # Extraire l'object_key depuis l'URL locale
                object_key = element.url[len(LOCAL_FILES_ROUTE) + 1:]
                # Décoder les caractères URL-encodés (espaces, accents, etc.)
                from urllib.parse import unquote
                object_key = unquote(object_key)
                disk_path = os.path.join(STORAGE_BASE_PATH, object_key)

                if os.path.exists(disk_path):
                    async with aiofiles.open(disk_path, "rb") as f:
                        content = await f.read()
                else:
                    logger.warning(
                        f"Local file not found: {disk_path} "
                        f"(from URL: {element.url})"
                    )
                    content = None
            else:
                # URL externe classique : fetch via aiohttp (comportement par défaut)
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(element.url) as response:
                        if response.status == 200:
                            content = await response.read()
                        else:
                            content = None

        elif element.content:
            content = element.content
        else:
            raise ValueError("Element url, path or content must be provided")

        if content is None:
            raise ValueError("Content is None, cannot upload file")

        user_id = await self._get_user_id_by_thread(element.thread_id) or "unknown"
        # --- MODIFICATION : STANDARDISATION DU NOM ---
        # 1. On récupère l'extension d'origine (.png, .jpg)
        _, ext = os.path.splitext(element.name or "")
        if not ext:
            ext = ".bin" # Fallback si pas d'extension

        # 2. On génère un nom purement ASCII basé sur l'heure
        # Format : Screenshot_2026-02-09_16-15-30.png
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_filename = f"Screenshot_{timestamp}{ext}"
        
        file_object_key = f"{user_id}/{element.id}/{safe_filename}"

        if not element.mime:
            element.mime = "application/octet-stream"

        uploaded_file = await self.storage_provider.upload_file(
            object_key=file_object_key,
            data=content,
            mime=element.mime,
            overwrite=True,
        )
        if not uploaded_file:
            raise ValueError(
                "SQLAlchemy Error: create_element, "
                "Failed to persist data in storage_provider"
            )

        element_dict: ElementDict = element.to_dict()
        element_dict["url"] = uploaded_file.get("url")
        element_dict["objectKey"] = uploaded_file.get("object_key")

        element_dict_cleaned = {
            k: v for k, v in element_dict.items() if v is not None
        }
        if "props" in element_dict_cleaned:
            element_dict_cleaned["props"] = json.dumps(
                element_dict_cleaned["props"]
            )

        columns = ", ".join(
            f'"{col}"' for col in element_dict_cleaned.keys()
        )
        placeholders = ", ".join(
            f":{col}" for col in element_dict_cleaned.keys()
        )
        updates = ", ".join(
            f'"{col}" = :{col}'
            for col in element_dict_cleaned.keys()
            if col != "id"
        )
        query = (
            f"INSERT INTO elements ({columns}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT (id) DO UPDATE SET {updates};"
        )
        await self.execute_sql(query=query, parameters=element_dict_cleaned)


    async def get_thread(self, thread_id: str):
        """
        Override to fix element URL resolution on chat resume.

        The base SQLAlchemyDataLayer.get_thread() does not reliably
        include element URLs when loading thread history. This override
        explicitly queries elements and regenerates their URLs from
        the local storage provider, ensuring images display correctly
        after the browser cache expires.

        See: https://github.com/Chainlit/chainlit/issues/2484
        """
        from chainlit.logger import logger

        thread = await super().get_thread(thread_id)
        if thread is None:
            return None

        # Force-load elements for this thread from DB
        elements_query = """
            SELECT * FROM elements WHERE "threadId" = :thread_id
        """
        elements_results = await self.execute_sql(
            query=elements_query,
            parameters={"thread_id": thread_id},
        )

        if isinstance(elements_results, list) and elements_results:
            for elem in elements_results:
                # If URL is missing but objectKey exists, regenerate it
                if not elem.get("url") and elem.get("objectKey"):
                    if self.storage_provider:
                        elem["url"] = await self.storage_provider.get_read_url(
                            object_key=elem["objectKey"],
                        )
                # Deserialize props if stored as JSON string
                if isinstance(elem.get("props"), str):
                    try:
                        elem["props"] = json.loads(elem["props"])
                    except (json.JSONDecodeError, TypeError):
                        pass

            thread["elements"] = elements_results
            if self.show_logger:
                logger.info(
                    f"get_thread: loaded {len(elements_results)} "
                    f"elements for thread {thread_id}"
                )

        return thread

    async def get_element(self, thread_id: str, element_id: str):
        """
        Retrieve a single element with proper URL resolution.

        Ensures the element URL points to a valid local file path
        so that images render correctly in the frontend.

        See: https://github.com/Chainlit/chainlit/issues/1205
        """
        query = """
            SELECT * FROM elements
            WHERE "id" = :element_id AND "threadId" = :thread_id
        """
        parameters = {"element_id": element_id, "thread_id": thread_id}
        records = await self.execute_sql(query=query, parameters=parameters)

        if not records or not isinstance(records, list):
            return None

        elem = dict(records[0])

        # Resolve URL from objectKey if missing
        if not elem.get("url") and elem.get("objectKey"):
            if self.storage_provider:
                elem["url"] = await self.storage_provider.get_read_url(
                    object_key=elem["objectKey"],
                )

        # Deserialize props if stored as JSON string
        if isinstance(elem.get("props"), str):
            try:
                elem["props"] = json.loads(elem["props"])
            except (json.JSONDecodeError, TypeError):
                pass

        return elem


# ============================================================
# SECTION 4 : GESTION UTILISATEURS (DB + BCRYPT)
# ============================================================
# Les identifiants sont désormais stockés en base PostgreSQL
# avec hash bcrypt. La variable ELA_AUTH_DATA n'est plus utilisée.
# ============================================================

async def _query_db(query_str: str, params: dict = None):
    """Execute an async SQL query and return result rows.

    Reuses the DATABASE_URL from the environment (same DB as Chainlit).
    Creates a short-lived engine per call — acceptable for auth/quota
    checks which happen infrequently.

    Args:
        query_str: SQL query with :named placeholders.
        params: Dict of parameter values.

    Returns:
        List of row mappings, or empty list on error.
    """
    db_url = os.getenv("DATABASE_URL")
    engine = create_async_engine(db_url)
    AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(text(query_str), params or {})
            rows = result.mappings().all()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"⚠️ DB query error: {e}")
        return []
    finally:
        await engine.dispose()


async def _update_db(query_str: str, params: dict = None):
    """Execute an async SQL UPDATE/INSERT and commit.

    Args:
        query_str: SQL statement with :named placeholders.
        params: Dict of parameter values.
    """
    db_url = os.getenv("DATABASE_URL")
    engine = create_async_engine(db_url)
    AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text(query_str), params or {})
            await session.commit()
    except Exception as e:
        print(f"⚠️ DB update error: {e}")
    finally:
        await engine.dispose()


async def get_user_profile(identifier: str) -> dict:
    """Fetch full user profile from the database.

    Args:
        identifier: The user's login username.

    Returns:
        Dict with user profile fields, or empty dict if not found.
    """
    rows = await _query_db(
        """
        SELECT id, identifier, role, level, is_active, daily_quota
        FROM users
        WHERE identifier = :identifier
        """,
        {"identifier": identifier},
    )
    return rows[0] if rows else {}


async def check_quota(identifier: str) -> dict:
    """Check if user has remaining quota for today.

    Args:
        identifier: The user's login username.

    Returns:
        Dict with keys: allowed (bool), used (int), limit (int or None).
    """
    profile = cl.user_session.get("user_profile", {})
    quota = profile.get("daily_quota")

    # NULL quota = unlimited (admin)
    if quota is None:
        return {"allowed": True, "used": 0, "limit": None}

    rows = await _query_db(
        """
        SELECT COUNT(s.id) AS msg_count
        FROM steps s
        JOIN threads t ON s."threadId" = t.id
        WHERE t."userIdentifier" = :identifier
        AND s.type = 'user_message'
        AND s."createdAt"::DATE = CURRENT_DATE
        """,
        {"identifier": identifier},
    )

    used = rows[0]["msg_count"] if rows else 0
    return {"allowed": used < quota, "used": used, "limit": quota}


# ============================================================
# SECTION 5 : ACTIVATION DU DATA LAYER
# ============================================================
@cl.data_layer
def get_data_layer():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("❌ DATABASE_URL manquante dans le fichier .env")

    storage = LocalStorageClient(base_path=".files_ela")

    # Utilise notre DataLayer custom au lieu de SQLAlchemyDataLayer
    return LocalSQLAlchemyDataLayer(
        conninfo=database_url,
        storage_provider=storage,
    )


# --- AUTHENTIFICATION (DB + BCRYPT) ---
@cl.password_auth_callback
async def auth_callback(username, password):
    """Verify credentials against the users table with bcrypt.

    Also updates last_login timestamp on successful authentication.

    Args:
        username: Login identifier.
        password: Plaintext password to verify.

    Returns:
        cl.User on success, None on failure.
    """
    rows = await _query_db(
        """
        SELECT identifier, password_hash, role, is_active
        FROM users
        WHERE identifier = :username
        """,
        {"username": username},
    )

    if not rows:
        return None

    user = rows[0]

    # Account must be active
    if not user["is_active"]:
        return None

    # Verify bcrypt hash
    if not user["password_hash"]:
        return None

    if not bcrypt.checkpw(
        password.encode("utf-8"),
        user["password_hash"].encode("utf-8"),
    ):
        return None

    # Update last_login
    await _update_db(
        "UPDATE users SET last_login = NOW() WHERE identifier = :username",
        {"username": username},
    )

    print(f"✅ Login réussi : {username} (rôle: {user['role']})")
    return cl.User(identifier=username)


# --- DÉMARRAGE DE SESSION ---
@cl.on_chat_start
async def start():
    """Initialize bot and load user profile on new session."""
    print("🚀 Démarrage nouvelle session")

    # Load user profile from DB
    user = cl.context.session.user
    if user:
        profile = await get_user_profile(user.identifier)
        cl.user_session.set("user_profile", profile)
        print(f"👤 Profil chargé : {user.identifier} | rôle={profile.get('role')} | niveau={profile.get('level')}")

    user_level = profile.get("level", "ALL") if user else "ALL"
    ela_instance = ELA_Bot(user_level=user_level)
    cl.user_session.set("ela_bot", ela_instance)


# --- REPRISE DE CONVERSATION ---
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Reload bot and user profile when resuming a conversation."""
    print(f"🔄 Reprise de la conversation {thread['id']}")

    # Reload user profile
    user = cl.context.session.user
    if user:
        profile = await get_user_profile(user.identifier)
        cl.user_session.set("user_profile", profile)

    user_level = profile.get("level", "ALL") if user else "ALL"
    ela_instance = ELA_Bot(user_level=user_level)
    cl.user_session.set("ela_bot", ela_instance)


# ============================================================
# SECTION 6 : GESTION DES MESSAGES
# ============================================================
@cl.on_message
async def main(message: cl.Message):
    ela_bot = cl.user_session.get("ela_bot")
    
    # 0. QUOTA CHECK — Bloque si l'utilisateur a épuisé son quota
    user = cl.context.session.user
    if user:
        quota_info = await check_quota(user.identifier)
        if not quota_info["allowed"]:
            await cl.Message(
                content=(
                    f"⚠️ **Quota journalier atteint** ({quota_info['used']}/{quota_info['limit']} messages).\n\n"
                    "Votre quota sera réinitialisé demain. "
                    "Contactez votre superviseur si vous avez besoin de messages supplémentaires."
                )
            ).send()
            return

    # 1. GESTION DES COMMANDES SPECIALES

    # A. Démarrage du Quiz
    if message.content == "/start_quiz":
        await rename_current_thread("🎓 Nouveau Quiz")
        await cl.Message(
            content="🎓 **Mode Quiz activé !**\nSur quel concept du cours "
            "voulez-vous vous tester ? (ex: *MCO, Séries Temporelles, "
            "Tests de racine unitaire...*)"
        ).send()
        cl.user_session.set("quiz_mode", "waiting_topic")
        return

    # B. Atelier Code
    if message.content == "/code_workshop":
        await rename_current_thread("💻 Atelier Code")
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

    # 2. LOGIQUE DU QUIZ (Machine à états)
    quiz_mode = cl.user_session.get("quiz_mode")

    if quiz_mode == "waiting_topic":
        topic = message.content
        safe_topic = (topic[:25] + "..") if len(topic) > 25 else topic
        await rename_current_thread(f"🎓 Quiz : {safe_topic}")

        msg_wait = cl.Message(
            content=f"🔍 Analyse de vos cours sur **{topic}** et génération des questions..."
        )
        await msg_wait.send()

        quiz_data = await ela_bot.generate_quiz_json(topic, num_questions=3)

        if not quiz_data:
            msg_wait.content = "⚠️ Je n'ai pas trouvé assez d'informations dans le cours pour ce sujet."
            await msg_wait.update()
            return

        cl.user_session.set("quiz_data", quiz_data)
        cl.user_session.set("quiz_index", 0)
        cl.user_session.set("quiz_score", 0)
        cl.user_session.set("quiz_mode", "active")
        await ask_next_question()
        return

    if quiz_mode == "active":
        await cl.Message(content="💡 Utilisez les boutons ci-dessus pour répondre !").send()
        return

    # 3. GESTION DU CODE WORKSHOP
    if cl.user_session.get("code_mode") == "waiting_topic":
        topic = message.content
        language = cl.user_session.get("code_lang_choice")
        await rename_current_thread(f"💻 Code : {topic}")

        msg_load = cl.Message(
            content=f"Génération du script **{language}** pour **{topic}**..."
        )
        await msg_load.send()

        response = await ela_bot.generate_practical_code(topic, language)
        msg_load.content = response
        await msg_load.update()

        cl.user_session.set("code_mode", None)
        return

    # 4. GESTION DES IMAGES
    image_path = None

    if message.elements:
        file = message.elements[0]
        if "image" in file.mime:
            image_path = file.path
        else:
            await cl.Message(
                content="⚠️ Désolé, je n'accepte que les images (.png, .jpg, .jpeg)."
            ).send()
            return

    # 4.2 Reconstruction Historique
    context_messages = cl.chat_context.get()
    history_langchain = []

    for msg_ctx in context_messages:
        if msg_ctx.id == message.id:
            continue
        if msg_ctx.type == "user_message":
            history_langchain.append(HumanMessage(content=msg_ctx.content))
        elif msg_ctx.type == "assistant_message":
            history_langchain.append(AIMessage(content=msg_ctx.content))

    # 4.3 Appel à  ELA avec streaming
    msg = cl.Message(content="", author="ELA AI 🤖")
    await msg.send()

    async for token in ela_bot.ask(
        question=message.content,
        chat_history=history_langchain,
        image_path=image_path,
    ):
        await msg.stream_token(token)

    await msg.update()


# ============================================================
# SECTION 8 : UTILITAIRES (Renommage, Starters, Quiz, Code)
# ============================================================
@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"Chatbot": "ELA", "Assistant": "ELA 🤖 "}
    return rename_dict.get(orig_author, orig_author)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Générer un quiz",
            message="/start_quiz",
            icon="/public/quiz.svg",
        ),
        cl.Starter(
            label="Atelier Code",
            message="/code_workshop",
            icon="/public/terminal.svg",
        ),
    ]


async def ask_next_question():
    """Affiche la question actuelle sous forme de message avec boutons."""
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
    letters = ["A", "B", "C", "D"]
    num_options = min(len(q["options"]), 4)

    for i in range(num_options):
        raw_option = q["options"][i]
        clean_option = re.sub(r"^[A-D0-9][\)\.]\s*", "", raw_option).strip()
        actions.append(
            cl.Action(
                name="quiz_answer",
                payload={"value": str(i)},
                label=f"{letters[i]}) {clean_option}",
                description="Cliquez pour choisir",
            )
        )

    await cl.Message(
        content=f"✅ **Question {index + 1}/{len(quiz_data)}**\n\n{q['question']}",
        actions=actions,
    ).send()


@cl.action_callback("quiz_answer")
async def on_quiz_answer(action: cl.Action):
    """Gère le clic sur un bouton de réponse."""
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
        letters = ["A", "B", "C", "D"]
        raw_correct = current_q["options"][correct_idx]
        clean_correct = re.sub(r"^[A-D0-9][\)\.]\s*", "", raw_correct).strip()
        feedback = (
            f"❌ **Incorrect.**\nLa bonne réponse était "
            f"**{letters[correct_idx]}** : {clean_correct}.\n\n"
            f"_{current_q['explanation']}_"
        )

    await action.remove()
    await cl.Message(content=feedback).send()

    cl.user_session.set("quiz_index", index + 1)
    await asyncio.sleep(1)
    await ask_next_question()


async def rename_current_thread(new_name: str):
    """Renomme la conversation actuelle dans la base de données."""
    thread_id = cl.context.session.thread_id
    if not thread_id:
        return

    db_url = os.getenv("DATABASE_URL")
    engine = create_async_engine(db_url)
    AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    try:
        async with AsyncSessionLocal() as session:
            query = text('UPDATE threads SET name = :name WHERE id = :id')
            await session.execute(query, {"name": new_name, "id": thread_id})
            await session.commit()
    except Exception as e:
        print(f"⚠️ Erreur renommage thread : {e}")
    finally:
        await engine.dispose()


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