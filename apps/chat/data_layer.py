"""Local file storage and PostgreSQL data layer for the Chainlit app.

* :class:`LocalStorageClient` stores uploaded files under ``STORAGE_DIR``
  and serves them through a static route mounted on Chainlit's FastAPI app.
* :class:`LocalSQLAlchemyDataLayer` extends Chainlit's SQLAlchemy data layer
  so that elements with local URLs are read from disk (no HTTP round trip)
  and resolved correctly when a thread is resumed.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING
from urllib.parse import quote, unquote

import aiofiles
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.data.utils import queue_until_user_message
from chainlit.server import app as fastapi_app
from starlette.staticfiles import StaticFiles

from ela.config import get_settings

if TYPE_CHECKING:
    from chainlit.element import Element

logger = logging.getLogger(__name__)

# ============================================================
# SECTION 1 : MONTAGE STATIQUE DES FICHIERS LOCAUX
# ============================================================
# Chainlit est bâti sur FastAPI/Starlette. On monte le dossier
# de stockage comme route HTTP pour que le navigateur puisse
# charger les images via /local_files/{object_key}.
# ============================================================


STORAGE_BASE_PATH = os.path.abspath(get_settings().storage_dir)
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
    logger.debug("Route /local_files moved to top priority.")
except IndexError:
    logger.warning("Could not reorder routes for /local_files.")


# ============================================================
# SECTION 2 : STOCKAGE LOCAL (Custom Storage Client)
# ============================================================
class LocalStorageClient:
    """Local storage compatible with Chainlit's DataLayer.

    Files are stored on disk under STORAGE_BASE_PATH and served
    via the /local_files/ static route mounted on FastAPI.
    Cloud providers (S3, Azure) return presigned HTTPS URLs;
    we return relative HTTP paths that the static mount resolves.
    """

    def __init__(self, base_path: str | None = None):
        base_path = base_path or get_settings().storage_dir
        self.base_path = os.path.abspath(base_path)
        os.makedirs(self.base_path, exist_ok=True)

    async def upload_file(
        self,
        object_key: str,
        data: bytes | str,
        mime: str = "application/octet-stream",
        overwrite: bool = True,
    ) -> dict:
        """Write file to disk and return an HTTP-accessible URL.

        ``data`` may be text (``cl.Text`` elements such as the sources panel):
        it is stored as UTF-8.
        """
        path_parts = object_key.split("/")
        file_path = os.path.join(self.base_path, *path_parts)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(data.encode("utf-8") if isinstance(data, str) else data)

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
                logger.info("File deleted: %s", file_path)

                # Nettoyage des dossiers vides (remonte 2 niveaux max)
                directory = os.path.dirname(file_path)
                for _ in range(2):
                    try:
                        os.rmdir(directory)
                        logger.debug("Empty directory removed: %s", directory)
                        directory = os.path.dirname(directory)
                        if os.path.abspath(directory) == self.base_path:
                            break
                    except OSError:
                        break
        except Exception as e:
            logger.warning("Delete failed: %s", e)


# ============================================================
# SECTION 3 : DATA LAYER CUSTOM
# ============================================================
# Chainlit's SQLAlchemyDataLayer.create_element() fetches the
# element URL via aiohttp when element.path is not set.
# Relative URLs like /local_files/... crash aiohttp.
# This subclass intercepts local URLs and reads from disk.
# ============================================================
# Step columns that only move forward (see ``LocalSQLAlchemyDataLayer.create_step``).
def _keep_if_empty(col: str) -> str:
    return f'CASE WHEN EXCLUDED."{col}" IS NULL OR EXCLUDED."{col}" = \'\' THEN steps."{col}" ELSE EXCLUDED."{col}" END'


_FORWARD_ONLY = {
    "output": _keep_if_empty("output"),
    "input": _keep_if_empty("input"),
    "end": 'COALESCE(EXCLUDED."end", steps."end")',
    "streaming": 'CASE WHEN steps."end" IS NOT NULL THEN FALSE ELSE EXCLUDED."streaming" END',
}


class LocalSQLAlchemyDataLayer(SQLAlchemyDataLayer):
    """Extended SQLAlchemyDataLayer that handles local file URLs.

    Overrides create_element() to read files from disk when the
    element URL points to our local static route, avoiding the
    aiohttp InvalidUrlClientError on relative URLs.
    """

    @queue_until_user_message()
    async def create_step(self, step_dict):
        """Upsert a step without letting a late *create* erase a finished step.

        Chainlit persists ``send()`` and ``update()`` as independent background
        tasks; when a step finishes quickly (tool call, streamed answer) the
        initial insert can land after the final update and overwrite its
        ``output`` / ``end`` with empty values. Those columns only move
        forward here: an empty incoming value never replaces a stored one.
        """
        from chainlit.logger import logger

        await self.update_thread(step_dict["threadId"])
        if self.show_logger:
            logger.info(f"SQLAlchemy: create_step, step_id={step_dict.get('id')}")

        step_dict["showInput"] = str(step_dict.get("showInput", "")).lower() if "showInput" in step_dict else None
        parameters = {
            key: value
            for key, value in step_dict.items()
            if value is not None and not (isinstance(value, dict) and not value)
        }
        parameters["metadata"] = json.dumps(step_dict.get("metadata", {}))
        parameters["generation"] = json.dumps(step_dict.get("generation", {}))
        columns = ", ".join(f'"{key}"' for key in parameters)
        values = ", ".join(f":{key}" for key in parameters)
        updates = ", ".join(
            f'"{key}" = {_FORWARD_ONLY.get(key, "EXCLUDED." + chr(34) + key + chr(34))}'
            for key in parameters
            if key != "id"
        )
        query = f"""
            INSERT INTO steps ({columns})
            VALUES ({values})
            ON CONFLICT (id) DO UPDATE
            SET {updates};
        """
        await self.execute_sql(query=query, parameters=parameters)

    @queue_until_user_message()
    async def create_element(self, element: Element):
        """Persist an element, reading local files from disk instead of HTTP."""
        from chainlit.element import ElementDict
        from chainlit.logger import logger

        if self.show_logger:
            logger.info(f"SQLAlchemy: create_element, element_id = {element.id}")

        if not self.storage_provider:
            logger.warning("SQLAlchemy: create_element error. No blob_storage_client is configured!")
            return
        if not element.for_id:
            return

        content: bytes | str | None = None

        if element.path:
            # --- CAS NORMAL : fichier temporaire uploadé par l'utilisateur ---
            async with aiofiles.open(element.path, "rb") as f:
                content = await f.read()

        elif element.url:
            # --- FIX : intercepter les URLs locales ---
            if element.url.startswith(LOCAL_FILES_ROUTE + "/"):
                # Extraire l'object_key depuis l'URL locale
                object_key = element.url[len(LOCAL_FILES_ROUTE) + 1 :]
                # Décoder les caractères URL-encodés (espaces, accents, etc.)
                object_key = unquote(object_key)
                disk_path = os.path.join(STORAGE_BASE_PATH, object_key)

                if os.path.exists(disk_path):
                    async with aiofiles.open(disk_path, "rb") as f:
                        content = await f.read()
                else:
                    logger.warning(f"Local file not found: {disk_path} (from URL: {element.url})")
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
            ext = ".md" if isinstance(content, str) else ".bin"  # texte (panneau de sources) ou binaire

        # 2. On génère un nom purement ASCII basé sur l'heure
        # Format : Screenshot_2026-02-09_16-15-30.png
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_filename = f"Screenshot_{timestamp}{ext}"

        file_object_key = f"{user_id}/{element.id}/{safe_filename}"

        if not element.mime:
            element.mime = "text/markdown" if isinstance(content, str) else "application/octet-stream"

        uploaded_file = await self.storage_provider.upload_file(
            object_key=file_object_key,
            data=content,
            mime=element.mime,
            overwrite=True,
        )
        if not uploaded_file:
            raise ValueError("SQLAlchemy Error: create_element, Failed to persist data in storage_provider")

        element_dict: ElementDict = element.to_dict()
        element_dict["url"] = uploaded_file.get("url")
        element_dict["objectKey"] = uploaded_file.get("object_key")

        element_dict_cleaned = {k: v for k, v in element_dict.items() if v is not None}
        if "props" in element_dict_cleaned:
            element_dict_cleaned["props"] = json.dumps(element_dict_cleaned["props"])

        columns = ", ".join(f'"{col}"' for col in element_dict_cleaned.keys())
        placeholders = ", ".join(f":{col}" for col in element_dict_cleaned.keys())
        updates = ", ".join(f'"{col}" = :{col}' for col in element_dict_cleaned.keys() if col != "id")
        query = f"INSERT INTO elements ({columns}) VALUES ({placeholders}) ON CONFLICT (id) DO UPDATE SET {updates};"
        await self.execute_sql(query=query, parameters=element_dict_cleaned)

    async def get_thread(self, thread_id: str):
        """Override to fix element URL resolution on chat resume.

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
                logger.info(f"get_thread: loaded {len(elements_results)} elements for thread {thread_id}")

        return thread

    async def get_element(self, thread_id: str, element_id: str):
        """Retrieve a single element with proper URL resolution.

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
