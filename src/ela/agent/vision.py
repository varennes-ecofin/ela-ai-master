"""Image transcription by the vision model.

The text agent does not see images. When the student attaches one, the
multimodal model transcribes it faithfully (equations in LaTeX, tables,
software output, graph description) and the transcription is appended to
the student's message. The agent then works with its usual tools; nothing
depends on the vision model supporting tool calls.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ela.agent.prompts import render
from ela.llm import encode_image_block

logger = logging.getLogger(__name__)

ImageInput = tuple[str, str | None]

TRANSCRIPTION_HEADER = "--- Contenu de l'image jointe (transcription automatique) ---"


async def transcribe_images(vision_llm: BaseChatModel, images: Sequence[ImageInput], question: str) -> str:
    """Transcribe the attached images for the text agent.

    Args:
        vision_llm: Multimodal model.
        images: ``(path, mime)`` tuples.
        question: Student question (context only).

    Returns:
        The transcription text (may be empty when nothing is readable).
    """
    content: list = [{"type": "text", "text": "Transcris fidèlement le contenu de cette ou ces image(s)."}]
    for path, mime in images:
        content.append(encode_image_block(path, mime))
    messages = [SystemMessage(content=render("transcribe", question=question)), HumanMessage(content=content)]
    response = await vision_llm.ainvoke(messages)
    text = response.content if isinstance(response.content, str) else _text_blocks(response.content)
    return text.strip()


def _text_blocks(content) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def with_transcription(question: str, transcription: str) -> str:
    """Append the transcription to the student's message."""
    if not transcription:
        return question
    return f"{question}\n\n{TRANSCRIPTION_HEADER}\n{transcription}"
