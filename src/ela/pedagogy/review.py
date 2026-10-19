"""Review sheets: one chapter → one structured, one-page summary.

The sheet is generated from the chapter's own slides (taken from the course
outline, not from a similarity search, so it cannot drift to another topic).
Long chapters are summarised part by part and the parts are merged.
"""

from __future__ import annotations

import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ela.agent.outline import ChapterOutline, CourseOutline
from ela.agent.prompts import render
from ela.pedagogy.quiz import GENERATION_MAX_TOKENS
from ela.pedagogy.relevance import covers, normalize
from ela.pedagogy.schemas import ReviewSheet, ReviewSheetPayload
from ela.structured import ainvoke_structured

logger = logging.getLogger(__name__)

PART_MAX_CHARS = 24_000  # slides text per model call (≈ 8k tokens)
_CONTEXT_HEADER = re.compile(r"\A(?:[A-ZÉ_ ]{4,}:[^\n]*\n)+CONTENU:\s*\n", re.MULTILINE)
_LEADING_NUMBER = re.compile(r"^\s*(?:chap(?:itre)?\.?\s*)?(\d+)\b", re.IGNORECASE)


def resolve_chapter(outline: CourseOutline, name: str) -> ChapterOutline | None:
    """Find a chapter of ``outline`` from a title, a number (« 2 », « chapitre 2 ») or a fragment.

    Args:
        outline: Course outline.
        name: Chapter as given by the student or the model.

    Returns:
        The chapter, or ``None``.
    """
    needle = normalize(name).strip()
    if not needle or not outline.chapters:
        return None
    for ch in outline.chapters:
        if normalize(ch.title).strip() == needle:
            return ch
    m = _LEADING_NUMBER.match(name)
    if m and (needle.isdigit() or needle.startswith("chap")):
        number = int(m.group(1))
        for ch in outline.chapters:  # a chapter numbered in its own title wins
            cm = _LEADING_NUMBER.match(ch.title)
            if cm and int(cm.group(1)) == number:
                return ch
        if 1 <= number <= len(outline.chapters):
            return outline.chapters[number - 1]
    for ch in outline.chapters:
        if needle in normalize(ch.title):
            return ch
    hits = [ch for ch in outline.chapters if covers(name, [ch.title], all_terms=True)]
    return hits[0] if len(hits) == 1 else None


def chapter_slides(outline: CourseOutline, chapter: ChapterOutline) -> list[tuple[str, str]]:
    """``(slide title, text)`` of the slides of ``chapter``, in course order."""
    out: list[tuple[str, str]] = []
    for doc in outline.slides:
        meta = doc.metadata or {}
        if (str(meta.get("chapter") or "").strip() or "(sans chapitre)") != chapter.title:
            continue
        title = str(meta.get("slide_title") or "").strip() or "(sans titre)"
        out.append((title, _CONTEXT_HEADER.sub("", doc.page_content, count=1).strip()))
    return out


def format_slides(slides: list[tuple[str, str]]) -> str:
    """Render slides for the prompt."""
    return "\n\n".join(f"— {title} —\n{text}" for title, text in slides)


def split_parts(slides: list[tuple[str, str]], max_chars: int | None = None) -> list[list[tuple[str, str]]]:
    """Group slides into consecutive parts under ``max_chars`` (default :data:`PART_MAX_CHARS`) characters each."""
    max_chars = PART_MAX_CHARS if max_chars is None else max_chars
    parts: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    size = 0
    for slide in slides:
        n = len(slide[0]) + len(slide[1]) + 8
        if current and size + n > max_chars:
            parts.append(current)
            current, size = [], 0
        current.append(slide)
        size += n
    if current:
        parts.append(current)
    return parts


def merge_payloads(parts: list[ReviewSheetPayload]) -> ReviewSheetPayload:
    """Merge the payloads of the parts of a long chapter into one sheet."""
    if len(parts) == 1:
        return parts[0]
    objectives: list[str] = []
    takeaways: list[str] = []
    sections = []
    for p in parts:
        objectives.extend(o for o in p.objectives if o not in objectives)
        takeaways.extend(t for t in p.takeaways if t not in takeaways)
        sections.extend(p.sections)
    return ReviewSheetPayload(
        title=parts[0].title, objectives=objectives[:6], sections=sections, takeaways=takeaways[:8]
    )


async def generate_review_sheet(
    llm: BaseChatModel, outline: CourseOutline, chapter: ChapterOutline, *, level: str = "ALL"
) -> ReviewSheet:
    """Generate the review sheet of one chapter.

    Args:
        llm: Text model.
        outline: Course outline (gives the slides).
        chapter: Chapter to summarise.
        level: Student level, for the prompt.

    Returns:
        The sheet.

    Raises:
        ValueError: When the chapter has no slide text, or the model produced nothing usable.
    """
    slides = chapter_slides(outline, chapter)
    if not slides:
        raise ValueError(f"chapter « {chapter.title} » has no slide")
    parts = split_parts(slides)
    payloads: list[ReviewSheetPayload] = []
    for i, part in enumerate(parts, start=1):
        note = f" (partie {i}/{len(parts)} du chapitre)" if len(parts) > 1 else ""
        system = render(
            "review_sheet",
            level=level,
            course=outline.course,
            chapter=chapter.title,
            part_note=note,
            context=format_slides(part),
        )
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Rédige la fiche de révision du chapitre « {chapter.title} »{note}."),
        ]
        payloads.append(await ainvoke_structured(llm, ReviewSheetPayload, messages, max_tokens=GENERATION_MAX_TOKENS))
    merged = merge_payloads(payloads)
    return ReviewSheet(
        **merged.model_dump(),
        course=outline.course,
        chapter=chapter.title,
        level=outline.level or level,
        slides=[title for title, _ in slides],
    )


class ReviewLookupError(LookupError):
    """The course or the chapter asked for does not exist at this level (message for the student)."""


def resolve_target(level: str, course: str, chapter: str) -> tuple[CourseOutline, ChapterOutline]:
    """Resolve a course and a chapter among the outlines visible at ``level``.

    Raises:
        ReviewLookupError: With a French message listing what exists.
    """
    from ela.agent.outline import get_outlines, resolve_course

    outlines = get_outlines(level)
    if not outlines:
        raise ReviewLookupError("Aucun cours n'est disponible pour ce niveau.")
    name = resolve_course(course, outlines)
    if name is None:
        raise ReviewLookupError(
            f"Cours « {course} » inconnu pour ce niveau. Cours disponibles : {', '.join(outlines)}."
        )
    outline = outlines[name]
    target = resolve_chapter(outline, chapter)
    if target is None:
        chapters = "\n".join(f"- {ch.title}" for ch in outline.chapters)
        raise ReviewLookupError(f"Chapitre « {chapter} » inconnu dans {name}. Chapitres :\n{chapters}")
    return outline, target


async def review_sheet_for(llm: BaseChatModel, level: str, course: str, chapter: str) -> ReviewSheet:
    """Return the sheet of a chapter, from the cache when the corpus and the prompt have not changed.

    Args:
        llm: Text model (used only on a cache miss).
        level: Student level (scopes the visible courses).
        course: Course name, acronym or file name.
        chapter: Chapter title, number or fragment.

    Returns:
        The sheet.

    Raises:
        ReviewLookupError: Unknown course or chapter.
        ValueError: Generation failure.
    """
    from ela.agent.prompts import prompt_version
    from ela.db.review_sheets import get_cached_sheet, store_sheet
    from ela.retrieval import resources

    outline, target = resolve_target(level, course, chapter)
    key = {"prompt_version": prompt_version(), "store_fingerprint": resources.store_fingerprint()}
    cached = await get_cached_sheet(outline.course, target.title, outline.level, **key)
    if cached is not None:
        logger.info("Review sheet %s › %s served from cache", outline.course, target.title)
        return cached
    sheet = await generate_review_sheet(llm, outline, target, level=level)
    await store_sheet(sheet, **key)
    return sheet
