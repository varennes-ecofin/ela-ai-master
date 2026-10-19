"""Course outlines (chapter › section › slide titles) built from the store.

The outline is what ``get_course_outline`` shows the model and what
``get_slide`` uses to find a slide by title. It is computed from the slide
chunks of the levels visible to the user and cached per store generation.
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from langchain_core.documents import Document

from ela.retrieval import resources
from ela.retrieval.retriever import allowed_levels


@dataclass
class SectionOutline:
    """Section of a chapter with its slide titles."""

    title: str
    slides: list[str] = field(default_factory=list)


@dataclass
class ChapterOutline:
    """Chapter of a course."""

    title: str
    sections: list[SectionOutline] = field(default_factory=list)

    @property
    def n_slides(self) -> int:
        """Number of slides in the chapter."""
        return sum(len(s.slides) for s in self.sections)


@dataclass
class CourseOutline:
    """Outline of one course (folder) with its slide documents."""

    course: str
    level: str
    sources: list[str] = field(default_factory=list)
    chapters: list[ChapterOutline] = field(default_factory=list)
    slides: list[Document] = field(default_factory=list)

    @property
    def n_slides(self) -> int:
        """Number of slides in the course."""
        return len(self.slides)

    @property
    def aliases(self) -> set[str]:
        """Lower-case names the course may be called by (folder, file stems, acronym)."""
        names = {self.course.lower(), self.course.replace("_", " ").lower()}
        for src in self.sources:
            stem = re.sub(r"\.tex$", "", src, flags=re.IGNORECASE)
            names.add(stem.lower())
            names.add(re.split(r"[_\-]", stem)[0].lower())  # HFE_C4 → hfe
        acronym = "".join(w[0] for w in re.split(r"[_\s]+", self.course) if w)
        if len(acronym) >= 2:
            names.add(acronym.lower())
        return names


def _sort_key(doc: Document) -> tuple:
    meta = doc.metadata or {}
    return (str(meta.get("source", "")), int(meta.get("ordinal", 0) or 0), int(meta.get("part", 0) or 0))


def build_outlines(docs: list[Document]) -> dict[str, CourseOutline]:
    """Group slide documents into per-course outlines.

    Args:
        docs: Documents of the store (any chunk type; only slides are kept).

    Returns:
        ``course → CourseOutline`` in alphabetical order of course.
    """
    slides = [d for d in docs if (d.metadata or {}).get("chunk_type", "slide") == "slide"]
    slides.sort(key=_sort_key)
    outlines: OrderedDict[str, CourseOutline] = OrderedDict()
    for doc in slides:
        meta = doc.metadata or {}
        course = str(meta.get("course") or meta.get("source") or "?")
        outline = outlines.get(course)
        if outline is None:
            outline = outlines[course] = CourseOutline(course=course, level=str(meta.get("level") or ""))
        src = str(meta.get("source") or "")
        if src and src not in outline.sources:
            outline.sources.append(src)
        outline.slides.append(doc)
        chapter_title = str(meta.get("chapter") or "").strip() or "(sans chapitre)"
        section_title = str(meta.get("section") or "").strip() or ""
        if not outline.chapters or outline.chapters[-1].title != chapter_title:
            outline.chapters.append(ChapterOutline(title=chapter_title))
        chapter = outline.chapters[-1]
        if not chapter.sections or chapter.sections[-1].title != section_title:
            chapter.sections.append(SectionOutline(title=section_title))
        title = str(meta.get("slide_title") or "").strip()
        if title and (not chapter.sections[-1].slides or chapter.sections[-1].slides[-1] != title):
            chapter.sections[-1].slides.append(title)
    return dict(sorted(outlines.items()))


_cache: dict[tuple, dict[str, CourseOutline]] = {}
_lock = threading.Lock()


def get_outlines(user_level: str) -> dict[str, CourseOutline]:
    """Outlines of the courses visible to a level (cached per store generation).

    Args:
        user_level: Academic level (``M1``, ``M2``, ``ALL``...).

    Returns:
        ``course → CourseOutline``.
    """
    levels = allowed_levels(user_level)
    key = (resources.store_fingerprint(), tuple(levels) if levels else None)
    if key not in _cache:
        with _lock:
            if key not in _cache:
                for old in [k for k in _cache if k[0] != key[0]]:
                    _cache.pop(old, None)
                _cache[key] = build_outlines(resources.get_store().all_documents(levels))
    return _cache[key]


def reset_cache() -> None:
    """Forget cached outlines (tests)."""
    with _lock:
        _cache.clear()


def courses_elsewhere(topic: str, user_level: str) -> list[tuple[str, str]]:
    """Courses hidden from ``user_level`` whose slides cover ``topic``.

    Used to tell a student that a notion is taught in another year's course
    rather than « the course does not cover it ».

    Args:
        topic: Requested topic.
        user_level: The student's level.

    Returns:
        ``(course, level)`` pairs, or an empty list (also when the store is unreachable).
    """
    from ela.pedagogy.relevance import covers

    try:
        visible = set(get_outlines(user_level))
        everything = get_outlines("ALL")
    except Exception:  # store unreachable
        return []
    found: list[tuple[str, str]] = []
    for course, outline in everything.items():
        if course in visible:
            continue
        texts = [f"{(d.metadata or {}).get('slide_title', '')} {d.page_content}" for d in outline.slides]
        if covers(topic, texts, all_terms=True):
            found.append((course, outline.level))
    return found


def resolve_course(name: str | None, outlines: dict[str, CourseOutline]) -> str | None:
    """Map a course name given by the model to a course folder.

    Accepts the folder name, a master file name (``UTS.tex``, ``HFE_C4``),
    the acronym (``mts``) or a case-insensitive substring.

    Args:
        name: Name given by the model (may be ``None``).
        outlines: Visible outlines.

    Returns:
        The course folder, or ``None`` when nothing matches.
    """
    if not name:
        return None
    needle = name.strip().lower()
    if not needle:
        return None
    for course, outline in outlines.items():
        if needle in outline.aliases:
            return course
    for course in outlines:
        if needle in course.lower() or needle.replace(" ", "_") in course.lower():
            return course
    for course, outline in outlines.items():
        if any(needle in alias or alias in needle for alias in outline.aliases if len(alias) >= 3):
            return course
    return None


def render_outline(outline: CourseOutline, *, max_chars: int = 6000) -> str:
    """Render an outline as an indented text list within a character budget.

    Args:
        outline: The course outline.
        max_chars: Budget; slide titles are dropped first, then sections.

    Returns:
        Text for the model.
    """
    header = (
        f"Cours {outline.course} (niveau {outline.level}, {outline.n_slides} slides, "
        f"fichiers : {', '.join(outline.sources)})"
    )
    for detail in ("slides", "sections", "chapters"):
        lines = [header]
        for ch in outline.chapters:
            lines.append(f"- {ch.title} ({ch.n_slides} slides)")
            if detail == "chapters":
                continue
            for sec in ch.sections:
                if sec.title:
                    lines.append(f"  - {sec.title}")
                if detail == "slides":
                    lines.extend(f"    - {t}" for t in sec.slides)
        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text
    return text[:max_chars] + "\n…(plan tronqué)"


def render_course_list(outlines: dict[str, CourseOutline]) -> str:
    """One line per visible course with its chapter titles."""
    if not outlines:
        return "Aucun cours n'est disponible pour ce niveau."
    lines = ["Cours disponibles :"]
    for course, outline in outlines.items():
        chapters = "; ".join(ch.title for ch in outline.chapters[:12])
        lines.append(f"- {course} (niveau {outline.level}, {outline.n_slides} slides) — chapitres : {chapters}")
    return "\n".join(lines)


def find_slides(outlines: dict[str, CourseOutline], title: str, *, course: str | None = None) -> list[Document]:
    """Find slides whose title contains ``title`` (case-insensitive).

    Args:
        outlines: Visible outlines.
        title: Title fragment.
        course: Restrict to one course folder.

    Returns:
        Matching slide documents, exact title matches first.
    """
    needle = title.strip().lower()
    if not needle:
        return []
    pool = [outlines[course]] if course and course in outlines else list(outlines.values())
    exact, partial = [], []
    for outline in pool:
        for doc in outline.slides:
            t = str((doc.metadata or {}).get("slide_title") or "").lower()
            if t == needle:
                exact.append(doc)
            elif needle in t:
                partial.append(doc)
    return exact + partial
