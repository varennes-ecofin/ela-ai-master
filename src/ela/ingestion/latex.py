r"""LaTeX corpus parsing for ingestion (Phase 2 parser).

Scans ``latex/{level}/{course}/*.tex``, resolves ``\\include``/``\\input``
recursively (cycle-safe), walks the document structure (parts, chapters,
sections, beamer frames or manuscript paragraphs) and produces typed
:class:`Chunk` objects:

* ``slide`` — one per beamer frame (title + readable body + speaker notes);
* ``definition``, ``theorem``, ``proposition``, ``lemma``, ``corollary``,
  ``example``, ``remark``, ``model``, ``assumption`` — one per theorem-like
  environment, linked to its parent slide/section by ``parent_id``;
* ``text`` — paragraphs of a non-beamer document (manuscript, article),
  grouped to about ``target_chars`` characters.

Each chunk carries ``doc_kind`` (``slides`` or ``manuscript``) so that the
retriever can rank course slides first and use manuscripts as a complement.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from langchain_core.documents import Document

from ela.ingestion import texclean

VALID_LEVELS = {"M1", "M2", "Commun"}

#: Metadata keys stored with every chunk (empty string when not applicable).
METADATA_KEYS = (
    "chunk_id",
    "parent_id",
    "content_hash",
    "source",
    "level",
    "course",
    "doc_kind",
    "chunk_type",
    "chapter",
    "section",
    "subsection",
    "slide_title",
    "element_title",
    "ordinal",
    "part",
    "n_chars",
)

# --- INCLUDE/INPUT RESOLVER ---

_INCLUDE_PATTERN = re.compile(r"^[ \t]*\\(?:include|input)\{([^}]+)\}", re.MULTILINE)


def resolve_includes(
    content: str,
    base_dir: str,
    visited: set[str] | None = None,
    depth: int = 0,
    max_depth: int = 10,
    log: list[str] | None = None,
) -> str:
    r"""Recursively resolve ``\include{}`` and ``\input{}`` directives.

    Args:
        content: Raw LaTeX string.
        base_dir: Directory of the current .tex file (for relative paths).
        visited: Absolute paths already resolved (cycle detection).
        depth: Current recursion depth.
        max_depth: Maximum recursion depth.
        log: Optional list receiving human-readable messages.

    Returns:
        LaTeX string with all includes resolved inline.
    """
    if visited is None:
        visited = set()
    log = log if log is not None else []
    if depth > max_depth:
        log.append(f"profondeur max ({max_depth}) atteinte")
        return content

    def _replace(match: re.Match) -> str:
        line_start = content.rfind("\n", 0, match.start()) + 1
        if content[line_start : match.start()].lstrip().startswith("%"):
            return match.group(0)
        ref_name = match.group(1).strip()
        candidates = [ref_name] if ref_name.endswith(".tex") else [ref_name, ref_name + ".tex"]
        for candidate in candidates:
            ref_path = os.path.normpath(os.path.join(base_dir, candidate))
            if ref_path in visited:
                log.append(f"référence circulaire : {candidate}")
                return ""
            if os.path.isfile(ref_path):
                visited.add(ref_path)
                log.append(f"include résolu : {candidate}")
                with open(ref_path, encoding="utf-8", errors="replace") as f:
                    included = f.read()
                return resolve_includes(included, os.path.dirname(ref_path), visited, depth + 1, max_depth, log)
        log.append(f"fichier inclus introuvable : {ref_name}")
        return ""

    return _INCLUDE_PATTERN.sub(_replace, content)


# --- CHUNKS ---


@dataclass
class Chunk:
    """A retrievable unit of course content.

    Attributes:
        text: Readable body (math preserved as ``$...$`` / ``$$...$$``).
        source: Master .tex file name.
        level: Academic level (``M1``, ``M2``, ``Commun``, ``legacy``).
        course: Course folder name.
        doc_kind: ``slides`` (beamer) or ``manuscript`` (article/book).
        chunk_type: ``slide``, ``text``, or a theorem-like type.
        chapter: Chapter title (empty when none).
        section: Section title (empty when none).
        subsection: Subsection title (empty when none).
        slide_title: Frame title (empty for manuscript chunks).
        element_title: Title of a theorem-like element (empty otherwise).
        ordinal: Position of the parent unit in the document.
        part: Part index when a long unit was split (0 otherwise).
        notes: Speaker notes attached to the slide.
        chunk_id: Stable content-based identifier (set by :meth:`finalize`).
        parent_id: ``chunk_id`` of the parent slide/text chunk (elements only).
        content_hash: SHA-256 of the indexed text and locating metadata.
    """

    text: str
    source: str
    level: str
    course: str
    doc_kind: str
    chunk_type: str
    chapter: str = ""
    section: str = ""
    subsection: str = ""
    slide_title: str = ""
    element_title: str = ""
    ordinal: int = 0
    part: int = 0
    notes: list[str] = field(default_factory=list)
    chunk_id: str = ""
    parent_id: str = ""
    content_hash: str = ""

    # ---- rendering ----

    @property
    def location(self) -> str:
        """Human-readable location (``Chapitre > Section > Slide``)."""
        bits = [b for b in (self.chapter, self.section, self.subsection) if b]
        if self.slide_title:
            bits.append(self.slide_title)
        return " > ".join(bits)

    def page_content(self) -> str:
        """Text indexed and shown to the LLM (short context header + body)."""
        header = [f"COURS: {self.course} ({self.level})"]
        if self.chapter:
            header.append(f"CHAPITRE: {self.chapter}")
        if self.section:
            header.append(f"SECTION: {self.section}" + (f" / {self.subsection}" if self.subsection else ""))
        if self.slide_title:
            header.append(f"SLIDE: {self.slide_title}")
        if self.element_title and self.chunk_type not in {"slide", "text"}:
            header.append(f"{self.chunk_type.upper()}: {self.element_title}")
        body = self.text
        if self.notes:
            body += "\n\nNOTES DE L'ENSEIGNANT:\n" + "\n".join(self.notes)
        return "\n".join(header) + "\nCONTENU:\n" + body

    def finalize(self) -> Chunk:
        """Compute ``content_hash`` and ``chunk_id`` (idempotent)."""
        key = "|".join(
            [self.level, self.course, self.source, self.doc_kind, self.chunk_type, self.location, str(self.part)]
        )
        digest = hashlib.sha256((key + "\n" + self.page_content()).encode("utf-8")).hexdigest()
        self.content_hash = digest
        self.chunk_id = digest[:24]
        return self

    def metadata(self) -> dict:
        """Flat metadata dict (strings/ints only, Chroma- and SQL-friendly)."""
        data = asdict(self)
        meta = {k: data[k] for k in METADATA_KEYS if k != "n_chars"}
        meta["n_chars"] = len(self.text)
        # Backward compatibility with the Phase 1 metadata (``type``)
        meta["type"] = (
            "slide" if self.chunk_type == "slide" else ("text_block" if self.chunk_type == "text" else "element")
        )
        return meta

    def to_document(self) -> Document:
        """LangChain ``Document`` view of the chunk."""
        return Document(page_content=self.page_content(), metadata=self.metadata())


# --- STRUCTURE WALKER ---

_HEADING_RE = re.compile(
    r"\\(part|chapter|section|subsection|subsubsection)\*?\s*(\[[^\]]*\])?\s*(?=\{)"
    r"|\\begin\{chapter\}"
    r"|\\begin\{frame\}"
)
_FRAMETITLE_RE = re.compile(r"\\frametitle\s*(\[[^\]]*\])?\s*(?=\{)")
_TITLEPAGE_FRAME = {"", "références", "references", "plan", "sommaire", "outline", "table des matières"}


@dataclass
class _Heading:
    chapter: str = ""
    section: str = ""
    subsection: str = ""


class LatexParser:
    r"""Parse one resolved LaTeX document into chunks.

    Args:
        source: Master file name (used as ``source`` metadata).
        level: Academic level.
        course: Course name.
        target_chars: Target size of manuscript text chunks.
        max_chars: Hard limit above which slides/text are split into parts.
        include_notes: Attach ``\\note{}`` speaker notes to slides.
    """

    def __init__(
        self,
        source: str,
        level: str,
        course: str,
        *,
        target_chars: int = 1200,
        max_chars: int = 2500,
        include_notes: bool = True,
    ) -> None:
        self.source = source
        self.level = level
        self.course = course
        self.target_chars = target_chars
        self.max_chars = max_chars
        self.include_notes = include_notes
        self.warnings: list[str] = []

    # ---- entry point ----

    def parse(self, content: str) -> list[Chunk]:
        """Parse resolved LaTeX content.

        Args:
            content: Full LaTeX source (includes resolved).

        Returns:
            Finalised chunks in document order.
        """
        content = texclean.strip_comments(content)
        self.macros = texclean.collect_macros(content)
        body = self._document_body(content)
        self.doc_kind = "slides" if "\\begin{frame}" in body else "manuscript"

        chunks: list[Chunk] = []
        heading = _Heading()
        ordinal = 0
        pos = 0
        pending_text: list[str] = []  # manuscript text between headings

        def flush_text() -> None:
            nonlocal ordinal
            raw = "\n".join(pending_text)
            pending_text.clear()
            if raw.strip():
                ordinal += 1
                chunks.extend(self._text_chunks(raw, heading, ordinal))

        for m in _HEADING_RE.finditer(body):
            if m.start() < pos:
                continue
            if self.doc_kind == "manuscript":
                pending_text.append(body[pos : m.start()])
            token = m.group(0)
            if token == "\\begin{frame}":
                end = texclean.find_env_end(body, "frame", m.end())
                if end is None:
                    self.warnings.append("frame non fermée")
                    pos = m.end()
                    continue
                ordinal += 1
                chunks.extend(self._frame_chunks(body[m.end() : end[0]], heading, ordinal))
                pos = end[1]
                continue
            if token == "\\begin{chapter}":
                flush_text()
                end = texclean.find_env_end(body, "chapter", m.end())
                if end is None:
                    pos = m.end()
                    continue
                heading = _Heading(chapter=self._custom_chapter_title(body[m.end() : end[0]]))
                pos = end[1]
                continue
            kind = m.group(1)
            grp = texclean.read_group(body, m.end())
            if grp is None:
                pos = m.end()
                continue
            flush_text()
            title = texclean.plain_title(grp[0], self.macros)
            if kind in {"part", "chapter"}:
                heading = _Heading(chapter=title)
            elif kind == "section":
                heading = _Heading(chapter=heading.chapter, section=title)
            elif kind == "subsection":
                heading = _Heading(chapter=heading.chapter, section=heading.section, subsection=title)
            else:  # subsubsection: keep the parent subsection
                pass
            pos = grp[1]
        if self.doc_kind == "manuscript":
            pending_text.append(body[pos:])
            flush_text()
        return [c.finalize() for c in chunks]

    # ---- helpers ----

    @staticmethod
    def _document_body(content: str) -> str:
        m = re.search(r"\\begin\{document\}", content)
        if not m:
            return content
        end = content.rfind("\\end{document}")
        return content[m.end() : end if end > m.end() else None]

    def _custom_chapter_title(self, inner: str) -> str:
        r"""Title of the beamer ``chapter`` environment used by the course theme.

        ``\\begin{chapter}[img]{color}{Chapter 1} \\textit{Titre} \\end{chapter}``
        becomes ``Chapter 1 — Titre``.
        """
        _, p = texclean.read_optional(inner, 0)
        args: list[str] = []
        while True:
            q = texclean.skip_spaces(inner, p)
            grp = texclean.read_group(inner, q)
            if grp is None:
                break
            args.append(grp[0])
            p = grp[1]
        number = texclean.plain_title(args[-1], self.macros) if args else ""
        rest = texclean.plain_title(inner[p:], self.macros).strip("* ")
        return " — ".join(b for b in (number, rest) if b)

    def _frame_chunks(self, inner: str, heading: _Heading, ordinal: int) -> list[Chunk]:
        """Chunks for one beamer frame: the slide and its typed elements."""
        _, p = texclean.read_optional(inner, 0)
        inner = inner[p:]
        title = ""
        q = texclean.skip_spaces(inner, 0)
        grp = texclean.read_group(inner, q)
        if grp is not None:
            title = grp[0]
            inner = inner[grp[1] :]
            # optional subtitle argument
            q = texclean.skip_spaces(inner, 0)
            grp2 = texclean.read_group(inner, q)
            if grp2 is not None and "\n" not in inner[:q]:
                inner = inner[grp2[1] :]
        else:
            m = _FRAMETITLE_RE.search(inner)
            if m:
                grp = texclean.read_group(inner, m.end())
                if grp is not None:
                    title = grp[0]
                    inner = inner[: m.start()] + inner[grp[1] :]
        title = texclean.plain_title(title, self.macros)
        conv = texclean.convert(inner, self.macros, with_notes=self.include_notes)
        body = conv.text
        if title.lower().strip() in _TITLEPAGE_FRAME and len(body) < 80:
            return []
        if len(re.sub(r"\[(Figure|Tableau)[^\]]*\]", "", body).strip()) < 15 and not conv.notes:
            return []  # figure-only or empty slide: nothing retrievable

        chunks: list[Chunk] = []
        base = dict(
            source=self.source,
            level=self.level,
            course=self.course,
            doc_kind=self.doc_kind,
            chapter=heading.chapter,
            section=heading.section,
            subsection=heading.subsection,
            slide_title=title,
            ordinal=ordinal,
        )
        parts = self._split_long(body)
        for idx, part in enumerate(parts):
            chunks.append(
                Chunk(
                    text=part,
                    chunk_type="slide",
                    part=idx if len(parts) > 1 else 0,
                    notes=conv.notes if idx == len(parts) - 1 and self.include_notes else [],
                    **base,
                )
            )
        parent = chunks[0].finalize()
        chunks.extend(self._element_chunks(conv.elements, base, parent.chunk_id, body))
        return chunks

    def _text_chunks(self, raw: str, heading: _Heading, ordinal: int) -> list[Chunk]:
        """Chunks for manuscript text between two headings."""
        conv = texclean.convert(raw, self.macros, with_notes=False)
        if len(conv.text) < 40:
            return []
        base = dict(
            source=self.source,
            level=self.level,
            course=self.course,
            doc_kind=self.doc_kind,
            chapter=heading.chapter,
            section=heading.section,
            subsection=heading.subsection,
            ordinal=ordinal,
        )
        chunks = [
            Chunk(text=part, chunk_type="text", part=idx, **base)
            for idx, part in enumerate(self._group_paragraphs(conv.text))
        ]
        if not chunks:
            return []
        parent = chunks[0].finalize()
        chunks.extend(self._element_chunks(conv.elements, base, parent.chunk_id, conv.text))
        return chunks

    def _element_chunks(self, elements: list[texclean.Element], base: dict, parent_id: str, body: str) -> list[Chunk]:
        """Typed element chunks (definitions, theorems...) attached to a parent."""
        out: list[Chunk] = []
        for el in elements:
            if el.kind == "proof" or len(el.text) < 20:
                continue
            # A unit made of one single element is already the element: skip duplication.
            if len(body) - len(el.text) < 60:
                continue
            label = el.label + (f" ({el.title})" if el.title else "")
            out.append(
                Chunk(
                    text=f"**{label}** : {el.text}",
                    chunk_type=el.kind,
                    element_title=el.title,
                    parent_id=parent_id,
                    **base,
                )
            )
        return out

    def _group_paragraphs(self, text: str) -> list[str]:
        """Group paragraphs up to ``target_chars`` (never splitting a formula)."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        groups: list[str] = []
        current: list[str] = []
        size = 0
        for p in paragraphs:
            if current and size + len(p) > self.target_chars:
                groups.append("\n\n".join(current))
                current, size = [], 0
            current.append(p)
            size += len(p) + 2
        if current:
            groups.append("\n\n".join(current))
        out: list[str] = []
        for g in groups:
            out.extend(self._split_long(g))
        return out

    def _split_long(self, text: str) -> list[str]:
        """Split a unit longer than ``max_chars`` on paragraph/line boundaries."""
        if len(text) <= self.max_chars:
            return [text]
        pieces = re.split(r"(\n\s*\n|\n)", text)
        parts: list[str] = []
        current = ""
        for piece in pieces:
            if len(current) + len(piece) > self.max_chars and current.strip():
                parts.append(current.strip())
                current = ""
            current += piece
        if current.strip():
            parts.append(current.strip())
        return parts


# --- CORPUS SCANNER ---


@dataclass
class ParseReport:
    """Outcome of a corpus scan."""

    chunks: list[Chunk]
    stats: dict[str, int]
    files: list[str]
    warnings: list[str]

    def by_type(self) -> dict[str, int]:
        """Return the number of chunks per ``chunk_type``."""
        return dict(Counter(c.chunk_type for c in self.chunks))


class LatexIngestor:
    r"""Scan ``latex/{level}/{course}/`` and parse every master .tex file.

    Master files using ``\\include{chapter}`` are fully resolved; chapter
    files pulled in by a master are skipped to avoid double ingestion.

    Args:
        source_dir: Root of the LaTeX tree.
        target_chars: Target size of manuscript text chunks.
        max_chars: Hard limit for one chunk.
        include_notes: Attach speaker notes to slides.
        log: Callable receiving progress messages (``print`` by default).
    """

    def __init__(
        self,
        source_dir: str,
        *,
        target_chars: int = 1200,
        max_chars: int = 2500,
        include_notes: bool = True,
        log=print,
    ):
        self.source_dir = source_dir
        self.target_chars = target_chars
        self.max_chars = max_chars
        self.include_notes = include_notes
        self._log = log

    # ---- file-level ----

    def parse_file(self, file_path: str, level: str, course: str) -> list[Chunk]:
        """Parse one master .tex file (includes resolved) into chunks."""
        filename = os.path.basename(file_path)
        self._log(f"   📖 [{level}/{course}] {filename}")
        with open(file_path, encoding="utf-8", errors="replace") as f:
            raw = f.read()
        messages: list[str] = []
        content = resolve_includes(
            raw, os.path.dirname(os.path.abspath(file_path)), {os.path.abspath(file_path)}, log=messages
        )
        for msg in messages:
            self._log(f"      📎 {msg}")
        parser = LatexParser(
            filename,
            level,
            course,
            target_chars=self.target_chars,
            max_chars=self.max_chars,
            include_notes=self.include_notes,
        )
        chunks = parser.parse(content)
        for w in parser.warnings:
            self._log(f"      ⚠️  {filename}: {w}")
        return chunks

    # ---- tree-level ----

    def scan(self) -> ParseReport:
        """Parse the whole tree.

        Returns:
            A :class:`ParseReport` with chunks, per-course counts and warnings.
        """
        chunks: list[Chunk] = []
        stats: Counter = Counter()
        files: list[str] = []
        warnings: list[str] = []

        if not os.path.isdir(self.source_dir):
            warnings.append(f"dossier source introuvable : {self.source_dir}")
            return ParseReport(chunks, dict(stats), files, warnings)

        self._log(f"📂 Scan récursif de {self.source_dir}/ ...")
        masters = self._find_master_files()

        for level_name in sorted(os.listdir(self.source_dir)):
            level_path = os.path.join(self.source_dir, level_name)
            if not os.path.isdir(level_path):
                if level_name.endswith(".tex"):
                    warnings.append(f"fichier à la racine (legacy) : {level_name}")
                    docs = self.parse_file(level_path, "legacy", "legacy")
                    chunks.extend(docs)
                    stats["legacy/legacy"] += len(docs)
                    files.append(level_path)
                continue
            if level_name not in VALID_LEVELS:
                warnings.append(f"dossier ignoré (niveau inconnu) : {level_name}/")
                continue
            for course_name in sorted(os.listdir(level_path)):
                course_path = os.path.join(level_path, course_name)
                if not os.path.isdir(course_path):
                    if course_name.endswith(".tex"):
                        docs = self.parse_file(course_path, level_name, "Général")
                        chunks.extend(docs)
                        stats[f"{level_name}/Général"] += len(docs)
                        files.append(course_path)
                    continue
                tex_files = sorted(f for f in os.listdir(course_path) if f.endswith(".tex"))
                if not tex_files:
                    warnings.append(f"aucun .tex dans {level_name}/{course_name}/")
                    continue
                local_masters = masters.get(os.path.abspath(course_path), set())
                included: set[str] = set()
                for master_path in local_masters:
                    included.update(self._get_included_basenames(master_path))
                for tex_name in tex_files:
                    tex_path = os.path.join(course_path, tex_name)
                    if os.path.splitext(tex_name)[0] in included and os.path.abspath(tex_path) not in local_masters:
                        self._log(f"      ⏭️  Skip (inclus par un master) : {tex_name}")
                        continue
                    docs = self.parse_file(tex_path, level_name, course_name)
                    chunks.extend(docs)
                    stats[f"{level_name}/{course_name}"] += len(docs)
                    files.append(tex_path)
        return ParseReport(chunks, dict(stats), files, warnings)

    def load_documents(self) -> tuple[list[Document], dict[str, int]]:
        """Phase 1-compatible entry point: ``(documents, stats)``."""
        report = self.scan()
        for w in report.warnings:
            self._log(f"   ⚠️  {w}")
        return [c.to_document() for c in report.chunks], report.stats

    # ---- helpers ----

    def _find_master_files(self) -> dict[str, set[str]]:
        masters: dict[str, set[str]] = {}
        for root, _dirs, files in os.walk(self.source_dir):
            for fname in files:
                if not fname.endswith(".tex"):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, encoding="utf-8", errors="replace") as f:
                        text = texclean.strip_comments(f.read())
                except OSError:
                    continue
                if _INCLUDE_PATTERN.search(text):
                    masters.setdefault(os.path.abspath(root), set()).add(os.path.abspath(fpath))
        return masters

    def _get_included_basenames(self, master_path: str) -> set[str]:
        names: set[str] = set()
        try:
            with open(master_path, encoding="utf-8", errors="replace") as f:
                text = texclean.strip_comments(f.read())
        except OSError:
            return names
        for m in _INCLUDE_PATTERN.finditer(text):
            ref = m.group(1).strip()
            names.add(os.path.basename(ref[:-4] if ref.endswith(".tex") else ref))
        return names
