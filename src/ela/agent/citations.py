"""Source registry and citation control.

Every passage the agent reads during one turn (search results, slides opened
with ``get_slide``) is registered here and receives a short identifier
``S1``, ``S2``... The model cites passages with ``[S1]`` and the final answer
is checked against the registry: a bracket whose identifier was never
handed to the model is a *phantom citation* and is removed.

The legacy ``[Source: file]`` format is still parsed so that the end-to-end
evaluation can measure the pre-agent pipeline.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from langchain_core.documents import Document

#: Tag the model uses for knowledge that does not come from the course.
GENERAL_KNOWLEDGE_TAG = "[Connaissances générales]"

_SID_RE = re.compile(r"^S\d+$")
CITATION_RE = re.compile(r"\[\s*([sS]\d+(?:\s*[,;]\s*[sS]\d+)*)\s*\]")
LEGACY_CITATION_RE = re.compile(r"\[Source\s*:\s*([^\]]+)\]", re.IGNORECASE)
_GENERAL_RE = re.compile(r"\[\s*Connaissances\s+g[ée]n[ée]rales\s*\]", re.IGNORECASE)


@dataclass
class SourceRef:
    """A passage handed to the model during the current turn.

    Attributes:
        sid: Short identifier used in citations (``S1``).
        chunk_id: Stable chunk identifier in the store.
        source: Master ``.tex`` file name.
        course: Course folder name.
        level: Academic level of the chunk.
        chapter: Chapter title (may be empty).
        section: Section title (may be empty).
        slide_title: Slide or element title (may be empty).
        content: Passage text (what the model saw).
        score: Fusion score from the retriever, when available.
        rerank_score: Reranker score, when available.
        origin: ``search`` or ``slide``.
        query: Search query that surfaced the passage (``search`` origin).
    """

    sid: str
    chunk_id: str
    source: str
    course: str
    level: str
    chapter: str
    section: str
    slide_title: str
    content: str
    score: float | None = None
    rerank_score: float | None = None
    origin: str = "search"
    query: str | None = None

    @property
    def breadcrumb(self) -> str:
        """``course › chapter › slide`` (empty parts skipped)."""
        parts = [p for p in (self.course or self.source, self.chapter, self.section, self.slide_title) if p]
        return " › ".join(dict.fromkeys(parts))  # drop exact duplicates, keep order

    @property
    def label(self) -> str:
        """Human-readable label with the file name."""
        where = " › ".join(p for p in (self.chapter, self.section, self.slide_title) if p)
        return f"{self.source} — {where}" if where else self.source

    @classmethod
    def from_document(cls, sid: str, doc: Document, *, origin: str = "search", query: str | None = None) -> SourceRef:
        """Build a reference from a retrieved document.

        Args:
            sid: Identifier to assign.
            doc: Retrieved document (chunk metadata in ``doc.metadata``).
            origin: ``search`` or ``slide``.
            query: Query that surfaced the document.

        Returns:
            The reference.
        """
        meta = doc.metadata or {}
        return cls(
            sid=sid,
            chunk_id=document_key(doc),
            source=str(meta.get("source") or "Inconnu"),
            course=str(meta.get("course") or ""),
            level=str(meta.get("level") or ""),
            chapter=str(meta.get("chapter") or ""),
            section=str(meta.get("section") or ""),
            slide_title=str(meta.get("slide_title") or meta.get("element_title") or ""),
            content=doc.page_content,
            score=_as_float(meta.get("score")),
            rerank_score=_as_float(meta.get("rerank_score")),
            origin=origin,
            query=query,
        )


def _as_float(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def document_key(doc: Document) -> str:
    """Stable key of a document: its ``chunk_id`` or a hash of its content."""
    cid = (doc.metadata or {}).get("chunk_id")
    if cid:
        return str(cid)
    return "h:" + hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()[:16]


@dataclass
class SourceRegistry:
    """Passages read during one turn, in order of first appearance."""

    _refs: list[SourceRef] = field(default_factory=list)
    _by_key: dict[str, SourceRef] = field(default_factory=dict)

    def register(self, doc: Document, *, origin: str = "search", query: str | None = None) -> SourceRef:
        """Register a document (idempotent per chunk) and return its reference.

        Args:
            doc: Retrieved document.
            origin: ``search`` or ``slide``.
            query: Query that surfaced it.

        Returns:
            The existing reference when the chunk was already registered,
            otherwise a new one with the next identifier.
        """
        key = document_key(doc)
        ref = self._by_key.get(key)
        if ref is None:
            ref = SourceRef.from_document(f"S{len(self._refs) + 1}", doc, origin=origin, query=query)
            self._refs.append(ref)
            self._by_key[key] = ref
        return ref

    def register_all(self, docs: Iterable[Document], **kwargs) -> list[SourceRef]:
        """Register several documents, preserving order."""
        return [self.register(d, **kwargs) for d in docs]

    def get(self, sid: str) -> SourceRef | None:
        """Return the reference for an identifier (``S3``), or ``None``."""
        sid = sid.strip().upper()
        if not _SID_RE.match(sid):
            return None
        index = int(sid[1:]) - 1
        return self._refs[index] if 0 <= index < len(self._refs) else None

    def by_chunk(self, chunk_id: str) -> SourceRef | None:
        """Return the reference for a chunk id, or ``None``."""
        return self._by_key.get(chunk_id)

    def ids(self) -> set[str]:
        """Set of identifiers handed out so far."""
        return {r.sid for r in self._refs}

    def cited(self, text: str) -> list[SourceRef]:
        """Return the references actually cited in ``text`` (document order, unique)."""
        wanted = set(extract_citations(text))
        return [r for r in self._refs if r.sid in wanted]

    def __iter__(self) -> Iterator[SourceRef]:
        return iter(self._refs)

    def __len__(self) -> int:
        return len(self._refs)


# ---------------------------------------------------------------------------
# Parsing and verification
# ---------------------------------------------------------------------------


def extract_citations(text: str) -> list[str]:
    """Return the identifiers cited with ``[S1]`` / ``[S1, S2]`` brackets (unique, in order).

    Args:
        text: Answer text.

    Returns:
        Ordered list of identifiers such as ``["S1", "S3"]``.
    """
    seen: dict[str, None] = {}
    for m in CITATION_RE.finditer(text):
        for sid in re.split(r"\s*[,;]\s*", m.group(1)):
            seen.setdefault(sid.strip().upper(), None)
    return list(seen)


def has_general_knowledge_tag(text: str) -> bool:
    """Whether the answer tags some content as general (non-course) knowledge."""
    return bool(_GENERAL_RE.search(text))


@dataclass(frozen=True)
class CitationCheck:
    """Result of :func:`verify_citations`.

    Attributes:
        text: Answer with phantom citations removed.
        valid: Identifiers that exist in the registry (in order of appearance).
        invalid: Identifiers that were cited but never handed to the model.
    """

    text: str
    valid: list[str]
    invalid: list[str]

    @property
    def clean(self) -> bool:
        """``True`` when no phantom citation was found."""
        return not self.invalid


def verify_citations(text: str, known_ids: Iterable[str]) -> CitationCheck:
    """Remove citations whose identifier is unknown.

    A bracket mixing known and unknown identifiers keeps the known ones
    (``[S1, S9]`` → ``[S1]``); a bracket with only unknown ones disappears
    together with a preceding space.

    Args:
        text: Answer text.
        known_ids: Identifiers handed to the model (``S1``...).

    Returns:
        The cleaned text and the lists of valid / invalid identifiers.
    """
    known = {k.strip().upper() for k in known_ids}
    valid: dict[str, None] = {}
    invalid: dict[str, None] = {}

    def _replace(m: re.Match) -> str:
        sids = [s.strip().upper() for s in re.split(r"\s*[,;]\s*", m.group(1))]
        keep = []
        for sid in sids:
            if sid in known:
                valid.setdefault(sid, None)
                if sid not in keep:
                    keep.append(sid)
            else:
                invalid.setdefault(sid, None)
        return f"[{', '.join(keep)}]" if keep else ""

    cleaned = CITATION_RE.sub(_replace, text)
    cleaned = re.sub(r"[ \t]+([.,)])", r"\1", cleaned)  # "mot ." left by a removed bracket (French keeps " ;" " :")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return CitationCheck(text=cleaned, valid=list(valid), invalid=list(invalid))


def verify_legacy_citations(text: str, retrieved_sources: Iterable[str]) -> CitationCheck:
    """Check ``[Source: …]`` citations of the pre-agent pipeline.

    A citation is valid when it names a retrieved file (``UTS.tex``, with or
    without the chapter/slide suffix) or the general-knowledge marker. The text
    is returned unchanged (the legacy pipeline never rewrote its answers).

    Args:
        text: Answer text.
        retrieved_sources: ``source`` metadata of the documents given to the model.

    Returns:
        Valid / invalid cited labels.
    """
    files = {_norm_file(s) for s in retrieved_sources}
    valid: dict[str, None] = {}
    invalid: dict[str, None] = {}
    for m in LEGACY_CITATION_RE.finditer(text):
        label = m.group(1).strip()
        if _GENERAL_RE.search(f"[{label}]") or "connaissances" in label.lower():
            valid.setdefault(label, None)
            continue
        file_part = re.split(r"\s+[—–-]\s+|\s*>\s*|\s*›\s*", label, maxsplit=1)[0].strip()
        if _norm_file(file_part) in files or any(_norm_file(file_part) == f for f in files):
            valid.setdefault(label, None)
        else:
            invalid.setdefault(label, None)
    return CitationCheck(text=text, valid=list(valid), invalid=list(invalid))


def _norm_file(name: str) -> str:
    name = name.strip().lower()
    return name[:-4] if name.endswith(".tex") else name
