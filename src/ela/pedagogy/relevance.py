"""Check that the retrieved material actually covers the requested topic.

The retriever always returns the *nearest* passages: asked for a quiz on
GARCH models, a student whose level only sees the univariate course gets
ARMA slides back. Before generating a quiz or an exercise, the generators
check that the topic's own terms appear in the passages; otherwise they
report that the course has nothing on the topic instead of drifting.

The check is lexical and cheap (no model call): accents are stripped,
words are compared on a short prefix (``estimateurs`` ≈ ``estimateur``),
acronyms written in capitals (``GARCH``, ``ADF``) must all appear, and at
least half of the remaining terms must.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

# Function words and words too generic to tell topics apart.
_STOPWORDS_RAW = """
    le la les un une des du de d l au aux et ou en sur pour dans par avec sans sous entre vers chez
    ce cet cette ces son sa ses leur leurs mon ma mes ton ta tes notre votre nos vos
    qui que quoi dont où est sont a ont être avoir il elle ils elles on nous vous je tu
    plus moins très trop peu tout tous toute toutes autre autres même mêmes
    the of and or in on for with to a an
    modele modeles model models test tests methode methodes cours notion notions question questions
    quiz exercice exercices chapitre chapitres theorie approche analyse introduction rappel rappels
    processus propriete proprietes definition definitions exemple exemples cas
    """.split()
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+")
_PREFIX = 6


def strip_accents(text: str) -> str:
    """ASCII form of ``text`` (accents stripped, case kept)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """Lower-case ASCII form of ``text`` (accents stripped)."""
    return strip_accents(text).lower()


_STOPWORDS = frozenset(normalize(w) for w in _STOPWORDS_RAW)


@dataclass(frozen=True)
class Term:
    """One topic term.

    Attributes:
        key: Comparison key (lower-case prefix) — or the acronym as written,
            since case tells ``VaR`` (value at risk) from ``VAR`` (vector
            autoregression).
        required: Whether the term must appear (acronyms).
    """

    key: str
    required: bool

    def found_in(self, corpus: str, corpus_lower: str) -> bool:
        """Whether the term appears in the (accent-stripped) corpus."""
        return self.key in corpus if self.required else self.key in corpus_lower


def topic_terms(topic: str) -> list[Term]:
    """Terms of a topic worth looking for in the passages.

    Args:
        topic: Topic as typed by the student or chosen by the model.

    Returns:
        Terms (deduplicated, in order). Acronyms (two capitals or more, such
        as ``GARCH`` or ``VaR``) are required and matched as written; other
        words are compared, case-insensitively, on their first ``6`` letters.
    """
    terms: list[Term] = []
    seen: set[str] = set()
    for word in _WORD_RE.findall(topic):
        key = normalize(word)
        if len(key) < 2 or key in _STOPWORDS:
            continue
        acronym = sum(1 for c in word if c.isupper()) >= 2
        if acronym:
            key = strip_accents(word)
        elif len(key) < 3:
            continue
        elif len(key) > _PREFIX:
            key = key[:_PREFIX]
        if key in seen:
            continue
        seen.add(key)
        terms.append(Term(key=key, required=acronym))
    return terms


def covers(topic: str, texts: Iterable[str], *, all_terms: bool = False) -> bool:
    """Whether ``texts`` (passages, titles) cover ``topic``.

    Args:
        topic: Requested topic.
        texts: Texts of the retrieved material.
        all_terms: Require every term (used against a whole course, where
            half of generic terms would match anything).

    Returns:
        ``True`` when every acronym of the topic and at least half of its
        other terms (all of them with ``all_terms``) appear in the texts, or
        when the topic has no usable term.
    """
    terms = topic_terms(topic)
    if not terms:
        return True
    corpus = strip_accents(" ".join(texts))
    corpus_lower = corpus.lower()
    matched = [t for t in terms if t.found_in(corpus, corpus_lower)]
    if any(t.required and t not in matched for t in terms):
        return False
    needed = len(terms) if all_terms else math.ceil(len(terms) / 2)
    return len(matched) >= needed
