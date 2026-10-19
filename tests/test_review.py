"""Tests for the review sheets (generation from the chapter slides, cache, tool)."""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

import ela.db.review_sheets as cache_mod
from ela.agent.outline import build_outlines
from ela.agent.tools import generate_review_sheet as sheet_tool
from ela.pedagogy.review import (
    ReviewLookupError,
    chapter_slides,
    generate_review_sheet,
    merge_payloads,
    resolve_chapter,
    resolve_target,
    review_sheet_for,
    split_parts,
)
from ela.pedagogy.schemas import ReviewSection, ReviewSheet, ReviewSheetPayload
from fakes import ScriptedChatModel
from test_agent_tools import CORPUS, _runtime, ctx  # noqa: F401  (fixture)


def _payload(title="Processus stationnaires", n_sections=2) -> dict:
    return {
        "title": title,
        "objectives": ["Savoir définir la stationnarité faible"],
        "sections": [
            {
                "heading": f"Section {i}",
                "key_points": [f"Point {i}"],
                "definitions": ["Stationnarité faible : moments d'ordre 2 invariants"] if i == 1 else [],
                "formulas": ["$E(X_t)=\\mu$"] if i == 1 else [],
                "pitfalls": [],
            }
            for i in range(1, n_sections + 1)
        ],
        "takeaways": ["La stationnarité faible porte sur les deux premiers moments"],
    }


@pytest.fixture
def no_cache(monkeypatch: pytest.MonkeyPatch):
    """Cache disabled (no database in the offline tests)."""
    stored = []

    async def _get(*a, **k):
        return None

    async def _store(sheet, **k):
        stored.append((sheet, k))
        return True

    monkeypatch.setattr(cache_mod, "get_cached_sheet", _get)
    monkeypatch.setattr(cache_mod, "store_sheet", _store)
    return stored


def test_resolve_chapter_by_title_number_and_fragment():
    uts = build_outlines(CORPUS)["Univariate_Time_Series"]
    assert resolve_chapter(uts, "Chapitre 1 : processus").title == "Chapitre 1 : processus"
    assert resolve_chapter(uts, "2").title == "Chapitre 2 : racines unitaires"
    assert resolve_chapter(uts, "chapitre 2").title == "Chapitre 2 : racines unitaires"
    assert resolve_chapter(uts, "Chap. 1").title == "Chapitre 1 : processus"
    assert resolve_chapter(uts, "racines unitaires").title == "Chapitre 2 : racines unitaires"
    assert resolve_chapter(uts, "les racines").title == "Chapitre 2 : racines unitaires"
    assert resolve_chapter(uts, "cointégration") is None and resolve_chapter(uts, "") is None


def test_chapter_slides_and_parts():
    uts = build_outlines(CORPUS)["Univariate_Time_Series"]
    slides = chapter_slides(uts, uts.chapters[0])
    assert [t for t, _ in slides] == ["Stationnarité faible", "AR(1) et solution stationnaire"]
    assert split_parts(slides, max_chars=10_000) == [slides]
    parts = split_parts(slides, max_chars=60)
    assert len(parts) == 2 and parts[0] == [slides[0]]


def test_merge_payloads_concatenates_sections_and_deduplicates():
    a = ReviewSheetPayload(**_payload("A"))
    b = ReviewSheetPayload(
        title="B",
        objectives=["Savoir définir la stationnarité faible", "Savoir inverser un AR(1)"],
        sections=[ReviewSection(heading="S3", key_points=["p"])],
        takeaways=["Autre point"],
    )
    merged = merge_payloads([a, b])
    assert merged.title == "A" and len(merged.sections) == 3
    assert merged.objectives == ["Savoir définir la stationnarité faible", "Savoir inverser un AR(1)"]
    assert len(merged.takeaways) == 2
    assert merge_payloads([a]) is a


async def test_generate_review_sheet_from_slides_in_parts(monkeypatch: pytest.MonkeyPatch):
    import ela.pedagogy.review as review_mod

    monkeypatch.setattr(review_mod, "PART_MAX_CHARS", 60)  # two parts for the two slides
    uts = build_outlines(CORPUS)["Univariate_Time_Series"]
    llm = ScriptedChatModel([json.dumps(_payload("Partie 1", 1)), json.dumps(_payload("Partie 2", 1))])
    sheet = await generate_review_sheet(llm, uts, uts.chapters[0], level="M1")
    assert sheet.course == "Univariate_Time_Series" and sheet.chapter == "Chapitre 1 : processus"
    assert sheet.level == "M1" and sheet.slides == ["Stationnarité faible", "AR(1) et solution stationnaire"]
    assert sheet.title == "Partie 1" and len(sheet.sections) == 2
    assert len(llm.calls) == 2
    first, second = llm.calls[0][0].content, llm.calls[1][0].content
    assert "(partie 1/2 du chapitre)" in first and "— Stationnarité faible —" in first
    assert "(partie 2/2 du chapitre)" in second and "AR(1) solution stationnarité phi" in second
    md = sheet.to_markdown()
    assert md.startswith("# 📄 Partie 1") and "## Objectifs" in md and "- $E(X_t)=\\mu$" in md and "## À retenir" in md
    assert "*Slides du chapitre : Stationnarité faible · AR(1) et solution stationnaire*" in md


async def test_generate_review_sheet_requires_slides():
    uts = build_outlines(CORPUS)["Univariate_Time_Series"]
    empty = type(uts.chapters[0])(title="Chapitre 9 : vide")
    with pytest.raises(ValueError):
        await generate_review_sheet(ScriptedChatModel([]), uts, empty)


def test_resolve_target_scoped_to_level(ctx):  # noqa: F811
    outline, chapter = resolve_target("M1", "UTS", "2")
    assert outline.course == "Univariate_Time_Series" and chapter.title == "Chapitre 2 : racines unitaires"
    with pytest.raises(ReviewLookupError, match="Cours « HFE » inconnu"):
        resolve_target("M1", "HFE", "4")  # M2 course hidden from an M1 student
    with pytest.raises(ReviewLookupError, match="Chapitre « 7 » inconnu"):
        resolve_target("M1", "UTS", "7")
    outline, chapter = resolve_target("ALL", "HFE_C4", "VaR")
    assert outline.course == "Financial_Econometrics" and chapter.title == "Chapitre 4 : VaR"


async def test_review_sheet_for_uses_the_cache(ctx, no_cache, monkeypatch):  # noqa: F811
    llm = ScriptedChatModel([json.dumps(_payload())])
    sheet = await review_sheet_for(llm, "M1", "UTS", "1")
    assert isinstance(sheet, ReviewSheet) and len(no_cache) == 1
    stored, key = no_cache[0]
    assert stored is sheet and key["prompt_version"] == "v1" and key["store_fingerprint"]

    async def _cached(*a, **k):
        return stored

    monkeypatch.setattr(cache_mod, "get_cached_sheet", _cached)
    again = await review_sheet_for(ScriptedChatModel([]), "M1", "UTS", "1")  # no model call
    assert again is stored


async def test_review_sheet_tool(ctx, no_cache):  # noqa: F811
    ctx.llm = GenericFakeChatModel(messages=iter([json.dumps(_payload())]))
    out = await sheet_tool.ainvoke({"course": "UTS", "chapter": "chapitre 1", "runtime": _runtime(ctx)})
    assert out.startswith(
        "Fiche « Processus stationnaires » générée pour Univariate_Time_Series › Chapitre 1 : processus"
    )
    assert ctx.review_sheet is not None and ctx.review_sheet.slides[0] == "Stationnarité faible"
    out = await sheet_tool.ainvoke({"course": "UTS", "chapter": "chapitre 9", "runtime": _runtime(ctx)})
    assert out.startswith("Chapitre « chapitre 9 » inconnu dans Univariate_Time_Series. Chapitres :")
    out = await sheet_tool.ainvoke({"course": "HFE", "chapter": "4", "runtime": _runtime(ctx)})
    assert out.startswith("Cours « HFE » inconnu pour ce niveau")


async def test_review_sheet_tool_failure_is_reported(ctx, no_cache):  # noqa: F811
    ctx.llm = GenericFakeChatModel(messages=iter(["pas du JSON"]))
    ctx.review_sheet = None
    out = await sheet_tool.ainvoke({"course": "UTS", "chapter": "1", "runtime": _runtime(ctx)})
    assert out.startswith("La génération de la fiche a échoué") and ctx.review_sheet is None


async def test_engine_generate_review_sheet(ctx, no_cache):  # noqa: F811
    from ela.engine import ElaEngine, GenerationError

    eng = ElaEngine("M1", retriever=ctx.retriever, llm=ScriptedChatModel([json.dumps(_payload())]), agent=object())
    sheet = await eng.generate_review_sheet("UTS", "1")
    assert sheet.chapter == "Chapitre 1 : processus"
    with pytest.raises(ReviewLookupError):
        await eng.generate_review_sheet("HFE", "4")
    eng = ElaEngine("M1", retriever=ctx.retriever, llm=ScriptedChatModel(["pas du JSON"]), agent=object())
    with pytest.raises(GenerationError):
        await eng.generate_review_sheet("UTS", "1")
