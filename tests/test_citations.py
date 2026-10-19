"""Tests for the source registry and the citation control."""

from __future__ import annotations

from langchain_core.documents import Document

from ela.agent.citations import (
    SourceRegistry,
    extract_citations,
    has_general_knowledge_tag,
    verify_citations,
    verify_legacy_citations,
)


def _doc(cid: str, **meta) -> Document:
    base = {"chunk_id": cid, "source": "UTS.tex", "course": "Univariate_Time_Series", "level": "M1"}
    return Document(page_content=f"contenu {cid}", metadata={**base, **meta})


def test_registry_assigns_sequential_ids_and_is_idempotent():
    reg = SourceRegistry()
    a = reg.register(_doc("c1", chapter="Chap. 2", slide_title="Stationnarité"), query="stationnarité")
    b = reg.register(_doc("c2"))
    again = reg.register(_doc("c1"))
    assert (a.sid, b.sid) == ("S1", "S2")
    assert again is a and len(reg) == 2
    assert reg.get("S2") is b and reg.get("s1") is a and reg.get("S9") is None and reg.get("X") is None
    assert a.breadcrumb == "Univariate_Time_Series › Chap. 2 › Stationnarité"
    assert a.label == "UTS.tex — Chap. 2 › Stationnarité"
    assert a.query == "stationnarité"


def test_registry_without_chunk_id_uses_content_hash():
    reg = SourceRegistry()
    d = Document(page_content="texte", metadata={"source": "X.tex"})
    assert reg.register(d).sid == "S1"
    assert reg.register(Document(page_content="texte", metadata={"source": "X.tex"})).sid == "S1"


def test_extract_and_verify_citations():
    text = "La stationnarité faible [S1] impose deux moments [S1, S2]. Voir aussi [S7] et [s3;S2]."
    assert extract_citations(text) == ["S1", "S2", "S7", "S3"]
    check = verify_citations(text, {"S1", "S2"})
    assert check.valid == ["S1", "S2"] and check.invalid == ["S7", "S3"]
    assert check.text == "La stationnarité faible [S1] impose deux moments [S1, S2]. Voir aussi et [S2]."
    assert not check.clean


def test_verify_citations_clean_text_untouched():
    text = "Réponse [S1]. [Connaissances générales] complément."
    check = verify_citations(text, ["S1"])
    assert check.clean and check.text == text
    assert has_general_knowledge_tag(text)
    assert not has_general_knowledge_tag("rien")


def test_registry_cited():
    reg = SourceRegistry()
    reg.register(_doc("c1"))
    reg.register(_doc("c2"))
    assert [r.sid for r in reg.cited("blabla [S2] [S5]")] == ["S2"]


def test_legacy_citations():
    text = "Def [Source: UTS.tex]. Autre [Source: MTS.tex — Chap 1 > VAR]. Hors [Source: Connaissances Générales]."
    check = verify_legacy_citations(text, ["UTS.tex", "HFE_C4.tex"])
    assert check.valid == ["UTS.tex", "Connaissances Générales"]
    assert check.invalid == ["MTS.tex — Chap 1 > VAR"]
    assert check.text == text
