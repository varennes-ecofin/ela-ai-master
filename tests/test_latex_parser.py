"""Tests of the Phase 2 LaTeX parser (texclean + structure walker)."""

from __future__ import annotations

from pathlib import Path

from ela.ingestion import texclean
from ela.ingestion.latex import LatexIngestor, LatexParser
from latex_fixture import CHAPTER, MANUSCRIPT, MASTER, make_course_tree

# ---------------------------------------------------------------------------
# texclean
# ---------------------------------------------------------------------------


def test_strip_comments_keeps_escaped_percent():
    src = "a 50\\% b % comment\n%whole line\nc"
    assert texclean.strip_comments(src) == "a 50\\% b\nc"


def test_read_group_and_env_end():
    text = r"\textbf{a {nested} b} rest"
    assert texclean.read_group(text, 7) == ("a {nested} b", 21)
    env = r"\begin{itemize}\begin{itemize}\item x\end{itemize}\end{itemize} tail"
    end = texclean.find_env_end(env, "itemize", len(r"\begin{itemize}"))
    assert env[end[1] :] == " tail"


def test_macros_zero_arg_only():
    macros = texclean.collect_macros(r"\newcommand{\bX}{\boldsymbol{X}} \newcommand{\bt}[1]{\textbf{#1}} \def\pto{\to}")
    assert macros == {"bX": r"\boldsymbol{X}", "pto": r"\to"}
    assert texclean.expand_macros(r"$\bX_t \pto \bXy$", macros) == r"$\boldsymbol{X}_t \to \bXy$"


def test_protect_and_restore_math():
    src = r"Soit $x_t$ et \[ y = 2 \] puis \begin{align} a &= b \label{e} \\ c &= d \end{align} fin \(z\)"
    protected, segs = texclean.protect_math(src)
    assert "\x00MATH" in protected and len(segs) == 4
    out = texclean.restore_math(protected, segs)
    assert "$x_t$" in out and "$$ y = 2 $$" in out and "$z$" in out
    assert "\\begin{aligned}" in out and "\\label" not in out


def test_convert_frame_body():
    body = CHAPTER.split(r"\begin{frame}{Les séries temporelles}")[1].split(r"\end{frame}")[0]
    conv = texclean.convert(body, {"bX": r"\boldsymbol{X}"})
    assert conv.text.startswith("**Définition (Processus)** : Un **processus stochastique**")
    assert "$\\{X_{t}\\}_{t \\in \\mathbb{Z}}$" in conv.text
    assert "- Note 1 on suppose" in conv.text
    assert "- Le vecteur $\\boldsymbol{X}$ est en gras **important** 50%" in conv.text
    assert "comment" not in conv.text and "\\medskip" not in conv.text
    assert [e.kind for e in conv.elements] == ["definition", "definition"]
    assert conv.elements[0].title == "Processus" and conv.elements[1].title == ""
    assert conv.notes == ["le symbole $\\supseteq$ dénote un sur-ensemble"]


def test_convert_columns_table_code():
    body = CHAPTER.split(r"\frametitle{Stationnarité}")[1].split(r"\end{frame}")[0]
    text = texclean.convert(body).text
    assert "**stationnaire au second ordre**" in text
    assert "$$\\begin{aligned}" in text and "\\end{aligned}$$" in text and "\\label" not in text
    assert "[Figure : Une trajectoire]" in text and "includegraphics" not in text

    body = CHAPTER.split(r"\begin{frame}[fragile]{Un tableau}")[1].split(r"\end{frame}")[0]
    text = texclean.convert(body).text
    assert "[Tableau]" in text and "Modèle | AIC | BIC" in text and "AR(1) | 10.2 | 11.0" in text
    assert "```\nimport numpy as np\n```" in text and "\\pause" not in text and "toprule" not in text


# ---------------------------------------------------------------------------
# LatexParser
# ---------------------------------------------------------------------------


def _parse_slides() -> list:
    content = MASTER.replace(r"\include{UTS_C1}", CHAPTER)
    return LatexParser("UTS.tex", "M1", "Univariate_Time_Series").parse(content)


def test_parser_beamer_structure():
    chunks = _parse_slides()
    slides = [c for c in chunks if c.chunk_type == "slide"]
    titles = [c.slide_title for c in slides]
    # title-only/figure-only, commented and bibliography frames are dropped
    assert titles == ["Les séries temporelles", "Stationnarité", "Un tableau"]
    first = slides[0]
    assert first.doc_kind == "slides" and first.chapter == "Chapter 1 — Les Processus ARMA"
    assert first.section == "Introduction" and first.source == "UTS.tex" and first.level == "M1"
    assert slides[2].section == "Tableaux"
    assert first.page_content().startswith(
        "COURS: Univariate_Time_Series (M1)\nCHAPITRE: Chapter 1 — Les Processus ARMA\nSECTION: Introduction\n"
        "SLIDE: Les séries temporelles\nCONTENU:\n"
    )
    assert "NOTES DE L'ENSEIGNANT:" in first.page_content()


def test_parser_elements_are_linked_to_parent():
    chunks = _parse_slides()
    elements = [c for c in chunks if c.chunk_type == "definition"]
    parent = next(c for c in chunks if c.slide_title == "Les séries temporelles" and c.chunk_type == "slide")
    assert len(elements) == 2 and all(e.parent_id == parent.chunk_id for e in elements)
    assert elements[0].element_title == "Processus" and elements[0].slide_title == parent.slide_title
    meta = elements[0].metadata()
    assert meta["type"] == "element" and meta["doc_kind"] == "slides" and meta["n_chars"] == len(elements[0].text)
    assert all(isinstance(v, (str, int)) for v in meta.values())


def test_chunk_ids_are_stable_and_content_based():
    a = _parse_slides()
    b = _parse_slides()
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)
    changed = MASTER.replace(r"\include{UTS_C1}", CHAPTER.replace("temps est discret", "temps est continu"))
    c = LatexParser("UTS.tex", "M1", "Univariate_Time_Series").parse(changed)
    diff = {x.chunk_id for x in c} ^ {x.chunk_id for x in a}
    assert len(diff) == 2  # the modified slide, old and new id


def test_parser_manuscript():
    chunks = LatexParser("hdr_main.tex", "M2", "Research_Econometrics", target_chars=200).parse(MANUSCRIPT)
    texts = [c for c in chunks if c.chunk_type == "text"]
    assert all(c.doc_kind == "manuscript" and c.slide_title == "" for c in texts)
    assert texts[0].chapter == "Modélisation" and texts[0].section == ""
    section_chunks = [c for c in texts if c.section == "La filtration"]
    assert len(section_chunks) >= 2 and all(c.chapter == "Modélisation" for c in section_chunks)
    assert "$$ y_t = \\sum_{k} d_k \\varepsilon_{t-k}. $$" in section_chunks[0].text
    theorem = next(c for c in chunks if c.chunk_type == "theorem")
    assert theorem.element_title == "Représentabilité" and theorem.parent_id == section_chunks[0].chunk_id
    assert theorem.location == "Modélisation > La filtration"


# ---------------------------------------------------------------------------
# LatexIngestor (tree scan)
# ---------------------------------------------------------------------------


def test_ingestor_scans_tree_and_skips_included_files(tmp_path: Path):
    latex = make_course_tree(tmp_path)
    messages: list[str] = []
    report = LatexIngestor(str(latex), log=messages.append).scan()
    assert set(report.stats) == {"M1/Univariate_Time_Series", "M2/Research_Econometrics", "Commun/Général"}
    sources = {c.source for c in report.chunks}
    assert sources == {"UTS.tex", "hdr_main.tex", "algebra.tex"}  # UTS_C1.tex merged into UTS.tex
    assert any("Skip" in m for m in messages)
    assert report.by_type()["slide"] >= 4 and report.warnings == []
    docs, stats = LatexIngestor(str(latex), log=lambda m: None).load_documents()
    assert len(docs) == len(report.chunks) and stats == report.stats
