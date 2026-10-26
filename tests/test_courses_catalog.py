"""Course tree operations (ela.courses.catalog) and the disk-vs-store diagnostic (ela.courses.state)."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime

import pytest

from ela.courses import catalog, state
from ela.courses.state import CourseStats, KnowledgeBase


@pytest.fixture
def root(tmp_path):
    (tmp_path / "M1" / "Series_Temporelles").mkdir(parents=True)
    (tmp_path / "M1" / "Series_Temporelles" / "main.tex").write_text("\\documentclass{beamer}", encoding="utf-8")
    (tmp_path / "M1" / "Series_Temporelles" / "ch1.tex").write_text("x", encoding="utf-8")
    (tmp_path / "M1" / "Series_Temporelles" / "img").mkdir()
    (tmp_path / "M1" / "Series_Temporelles" / "img" / "fig.tex").write_text("y", encoding="utf-8")
    (tmp_path / "M2" / "Vide").mkdir(parents=True)
    (tmp_path / "Commun").mkdir()
    (tmp_path / "Commun" / "loose.tex").write_text("z", encoding="utf-8")
    (tmp_path / "Autre").mkdir()
    return tmp_path


def test_scan(root):
    tree = catalog.scan(root)
    assert tree.exists and [c.name for c in tree.courses["M1"]] == ["Series_Temporelles"]
    c = tree.course("M1", "Series_Temporelles")
    assert c is not None and c.key == "M1/Series_Temporelles" and c.tex_count == 2 and len(c.files) == 3
    assert [f.path for f in c.files] == ["ch1.tex", "img/fig.tex", "main.tex"]
    assert not next(f for f in c.files if f.path == "img/fig.tex").ingested
    assert isinstance(c.files[0].modified, datetime) and c.size == 24
    assert tree.course("M2", "Vide").tex_count == 0
    assert tree.stray == ["Autre/", "Commun/loose.tex"]
    assert [c.key for c in tree.all_courses] == ["M1/Series_Temporelles", "M2/Vide"]
    missing = catalog.scan(root / "nope")
    assert not missing.exists and missing.all_courses == []


def test_validation():
    with pytest.raises(catalog.CatalogError):
        catalog.check_level("L3")
    assert catalog.check_course_name(" Series_Temporelles ") == "Series_Temporelles"
    for bad in ("", "a", "Séries", "a b", "../x", "x" * 65):
        with pytest.raises(catalog.CatalogError):
            catalog.check_course_name(bad)
    assert catalog.check_relative_path("chap\\sec/intro.tex") == "chap/sec/intro.tex"
    for bad in ("", "../x.tex", "/etc/passwd", "a/../b.tex", "bad name.tex"):
        with pytest.raises(catalog.CatalogError):
            catalog.check_relative_path(bad)
    with pytest.raises(catalog.CatalogError):
        catalog.check_extension("notes.pdf")
    assert catalog.human_size(512) == "512 o" and catalog.human_size(1536) == "1,5 Ko"


def test_create_save_delete(root):
    folder = catalog.create_course(root, "M2", "Nouveau")
    assert folder.is_dir()
    with pytest.raises(catalog.CatalogError):
        catalog.create_course(root, "M2", "Nouveau")
    assert catalog.save_file(root, "M2", "Nouveau", "main.tex", b"a") == ("main.tex", False)
    assert catalog.save_file(root, "M2", "Nouveau", "main.tex", b"b") == ("main.tex", True)
    assert catalog.save_file(root, "M2", "Nouveau", "sub/ch.tex", b"c") == ("sub/ch.tex", False)
    with pytest.raises(catalog.CatalogError):
        catalog.save_file(root, "M2", "Nouveau", "x.pdf", b"")
    with pytest.raises(catalog.CatalogError):
        catalog.save_file(root, "M2", "Nouveau", "big.tex", b"x" * (catalog.MAX_UPLOAD_BYTES + 1))
    with pytest.raises(catalog.CatalogError):
        catalog.save_file(root, "M2", "Inconnu", "main.tex", b"")
    assert catalog.file_path(root, "M2", "Nouveau", "sub/ch.tex").read_bytes() == b"c"
    catalog.delete_file(root, "M2", "Nouveau", "sub/ch.tex")
    assert not (folder / "sub").exists()  # emptied sub-folder removed
    with pytest.raises(catalog.CatalogError):
        catalog.delete_file(root, "M2", "Nouveau", "sub/ch.tex")
    assert catalog.delete_course(root, "M2", "Nouveau") == 1
    assert not folder.exists()


def test_extract_zip(root):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Cours/main.tex", "m")
        z.writestr("Cours/chapters/ch1.tex", "c")
        z.writestr("Cours/macros.sty", "s")
        z.writestr("Cours/figures/plot.png", "p")
        z.writestr("Cours/.DS_Store", "junk")
        z.writestr("__MACOSX/Cours/._main.tex", "junk")
    catalog.create_course(root, "M2", "Zip")
    report = catalog.extract_zip(root, "M2", "Zip", buf.getvalue())
    assert report.written == ["main.tex", "chapters/ch1.tex", "macros.sty"]
    assert sorted(report.skipped) == sorted(["figures/plot.png", "Cours/.DS_Store", "__MACOSX/Cours/._main.tex"])
    assert (root / "M2" / "Zip" / "chapters" / "ch1.tex").read_text() == "c"
    report = catalog.extract_zip(root, "M2", "Zip", buf.getvalue())
    assert report.replaced == ["main.tex", "chapters/ch1.tex", "macros.sty"] and report.written == []
    # malicious member
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../../evil.tex", "e")
        z.writestr("ok.tex", "o")
    report = catalog.extract_zip(root, "M2", "Zip", buf.getvalue())
    assert report.skipped == ["../../evil.tex"] and report.written == ["ok.tex"]
    assert not (root.parent / "evil.tex").exists()
    with pytest.raises(catalog.CatalogError):
        catalog.extract_zip(root, "M2", "Zip", b"not a zip")


def test_move_course(root):
    target = catalog.move_course(root, "M1", "Series_Temporelles", "M2")
    assert target == root / "M2" / "Series_Temporelles" and (target / "main.tex").exists()
    assert catalog.scan(root).course("M1", "Series_Temporelles") is None
    with pytest.raises(catalog.CatalogError):
        catalog.move_course(root, "M2", "Series_Temporelles", "M2")
    (root / "Commun" / "Series_Temporelles").mkdir()
    with pytest.raises(catalog.CatalogError):
        catalog.move_course(root, "M2", "Series_Temporelles", "Commun")
    with pytest.raises(catalog.CatalogError):
        catalog.move_course(root, "M1", "Inconnu", "M2")


class FakeStore:
    backend = "fake"

    def __init__(self, rows):
        self.rows = rows

    def stats(self):
        return self.rows

    def fingerprint(self):
        return "fake:1"


def test_knowledge_base_and_diagnose(root):
    rows = [
        {
            "level": "M1",
            "course": "Series_Temporelles",
            "doc_kind": "slides",
            "chunk_type": "slide",
            "n": 10,
            "updated_at": datetime(2026, 10, 1),
        },
        {
            "level": "M1",
            "course": "Series_Temporelles",
            "doc_kind": "slides",
            "chunk_type": "definition",
            "n": 2,
            "updated_at": datetime(2026, 10, 2),
        },
        {
            "level": "M2",
            "course": "Disparu",
            "doc_kind": "manuscript",
            "chunk_type": "text",
            "n": 3,
            "updated_at": None,
        },
    ]
    kb = state.knowledge_base(FakeStore(rows))
    assert kb.error is None and kb.total == 15 and kb.backend == "fake" and kb.fingerprint == "fake:1"
    assert kb.by_level == {"M1": 12, "M2": 3} and kb.by_type == {"slide": 10, "definition": 2, "text": 3}
    st = kb.course("M1", "Series_Temporelles")
    assert st.chunks == 12 and st.by_type == {"slide": 10, "definition": 2} and st.updated_at == datetime(2026, 10, 2)
    assert kb.course("M2", "Nope") is None

    diag = state.diagnose(catalog.scan(root), kb)
    assert [(d.key, d.status) for d in diag] == [
        ("M1/Series_Temporelles", state.OK),
        ("M2/Disparu", state.ORPHAN),
        ("M2/Vide", state.EMPTY),
    ]
    assert diag[0].tex_files == 2 and diag[0].chunks == 12 and diag[1].chunks == 3

    class Broken:
        backend = "x"

        def stats(self):
            raise RuntimeError("down")

    kb = state.knowledge_base(Broken())
    assert kb.error == "RuntimeError: down" and kb.total == 0
    diag = state.diagnose(catalog.scan(root), KnowledgeBase(backend="x"))
    assert [(d.key, d.status) for d in diag] == [("M1/Series_Temporelles", state.TO_INGEST), ("M2/Vide", state.EMPTY)]
    assert CourseStats("M1", "X", 1).key == "M1/X"
