"""Tests of the incremental ingestion pipeline and of the math normaliser."""

from __future__ import annotations

import json
from pathlib import Path

from ela.config import get_settings
from ela.ingestion import pipeline
from ela.mathfmt import normalize_math
from hashemb import HashEmb
from latex_fixture import make_course_tree


class MemoryStore:
    """Minimal in-memory :class:`ChunkStore` recording what the pipeline does."""

    backend = "memory"

    def __init__(self):
        self.rows: dict[str, tuple[str, list[float]]] = {}
        self.cleared = 0
        self.schema_calls = 0

    def ensure_schema(self):
        self.schema_calls += 1

    def fingerprint(self):
        return str(len(self.rows))

    def count(self):
        return len(self.rows)

    def existing(self):
        return {cid: h for cid, (h, _v) in self.rows.items()}

    def upsert(self, chunks, vectors):
        for c, v in zip(chunks, vectors, strict=True):
            self.rows[c.chunk_id] = (c.content_hash, v)

    def delete(self, ids):
        for i in ids:
            self.rows.pop(i, None)

    def clear(self):
        self.cleared += 1
        self.rows.clear()


def test_pipeline_is_incremental(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_DIM", "16")
    get_settings.cache_clear()
    latex = make_course_tree(tmp_path)
    store, emb = MemoryStore(), HashEmb(16)
    logs: list[str] = []

    dry = pipeline.run(source_dir=str(latex), dry_run=True, store=store, embeddings=emb, log=logs.append)
    assert dry.dry_run and dry.parsed > 0 and store.count() == 0 and dry.by_type["slide"] >= 4

    first = pipeline.run(source_dir=str(latex), store=store, embeddings=emb, log=logs.append)
    assert first.added == first.parsed == store.count() and first.unchanged == 0 and store.schema_calls == 1

    again = pipeline.run(source_dir=str(latex), store=store, embeddings=emb, log=logs.append)
    assert again.added == 0 and again.unchanged == first.parsed and again.removed == 0

    # Edit one slide: one chunk replaced, the rest untouched
    chapter = latex / "M1" / "Univariate_Time_Series" / "UTS_C1.tex"
    chapter.write_text(chapter.read_text(encoding="utf-8").replace("temps est discret", "temps est continu"), "utf-8")
    third = pipeline.run(source_dir=str(latex), store=store, embeddings=emb, log=logs.append)
    assert third.added == 1 and third.removed == 1 and third.unchanged == first.parsed - 1
    assert store.count() == first.parsed

    # Remove a course and dump the chunks
    (latex / "Commun" / "algebra.tex").unlink()
    dump = tmp_path / "chunks.jsonl"
    fourth = pipeline.run(
        source_dir=str(latex), full=True, dump=str(dump), store=store, embeddings=emb, log=logs.append
    )
    assert store.cleared == 1 and fourth.total == first.parsed - 1
    lines = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == fourth.total and {"text", "chunk_id", "slide_title", "doc_kind"} <= set(lines[0])


def test_pipeline_checks_embedding_dimension(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    get_settings.cache_clear()
    latex = make_course_tree(tmp_path)
    try:
        pipeline.run(source_dir=str(latex), store=MemoryStore(), embeddings=HashEmb(16), log=lambda m: None)
    except RuntimeError as exc:
        assert "EMBEDDING_DIM=8" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("dimension mismatch not detected")


def test_normalize_math():
    src = (
        "Soit \\(x_t\\) tel que \\[ y = \\beta x \\] et\n"
        "\\begin{align}\na &= b \\\\\nc &= d\n\\end{align}\nCode: `\\(a\\)`"
    )
    out = normalize_math(src)
    assert "$x_t$" in out and "\n$$\ny = \\beta x\n$$\n" in out
    assert "$$\n\\begin{aligned}\na &= b \\\\\nc &= d\n\\end{aligned}\n$$" in out
    assert "`\\(a\\)`" in out  # inline code untouched
    assert normalize_math("rien à faire $x$") == "rien à faire $x$"
    fenced = "```python\nprint('\\\\[x\\\\]')\n```"
    assert normalize_math(fenced) == fenced
