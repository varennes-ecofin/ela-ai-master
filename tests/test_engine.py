"""Tests for ``ela.engine.ElaEngine`` with fake retriever / LLM / guard."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from ela.engine import ElaEngine, chunk_text, format_context
from ela.guard import GradeResult


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.docs


class FakeGuard:
    def __init__(self, result: GradeResult):
        self.result = result

    async def grade(self, question, docs):
        return self.result


DOCS = [Document(page_content="Un processus est stationnaire si ...", metadata={"source": "UTS_C1.tex"})]


def _engine(answer: str, grade: GradeResult, images_model: str | None = None) -> ElaEngine:
    llm = GenericFakeChatModel(messages=iter([answer]))
    vision = GenericFakeChatModel(messages=iter([images_model or "vision answer"]))
    eng = ElaEngine("M1", retriever=FakeRetriever(DOCS), llm=llm, vision=vision)
    eng._guard = FakeGuard(grade)
    return eng


def test_chunk_text_handles_blocks():
    assert chunk_text("abc") == "abc"
    assert chunk_text([{"type": "text", "text": "a"}, {"type": "reasoning", "reasoning": "x"}, "b"]) == "ab"
    assert chunk_text(None) == ""


def test_format_context():
    assert format_context(DOCS).startswith(">> [Source: UTS_C1.tex]\nUn processus")


async def test_ask_streams_answer_with_relevant_docs():
    eng = _engine("Réponse [Source: UTS_C1.tex]", GradeResult(True, True, "ok", "answer_rag"))
    out = "".join([t async for t in eng.ask("Qu'est-ce que la stationnarité ?")])
    assert out == "Réponse [Source: UTS_C1.tex]"
    assert eng.retriever.queries == ["Qu'est-ce que la stationnarité ?"]


async def test_ask_refuses_out_of_domain():
    eng = _engine("never used", GradeResult(False, False, "hors domaine", "refuse_domain"))
    out = "".join([t async for t in eng.ask("Qui a gagné la coupe du monde ?")])
    assert "Séries Temporelles" in out and "hors domaine" in out


async def test_ask_routes_images_to_vision_model(tmp_path: Path):
    img = tmp_path / "g.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    eng = _engine("text answer", GradeResult(True, True, "ok", "answer_rag"), images_model="je vois un graphique")
    out = "".join([t async for t in eng.ask("Explique", images=[(str(img), "image/png")])])
    assert out == "je vois un graphique"


async def test_quiz_parses_json_fences():
    quiz = [{"question": "Q?", "options": ["A) a", "B) b"], "correct_index": 1, "explanation": "e"}]
    eng = _engine("```json\n" + json.dumps(quiz) + "\n```", GradeResult(True, True, "ok", "answer_rag"))
    assert await eng.generate_quiz("stationnarité") == quiz


async def test_quiz_out_of_domain_and_no_docs():
    eng = _engine("x", GradeResult(False, False, "non", "refuse_domain"))
    assert (await eng.generate_quiz("foot"))[0]["error"] == "domain"
    eng2 = _engine("x", GradeResult(False, True, "rien", "retry_query"))
    assert await eng2.generate_quiz("x") == []


async def test_generate_code():
    eng = _engine("```python\nprint(1)\n```", GradeResult(True, True, "ok", "answer_rag"))
    assert "print(1)" in await eng.generate_code("AR(1)", "Python")
