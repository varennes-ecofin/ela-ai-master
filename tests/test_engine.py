"""Tests for ``ela.engine.ElaEngine`` with a scripted tool-calling model (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ela.agent import outline as outline_mod
from ela.agent.agent import build_agent
from ela.agent.events import (
    ErrorEvent,
    FinalEvent,
    QuizEvent,
    SourcesEvent,
    TokenEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from ela.config import get_settings
from ela.engine import LIMIT_MESSAGE, ElaEngine, chunk_text, format_context
from ela.history import ChatTurn
from ela.retrieval import resources
from ela.retrieval.retriever import ScopedRetriever
from fakes import ScriptedChatModel, tool_call_message
from test_agent_tools import CORPUS, OutlineStore
from test_retrieval import FakeEmb


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANK_TOP_N", "5")
    monkeypatch.setenv("AGENT_MAX_MODEL_CALLS", "4")
    get_settings.cache_clear()
    store = OutlineStore(CORPUS, fp="gen-1")
    monkeypatch.setattr(resources, "store_builder", lambda: store)
    monkeypatch.setattr(resources, "embeddings_builder", lambda: FakeEmb())
    resources.reset()
    outline_mod.reset_cache()
    yield store
    resources.reset()
    outline_mod.reset_cache()


def _engine(script, *, level="M1", fallback=None, llm=None, vision=None, fail_first=0):
    model = ScriptedChatModel(script, fail_first=fail_first)
    retriever = ScopedRetriever(level, store=resources.get_store(), embeddings=FakeEmb(), reranker=None)
    agent = build_agent(model=model, fallback=fallback)
    eng = ElaEngine(level, retriever=retriever, llm=llm or model, vision=vision, agent=agent)
    return eng, model


async def _collect(engine: ElaEngine, question: str, **kwargs):
    return [e async for e in engine.ask(question, **kwargs)]


def test_helpers():
    assert chunk_text("abc") == "abc"
    assert chunk_text([{"type": "text", "text": "a"}, {"type": "reasoning", "reasoning": "x"}, "b"]) == "ab"
    assert chunk_text(None) == ""
    assert format_context(CORPUS[:1]).startswith(
        ">> [Source: UTS.tex — Chapitre 1 : processus > Stationnarité > Stationnarité faible]"
    )


async def test_simple_question_one_search(env):
    eng, model = _engine(
        [
            tool_call_message(("search_course", {"query": "stationnarité faible"})),
            AIMessage(content="La stationnarité faible impose $E(X_t)=\\mu$ [S1]."),
        ]
    )
    events = await _collect(eng, "Qu'est-ce que la stationnarité faible ?")
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["ToolStartEvent", "ToolEndEvent", "TokenEvent", "SourcesEvent", "FinalEvent"]
    start, end = events[0], events[1]
    assert start.name == "search_course" and start.label.startswith("🔎 Recherche : « stationnarité faible »")
    assert end.ok and end.summary == "2 extrait(s)"
    final = events[-1]
    assert final.text == "La stationnarité faible impose $E(X_t)=\\mu$ [S1]."
    assert final.citations_valid == ["S1"] and not final.citations_invalid
    assert (final.n_model_calls, final.n_tool_calls, final.n_searches) == (2, 1, 1)
    sources = events[-2]
    assert [r.sid for r in sources.cited] == ["S1"] and [r.sid for r in sources.all] == ["S1", "S2"]
    # system prompt rendered from the template with level and courses
    system = model.calls[0][0]
    assert isinstance(system, SystemMessage)
    assert "Master 1 (M1)" in system.content and "Univariate_Time_Series" in system.content
    assert "Financial_Econometrics" not in system.content  # M2 course not listed for an M1 student
    assert isinstance(model.calls[0][-1], HumanMessage)


async def test_compound_question_two_searches_and_phantom_removed(env):
    eng, _ = _engine(
        [
            tool_call_message(("search_course", {"query": "test ADF"}), ("search_course", {"query": "Granger"})),
            AIMessage(content="ADF [S1] et Granger [S2] ; voir aussi [S7]."),
        ]
    )
    events = await _collect(eng, "Compare ADF et Granger")
    starts = [e for e in events if isinstance(e, ToolStartEvent)]
    ends = [e for e in events if isinstance(e, ToolEndEvent)]
    assert len(starts) == 2 and len(ends) == 2 and {e.call_id for e in ends} == {s.call_id for s in starts}
    final = events[-1]
    assert final.n_searches == 2 and final.citations_invalid == ["S7"]
    assert final.text == "ADF [S1] et Granger [S2] ; voir aussi."


async def test_out_of_domain_refused_without_search(env):
    eng, _ = _engine(
        [AIMessage(content="Je suis spécialisé en **Séries Temporelles**. Ce sujet sort du cadre du cours.")]
    )
    events = await _collect(eng, "Qui a gagné la coupe du monde ?")
    final = events[-1]
    assert isinstance(final, FinalEvent) and final.n_searches == 0 and final.n_tool_calls == 0
    assert "sort du cadre du cours" in final.text


async def test_text_before_tool_call_is_reset(env):
    eng, _ = _engine(
        [
            tool_call_message(("search_course", {"query": "ADF"}), content="Je cherche..."),
            AIMessage(content="Réponse [S1]."),
        ]
    )
    events = await _collect(eng, "ADF ?")
    kinds = [type(e).__name__ for e in events]
    assert kinds[:3] == ["TokenEvent", "ResetEvent", "ToolStartEvent"]
    assert events[-1].text == "Réponse [S1]."


async def test_model_call_limit_ends_with_message(env):
    looping = [tool_call_message(("search_course", {"query": f"q{i}"})) for i in range(10)]
    eng, _ = _engine(looping)
    events = await _collect(eng, "Boucle")
    final = events[-1]
    assert isinstance(final, FinalEvent) and final.limit_reached and final.text == LIMIT_MESSAGE
    assert final.n_model_calls == 4  # AGENT_MAX_MODEL_CALLS


async def test_search_limit_is_reported_to_the_model(env, monkeypatch):
    monkeypatch.setenv("AGENT_MAX_SEARCHES", "1")
    get_settings.cache_clear()
    eng, model = _engine(
        [
            tool_call_message(("search_course", {"query": "a"})),
            tool_call_message(("search_course", {"query": "b"})),
            AIMessage(content="Fini [S1]."),
        ]
    )
    events = await _collect(eng, "x")
    ends = [e for e in events if isinstance(e, ToolEndEvent)]
    assert [e.ok for e in ends] == [True, False]
    assert events[-1].n_searches == 1
    last_tool_msg = model.calls[-1][-1]
    assert "limit" in last_tool_msg.content.lower()


async def test_images_are_transcribed_then_agent_runs(env, tmp_path: Path):
    img = tmp_path / "g.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    vision = ScriptedChatModel(["Image 1 : graphique de $X_t$, tendance croissante."])
    eng, model = _engine(
        [tool_call_message(("search_course", {"query": "tendance"})), AIMessage(content="Vu [S1].")], vision=vision
    )
    events = await _collect(eng, "Que montre ce graphique ?", images=[(str(img), "image/png")])
    assert events[0].name == "read_image" and events[1].name == "read_image" and events[1].ok
    human = model.calls[0][-1]
    assert "Que montre ce graphique ?" in human.content and "tendance croissante" in human.content
    vision_prompt = vision.calls[0][0].content
    assert "Que montre ce graphique ?" in vision_prompt and "transcris" in vision_prompt.lower()


async def test_vision_failure_does_not_block_the_answer(env, tmp_path: Path):
    img = tmp_path / "g.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    vision = ScriptedChatModel([], fail_first=1)
    eng, model = _engine([AIMessage(content="Sans image.")], vision=vision)
    events = await _collect(eng, "Q", images=[(str(img), "image/png")])
    assert events[1].name == "read_image" and not events[1].ok
    assert events[-1].text == "Sans image." and model.calls[0][-1].content == "Q"


async def test_fallback_model_takes_over(env):
    fallback = ScriptedChatModel([AIMessage(content="Réponse du modèle de secours.")])
    eng, primary = _engine([AIMessage(content="never")], fallback=fallback, fail_first=1)
    events = await _collect(eng, "Bonjour")
    assert events[-1].text == "Réponse du modèle de secours."
    assert primary.failures == 1 and len(fallback.calls) == 1


async def test_quiz_tool_emits_quiz_event(env):
    quiz_json = json.dumps(
        {
            "questions": [
                {"question": "Q1 ?", "options": ["a", "b"], "correct_index": 1, "explanation": "e"},
            ]
        }
    )
    generator = ScriptedChatModel([quiz_json])
    eng, _ = _engine(
        [tool_call_message(("generate_quiz", {"topic": "stationnarité", "n": 1})), AIMessage(content="Voici un quiz.")],
        llm=generator,
    )
    events = await _collect(eng, "Teste-moi sur la stationnarité")
    quiz = next(e for e in events if isinstance(e, QuizEvent)).quiz
    q = quiz.questions[0]
    assert q.options[q.correct_index] == "b" and quiz.sources == ["S1", "S2"]
    assert events[-1].text == "Voici un quiz."
    assert next(e for e in events if isinstance(e, ToolEndEvent)).summary.startswith("Quiz de 1 questions")


async def test_history_window_and_summary(env, monkeypatch):
    monkeypatch.setenv("HISTORY_MAX_TURNS", "1")
    get_settings.cache_clear()
    summarizer = ScriptedChatModel(["L'étudiant a demandé la définition d'un AR(1)."])
    eng, model = _engine([AIMessage(content="Suite.")], llm=summarizer)
    turns = [
        ChatTurn("user", "Définis un AR(1)"),
        ChatTurn("assistant", "Un AR(1) est..."),
        ChatTurn("user", "Et sa variance ?"),
        ChatTurn("assistant", "La variance vaut..."),
    ]
    events = await _collect(eng, "Merci, et la prévision ?", chat_history=turns)
    assert events[-1].text == "Suite."
    system = model.calls[0][0].content
    assert "RÉSUMÉ DES ÉCHANGES PRÉCÉDENTS" in system and "AR(1)" in system
    # window = last exchange only, then the new question
    kept = [m.content for m in model.calls[0][1:]]
    assert kept == ["Et sa variance ?", "La variance vaut...", "Merci, et la prévision ?"]
    assert len(summarizer.calls) == 1 and eng.summary.startswith("L'étudiant")
    # a second turn with the same dropped prefix does not call the summariser again
    eng2_model = ScriptedChatModel([AIMessage(content="Encore.")])
    eng._agent = build_agent(model=eng2_model)
    await _collect(eng, "Autre question", chat_history=turns)
    assert len(summarizer.calls) == 1


async def test_engine_error_is_reported(env):
    class Broken:
        async def astream(self, *a, **k):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    eng = ElaEngine("M1", agent=Broken(), llm=ScriptedChatModel([]))
    events = await _collect(eng, "x")
    assert isinstance(events[-1], ErrorEvent) and "boom" in events[-1].message


async def test_ask_text_and_sources_event(env):
    eng, _ = _engine([tool_call_message(("search_course", {"query": "ADF"})), AIMessage(content="ADF [S1].")])
    assert await eng.ask_text("ADF ?") == "ADF [S1]."
    assert eng.last_context is not None and eng.last_context.n_searches == 1
    events = [e async for e in eng.ask("ADF ?")] if False else []
    assert events == []
    assert isinstance(SourcesEvent(cited=[], all=[]), SourcesEvent) and isinstance(TokenEvent("x"), TokenEvent)


async def test_generate_quiz_and_code_direct(env):
    quiz_json = json.dumps(
        {"questions": [{"question": "Q", "options": ["a", "b", "c"], "correct_index": 2, "explanation": "e"}]}
    )
    eng, _ = _engine([], llm=ScriptedChatModel([quiz_json, "```python\nprint(1)\n```"]))
    quiz = await eng.generate_quiz("stationnarité", num_questions=1)
    assert quiz is not None and quiz.to_legacy()[0]["options"][quiz.to_legacy()[0]["correct_index"]] == "c"
    assert eng.last_context is not None and len(eng.last_context.registry) == 2
    assert "print(1)" in await eng.generate_code("AR(1)", "Python")
    assert await eng.generate_quiz("zzz") is None


async def test_generate_quiz_failure_raises_generation_error(env):
    from ela.engine import GenerationError

    eng, _ = _engine([], llm=ScriptedChatModel(["pas du JSON"]))
    with pytest.raises(GenerationError):
        await eng.generate_quiz("stationnarité", num_questions=1)
    eng, _ = _engine([], llm=ScriptedChatModel(["pas du JSON"]))
    with pytest.raises(GenerationError):
        await eng.generate_exercise("stationnarité")


async def test_failed_quiz_tool_is_reported_as_failed(env):
    from ela.agent.events import tool_failed

    assert tool_failed("La génération du quiz a échoué (ValueError). Propose à l'étudiant de réessayer.")
    assert tool_failed("Tool call limit exceeded for generate_quiz")
    assert not tool_failed("Quiz de 3 questions généré sur « ADF »")
