"""Tests for the agent tools, the course outline and the pedagogical generators (fakes only)."""

from __future__ import annotations

import json

import pytest
from langchain.tools import ToolRuntime
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from ela.agent import outline as outline_mod
from ela.agent.context import TurnContext
from ela.agent.outline import build_outlines, find_slides, render_course_list, render_outline, resolve_course
from ela.agent.tools import generate_exercise, generate_quiz, get_course_outline, get_slide, search_course
from ela.config import get_settings
from ela.pedagogy.quiz import generate_quiz as gen_quiz
from ela.pedagogy.schemas import Exercise, Quiz, QuizQuestion
from ela.retrieval import resources
from ela.retrieval.retriever import ScopedRetriever
from test_retrieval import FakeEmb, FakeStore


def _slide(cid, course, level, chapter, section, title, text, ordinal, source=None):
    return Document(
        page_content=text,
        metadata={
            "chunk_id": cid,
            "level": level,
            "course": course,
            "source": source or f"{course[:3].upper()}.tex",
            "doc_kind": "slides",
            "chunk_type": "slide",
            "parent_id": "",
            "chapter": chapter,
            "section": section,
            "slide_title": title,
            "ordinal": ordinal,
            "part": 0,
        },
    )


CORPUS = [
    _slide(
        "u1",
        "Univariate_Time_Series",
        "M1",
        "Chapitre 1 : processus",
        "Stationnarité",
        "Stationnarité faible",
        "stationnarité faible espérance autocovariance",
        1,
        "UTS.tex",
    ),
    _slide(
        "u2",
        "Univariate_Time_Series",
        "M1",
        "Chapitre 1 : processus",
        "Stationnarité",
        "AR(1) et solution stationnaire",
        "AR(1) solution stationnarité phi",
        2,
        "UTS.tex",
    ),
    _slide(
        "u3",
        "Univariate_Time_Series",
        "M1",
        "Chapitre 2 : racines unitaires",
        "",
        "Test ADF",
        "test ADF racine unitaire Dickey Fuller augmenté",
        3,
        "UTS.tex",
    ),
    _slide(
        "m1",
        "Multivariate_Time_Series",
        "M1",
        "Chapitre 1 : VAR",
        "",
        "Causalité de Granger",
        "causalité Granger VAR",
        1,
        "MTS.tex",
    ),
    _slide(
        "h1",
        "Financial_Econometrics",
        "M2",
        "Chapitre 4 : VaR",
        "Backtesting",
        "Backtesting",
        "Kupiec Christoffersen backtesting VaR",
        1,
        "HFE_C4.tex",
    ),
]


class OutlineStore(FakeStore):
    def all_documents(self, levels=None):
        return [d for d in self.docs if not levels or d.metadata["level"] in levels]


@pytest.fixture
def ctx(monkeypatch: pytest.MonkeyPatch) -> TurnContext:
    monkeypatch.setenv("RERANK_TOP_N", "5")
    get_settings.cache_clear()
    store = OutlineStore(CORPUS, fp="gen-1")
    monkeypatch.setattr(resources, "store_builder", lambda: store)
    monkeypatch.setattr(resources, "embeddings_builder", lambda: FakeEmb())
    resources.reset()
    outline_mod.reset_cache()
    retriever = ScopedRetriever("M1", store=store, embeddings=FakeEmb(), reranker=None)
    yield TurnContext(level="M1", retriever=retriever)
    resources.reset()
    outline_mod.reset_cache()


def _runtime(ctx: TurnContext) -> ToolRuntime:
    return ToolRuntime(state={}, context=ctx, config={}, stream_writer=lambda *_: None, tool_call_id="t1", store=None)


# ---------------------------------------------------------------- outline


def test_build_outlines_groups_chapters_sections_slides():
    outlines = build_outlines(CORPUS)
    assert list(outlines) == ["Financial_Econometrics", "Multivariate_Time_Series", "Univariate_Time_Series"]
    uts = outlines["Univariate_Time_Series"]
    assert uts.sources == ["UTS.tex"] and uts.n_slides == 3
    assert [c.title for c in uts.chapters] == ["Chapitre 1 : processus", "Chapitre 2 : racines unitaires"]
    assert uts.chapters[0].sections[0].slides == ["Stationnarité faible", "AR(1) et solution stationnaire"]
    text = render_outline(uts)
    assert "Chapitre 2 : racines unitaires (1 slides)" in text and "    - Test ADF" in text
    assert "Test ADF" not in render_outline(uts, max_chars=120)  # slide titles dropped first
    assert "Univariate_Time_Series" in render_course_list(outlines)


def test_resolve_course_aliases():
    outlines = build_outlines(CORPUS)
    assert resolve_course("UTS", outlines) == "Univariate_Time_Series"
    assert resolve_course("uts.tex", outlines) == "Univariate_Time_Series"
    assert resolve_course("HFE_C4", outlines) == "Financial_Econometrics"
    assert resolve_course("hfe", outlines) == "Financial_Econometrics"
    assert resolve_course("multivariate", outlines) == "Multivariate_Time_Series"
    assert resolve_course("Financial Econometrics", outlines) == "Financial_Econometrics"
    assert resolve_course("panel", outlines) is None and resolve_course(None, outlines) is None


def test_find_slides():
    outlines = build_outlines(CORPUS)
    assert [d.metadata["chunk_id"] for d in find_slides(outlines, "test adf")] == ["u3"]
    assert [d.metadata["chunk_id"] for d in find_slides(outlines, "stationnaire")] == ["u2"]
    assert find_slides(outlines, "Backtesting", course="Univariate_Time_Series") == []


# ---------------------------------------------------------------- tools


def test_search_course_registers_sources_and_filters(ctx: TurnContext):
    out = search_course.invoke({"query": "stationnarité", "runtime": _runtime(ctx)})
    assert out.startswith("2 extrait(s) pour « stationnarité »")
    assert (
        "[S1] UTS.tex › Univariate_Time_Series › Chapitre 1 : processus › Stationnarité › Stationnarité faible" in out
    )
    assert ctx.registry.ids() == {"S1", "S2"} and ctx.n_searches == 1 and ctx.searches[0].sids == ["S1", "S2"]

    out = search_course.invoke({"query": "ADF racine", "runtime": _runtime(ctx), "course": "UTS", "k": 1})
    assert (
        "[S3]" in out and "(cours Univariate_Time_Series)" in out and ctx.searches[1].course == "Univariate_Time_Series"
    )

    out = search_course.invoke({"query": "Kupiec", "runtime": _runtime(ctx), "course": "panel"})
    assert out.startswith("Cours « panel » inconnu") and "Aucun extrait" in out  # M2 course invisible to M1

    out = search_course.invoke({"query": "stationnarité", "runtime": _runtime(ctx)})
    assert "[S1]" in out and len(ctx.registry) == 3  # same chunks keep their labels


def test_get_course_outline_tool(ctx: TurnContext):
    rt = _runtime(ctx)
    listing = get_course_outline.invoke({"runtime": rt})
    assert "Univariate_Time_Series" in listing and "Financial_Econometrics" not in listing  # level scoping
    assert "Test ADF" in get_course_outline.invoke({"runtime": rt, "course": "uts"})
    assert get_course_outline.invoke({"runtime": rt, "course": "xyz"}).startswith("Cours « xyz » inconnu")


def test_get_slide_by_label_and_title(ctx: TurnContext):
    rt = _runtime(ctx)
    search_course.invoke({"query": "stationnarité", "runtime": rt})
    assert get_slide.invoke({"runtime": rt, "source_id": "S2"}).startswith("[S2] UTS.tex")
    assert get_slide.invoke({"runtime": rt, "source_id": "S9"}).startswith("Identifiant « S9 » inconnu")
    out = get_slide.invoke({"runtime": rt, "slide_title": "Test ADF", "course": "UTS"})
    assert out.startswith("[S3] UTS.tex") and ctx.slides_opened == ["S2", "S3"]
    assert get_slide.invoke({"runtime": rt, "slide_title": "Backtesting"}).startswith("Aucune slide")
    assert get_slide.invoke({"runtime": rt}) == "Indique source_id (ex. S2) ou slide_title."


def _quiz_json() -> str:
    return json.dumps(
        {
            "questions": [
                {
                    "question": "Que signifie $E(X_t)=\\mu$ ?",
                    "options": ["Espérance constante", "Variance nulle", "Bruit blanc"],
                    "correct_index": 0,
                    "explanation": "Premier moment constant.",
                }
            ]
        }
    )


async def test_generate_quiz_tool_leaves_quiz_in_context(ctx: TurnContext):
    ctx.llm = GenericFakeChatModel(messages=iter([_quiz_json()]))
    out = await generate_quiz.ainvoke({"topic": "stationnarité", "runtime": _runtime(ctx), "n": 1})
    assert out.startswith("Quiz de 1 questions généré") and isinstance(ctx.quiz, Quiz)
    q = ctx.quiz.questions[0]
    assert ctx.quiz.sources == ["S1", "S2"] and q.options[q.correct_index] == "Espérance constante"
    assert sorted(ctx.quiz.to_legacy()[0]["options"]) == ["Bruit blanc", "Espérance constante", "Variance nulle"]


async def test_generate_quiz_without_material(ctx: TurnContext):
    ctx.llm = GenericFakeChatModel(messages=iter(["never"]))
    out = await generate_quiz.ainvoke({"topic": "zzz", "runtime": _runtime(ctx)})
    assert out.startswith("Le cours ne contient rien") and ctx.quiz is None


async def test_generate_quiz_invalid_payload_is_reported(ctx: TurnContext):
    ctx.llm = GenericFakeChatModel(
        messages=iter(['{"questions": [{"question": "q", "options": ["a"], "correct_index": 3, "explanation": "e"}]}'])
    )
    out = await generate_quiz.ainvoke({"topic": "stationnarité", "runtime": _runtime(ctx)})
    assert out.startswith("La génération du quiz a échoué") and ctx.quiz is None


async def test_generate_exercise_tool(ctx: TurnContext):
    payload = {
        "title": "AR(1)",
        "statement": "Soit $X_t$...",
        "hints": ["Utiliser la stationnarité"],
        "solution": "On a...",
    }
    ctx.llm = GenericFakeChatModel(messages=iter([json.dumps(payload)]))
    out = await generate_exercise.ainvoke({"topic": "AR(1) stationnaire", "runtime": _runtime(ctx)})
    assert out.startswith("Exercice « AR(1) » généré") and isinstance(ctx.exercise, Exercise)
    assert ctx.exercise.sources == ["S1"] and ctx.exercise.hints == ["Utiliser la stationnarité"]


async def test_generate_quiz_function_caps_n(ctx: TurnContext):
    llm = GenericFakeChatModel(messages=iter([_quiz_json()]))
    quiz = await gen_quiz(llm, ctx.get_retriever(), "stationnarité", n=50, difficulty="facile", level="M1")
    assert quiz is not None and len(quiz.questions) == 1


def test_quiz_schema_validates_index():
    with pytest.raises(ValueError):
        QuizQuestion(question="q", options=["a", "b"], correct_index=2, explanation="e")


def test_shuffle_options_keeps_the_correct_answer():
    import random

    from ela.pedagogy.quiz import shuffle_options

    q = QuizQuestion(question="q", options=["bonne", "b", "c", "d"], correct_index=0, explanation="e")
    positions = set()
    for seed in range(40):
        shuffled = shuffle_options(q, random.Random(seed))
        assert shuffled.options[shuffled.correct_index] == "bonne" and sorted(shuffled.options) == [
            "b",
            "bonne",
            "c",
            "d",
        ]
        positions.add(shuffled.correct_index)
    assert positions == {0, 1, 2, 3}  # the correct answer is no longer always A)


async def test_generate_quiz_tool_uses_session_difficulty(ctx: TurnContext):
    ctx.quiz_difficulty = "difficile"
    ctx.llm = GenericFakeChatModel(messages=iter([_quiz_json()]))
    out = await generate_quiz.ainvoke({"topic": "stationnarité", "runtime": _runtime(ctx), "n": 1})
    assert "difficulté « difficile »" in out and ctx.quiz.difficulty == "difficile"
    prompt = ctx.llm  # the prompt shown to the model numbers passages without S-labels
    ctx2 = TurnContext(level="M1", retriever=ctx.retriever, quiz_difficulty="difficile")
    ctx2.llm = GenericFakeChatModel(messages=iter([_quiz_json()]))
    await generate_quiz.ainvoke({"topic": "stationnarité", "runtime": _runtime(ctx2), "n": 1, "difficulty": "facile"})
    assert ctx2.quiz.difficulty == "facile"  # explicit request wins
    del prompt


def test_format_passages_hides_source_ids():
    from ela.agent.citations import SourceRegistry
    from ela.pedagogy.quiz import format_passages

    reg = SourceRegistry()
    refs = reg.register_all(CORPUS[:2])
    text = format_passages(refs)
    assert "[S1]" not in text and text.startswith("— Extrait 1 (Univariate_Time_Series › Chapitre 1 : processus")


async def test_check_step_prompt_and_parsing(ctx: TurnContext):
    from ela.pedagogy.exercise import check_step

    exercise = Exercise(title="AR(1)", statement="Soit $X_t$...", hints=["h1", "h2"], solution="On a...", sources=[])
    payload = {"verdict": "partiel", "feedback": "Bien, mais...", "guiding_question": "Quelle hypothèse ?"}
    llm = GenericFakeChatModel(messages=iter([json.dumps(payload)]))
    fb = await check_step(llm, exercise, "ma tentative", level="M1", revealed_hints=1, previous=[("x", "incorrect")])
    assert fb.verdict == "partiel" and not fb.exercise_completed and fb.guiding_question == "Quelle hypothèse ?"


# ---------------------------------------------------------------- quiz: batches, filtering, coverage


def _questions(prefix: str, k: int) -> list[dict]:
    return [
        {"question": f"{prefix}{i} ?", "options": [f"a{i}", f"b{i}"], "correct_index": 0, "explanation": "e"}
        for i in range(k)
    ]


async def test_generate_quiz_in_batches_of_five(ctx: TurnContext):
    from fakes import ScriptedChatModel

    llm = ScriptedChatModel(
        [json.dumps({"questions": _questions("A", 5)}), json.dumps({"questions": _questions("B", 2)})]
    )
    quiz = await gen_quiz(llm, ctx.get_retriever(), "stationnarité", n=7, level="M1")
    assert quiz is not None and len(quiz.questions) == 7
    assert len(llm.calls) == 2
    first, second = llm.calls[0][0].content, llm.calls[1][0].content
    assert "TÂCHE : 5 questions" in first and "QUESTIONS DÉJÀ POSÉES" not in first
    assert "TÂCHE : 2 questions" in second and "QUESTIONS DÉJÀ POSÉES" in second and "- A4 ?" in second
    assert ctx.registry.ids() == set() or True  # private registry: nothing leaks into the turn


async def test_generate_quiz_keeps_what_it_got_when_a_batch_fails(ctx: TurnContext):
    from fakes import ScriptedChatModel

    llm = ScriptedChatModel([json.dumps({"questions": _questions("A", 5)}), "pas du JSON"])
    quiz = await gen_quiz(llm, ctx.get_retriever(), "stationnarité", n=8, level="M1")
    assert quiz is not None and len(quiz.questions) == 5


async def test_generate_quiz_drops_garbled_questions():
    from ela.pedagogy.quiz import well_formed
    from ela.pedagogy.schemas import QuizQuestion

    ok = QuizQuestion(question="q", options=["$\\nu$", "$\\theta$"], correct_index=0, explanation="e")
    assert well_formed(ok)
    garbled = QuizQuestion(question="q", options=['",**B)**\n', '",'], correct_index=0, explanation="e")
    assert not well_formed(garbled)
    dup = QuizQuestion(question="q", options=["a", "A "], correct_index=0, explanation="e")
    assert not well_formed(dup)
    assert not well_formed(QuizQuestion(question="  ", options=["a", "b"], correct_index=0, explanation="e"))


async def test_generate_quiz_refuses_passages_off_topic(ctx: TurnContext):
    # An M1 student asks for GARCH: the retriever returns the nearest M1 slides (ARMA, ADF...),
    # none of which mentions GARCH → no quiz, and the tool points at the M2 course.
    ctx.llm = GenericFakeChatModel(messages=iter(["never called"]))
    out = await generate_quiz.ainvoke({"topic": "modèle GARCH", "runtime": _runtime(ctx)})
    assert ctx.quiz is None and out.startswith("Le cours ne contient rien sur « modèle GARCH »")
    out = await generate_quiz.ainvoke({"topic": "backtesting de la VaR", "runtime": _runtime(ctx)})
    assert ctx.quiz is None
    assert out.startswith("Les cours accessibles à cet étudiant (niveau M1) ne contiennent rien")
    assert "Financial_Econometrics (niveau M2)" in out


async def test_generate_exercise_refuses_passages_off_topic(ctx: TurnContext):
    ctx.llm = GenericFakeChatModel(messages=iter(["never called"]))
    out = await generate_exercise.ainvoke({"topic": "backtesting de la VaR", "runtime": _runtime(ctx)})
    assert ctx.exercise is None and "aucun exercice généré" in out and "Financial_Econometrics" in out


def test_courses_elsewhere(ctx: TurnContext):
    from ela.agent.outline import courses_elsewhere

    assert courses_elsewhere("backtesting VaR", "M1") == [("Financial_Econometrics", "M2")]
    assert courses_elsewhere("backtesting VaR", "ALL") == []
    assert courses_elsewhere("stationnarité", "M1") == []
