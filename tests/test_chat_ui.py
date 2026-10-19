"""Tests for the Chainlit rendering layer with a fake ``chainlit`` module."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from ela.agent.citations import SourceRef
from ela.agent.events import (
    ErrorEvent,
    ExerciseEvent,
    FinalEvent,
    QuizEvent,
    ResetEvent,
    SourcesEvent,
    TokenEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from ela.pedagogy.schemas import Exercise, Quiz, QuizQuestion

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "chat"))


class _Sent:
    """Recorder shared by the fake Message / Step / Action classes."""

    log: list[tuple[str, object]] = []


class FakeMessage:
    def __init__(self, content="", author=None, actions=None, elements=None, **_):
        self.content = content
        self.author = author
        self.actions = actions or []
        self.elements = elements or []

    async def send(self):
        _Sent.log.append(("message.send", self))
        return self

    async def update(self):
        _Sent.log.append(("message.update", self))

    async def stream_token(self, token):
        self.content += token


class FakeStep:
    def __init__(self, name="", type="undefined", show_input="json", **_):
        self.name = name
        self.type = type
        self.output = None
        self.start = self.end = None

    async def send(self):
        _Sent.log.append(("step.send", self))

    async def update(self):
        _Sent.log.append(("step.update", self))


class FakeText:
    def __init__(self, name="", content="", display="inline", **_):
        self.name, self.content, self.display = name, content, display


class FakeFile:
    def __init__(self, name="", content=b"", mime="", display="inline", **_):
        self.name, self.content, self.mime, self.display = name, content, mime, display


class FakeImage:
    def __init__(self, name="", content=b"", display="inline", size="medium", **_):
        self.name, self.content, self.display, self.size = name, content, display, size


class FakeAction:
    def __init__(self, name="", payload=None, label="", tooltip=None, **_):
        self.name, self.payload, self.label = name, payload or {}, label

    async def remove(self):
        _Sent.log.append(("action.remove", self))


class FakeUserSession:
    def __init__(self):
        self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value


@pytest.fixture
def ui(monkeypatch: pytest.MonkeyPatch):
    fake_cl = types.SimpleNamespace(
        Message=FakeMessage,
        Step=FakeStep,
        Text=FakeText,
        File=FakeFile,
        Image=FakeImage,
        Action=FakeAction,
        user_session=FakeUserSession(),
    )
    monkeypatch.setitem(sys.modules, "chainlit", fake_cl)
    sys.modules.pop("ui", None)
    import ui as ui_mod

    _Sent.log = []
    ui_mod.RECORDED = {"questions": [], "answers": [], "attempts": []}

    def _recorder(bucket):
        async def _rec(item):
            ui_mod.RECORDED[bucket].append(item)
            return True

        return _rec

    monkeypatch.setattr(ui_mod, "record_question", _recorder("questions"))
    monkeypatch.setattr(ui_mod, "record_answer", _recorder("answers"))
    monkeypatch.setattr(ui_mod, "record_attempt", _recorder("attempts"))
    yield ui_mod
    sys.modules.pop("ui", None)


class FakeEngine:
    def __init__(self, events):
        self.events = events

    async def ask(self, question, chat_history=None, images=None, session_id=None, user_id=None, **kw):
        for e in self.events:
            yield e


def _ref(sid="S1"):
    return SourceRef(
        sid, "c1", "UTS.tex", "UTS", "M1", "Chapitre 2", "", "Test ADF", "contenu \\(x\\)", 0.03, None, "search", "adf"
    )


async def test_run_turn_renders_steps_tokens_reset_and_sources(ui):
    events = [
        TokenEvent("Je cherche"),
        ResetEvent(),
        ToolStartEvent("c1", "search_course", {"query": "ADF"}, "🔎 Recherche : « ADF »"),
        ToolEndEvent("c1", "search_course", "2 extrait(s)"),
        TokenEvent("Le test ADF "),
        TokenEvent("[S1]."),
        SourcesEvent(cited=[_ref()], all=[_ref(), _ref("S2")]),
        FinalEvent(text="Le test ADF [S1].", citations_valid=["S1"]),
    ]
    outcome = await ui.run_turn(FakeEngine(events), "ADF ?", [], [], session_id="t", user_id="u")
    assert isinstance(outcome, FinalEvent)
    msg = next(obj for kind, obj in _Sent.log if kind == "message.send")
    assert msg.content == "Le test ADF [S1]."
    assert [e.name for e in msg.elements] == ["S1"] and msg.elements[0].display == "side"
    assert "**UTS › Chapitre 2 › Test ADF**" in msg.elements[0].content and "$x$" in msg.elements[0].content
    step = next(obj for kind, obj in _Sent.log if kind == "step.send")
    assert step.name == "🔎 Recherche : « ADF »" and step.output == "2 extrait(s)" and step.end is not None
    kinds = [k for k, _ in _Sent.log]
    assert kinds == [
        "message.send",
        "message.update",
        "step.send",
        "step.update",
        "message.update",
        "message.update",
    ]  # reset, step, final, follow-up buttons
    assert [a.label for a in msg.actions] == ["🎓 Me tester : adf", "✏️ Exercice : adf", "📄 Fiche : Chapitre 2"]
    assert msg.actions[0].payload == {"kind": "quiz", "topic": "adf", "course": "UTS", "chapter": "Chapitre 2"}


async def test_run_turn_error_and_unfinished_step(ui):
    events = [ToolStartEvent("c1", "search_course", {"query": "x"}, "🔎 x"), ErrorEvent("❌ Erreur ELA : boom")]
    outcome = await ui.run_turn(FakeEngine(events), "x", [], [], session_id=None, user_id=None)
    assert isinstance(outcome, ErrorEvent)
    step = next(obj for kind, obj in _Sent.log if kind == "step.send")
    assert step.output.startswith("⚠️")
    msg = next(obj for kind, obj in _Sent.log if kind == "message.send")
    assert msg.content.startswith("❌")


async def test_quiz_flow(ui, monkeypatch):
    recorded = []

    async def fake_record(answer):
        recorded.append(answer)
        return True

    monkeypatch.setattr(ui, "record_answer", fake_record)
    quiz = Quiz(
        topic="t",
        questions=[
            QuizQuestion(question="Q1 $x$ ?", options=["a", "b"], correct_index=1, explanation="e1"),
            QuizQuestion(question="Q2 ?", options=["c", "d", "e"], correct_index=0, explanation="e2"),
        ],
        sources=["S1"],
        source_labels=["UTS › Chap 1 › Stationnarité"],
        course="UTS",
        difficulty="moyen",
    )
    await ui.run_turn(
        FakeEngine([QuizEvent(quiz), FinalEvent(text="Voici un quiz.")]),
        "quiz",
        [],
        [],
        session_id="t1",
        user_id="alice",
    )
    session = ui.cl.user_session
    assert session.get("quiz_mode") == "active" and session.get("quiz_index") == 0
    q1 = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert q1.content.startswith("🎓 **Quiz : t** — 2 question(s), difficulté « moyen »\n\n✅ **Question 1/2**")
    assert [a.label for a in q1.actions] == ["A", "B"]
    assert "**A)** a" in q1.content and "**B)** b" in q1.content and "Q1 $x$ ?" in q1.content

    await ui.answer_quiz(FakeAction(name="quiz_answer", payload={"value": "1"}))
    sent = [obj for kind, obj in _Sent.log if kind == "message.send"]
    assert sent[-2].content.startswith("✅ **Correct !**") and sent[-1].content.startswith("✅ **Question 2/2**")
    assert sent[-2].actions == []
    await ui.answer_quiz(FakeAction(name="quiz_answer", payload={"value": "2"}))
    sent = [obj for kind, obj in _Sent.log if kind == "message.send"]
    assert sent[-2].content.startswith("❌ **Incorrect.**") and "**A** : c" in sent[-2].content
    assert [a.name for a in sent[-2].actions] == ["quiz_explain"] and sent[-2].actions[0].payload == {"index": 1}
    final = sent[-1]
    assert final.content.startswith("🏆") and "1/2" in final.content and session.get("quiz_mode") is None
    assert [a.name for a in final.actions] == ["quiz_retry_errors"]
    assert session.get("quiz_difficulty") == "moyen"  # 50 %: unchanged

    # persistence: one row per answer, with the student, the quiz id and the sources
    assert len(recorded) == 2
    assert recorded[0].user_identifier == "alice" and recorded[0].thread_id == "t1" and recorded[0].correct
    assert recorded[1].question_index == 1 and recorded[1].answer_index == 2 and not recorded[1].correct
    assert recorded[1].quiz_id == quiz.quiz_id and recorded[1].sources == ["UTS › Chap 1 › Stationnarité"]
    assert recorded[1].course == "UTS" and recorded[1].difficulty == "moyen"

    # explanation prompt for the failed question
    prompt = ui.explain_prompt(1)
    assert "« Q2 ? »" in prompt and "« c »" in prompt and "quiz sur « t »" in prompt
    assert ui.explain_prompt(9) is None


async def test_quiz_explain_runs_a_turn(ui):
    ui.cl.user_session.set(
        "quiz", {"topic": "t", "questions": [{"question": "Q", "options": ["a", "b"], "correct_index": 0}]}
    )
    seen = {}

    class Eng:
        async def ask(self, question, chat_history=None, images=None, session_id=None, user_id=None, **kw):
            seen["q"] = question
            yield FinalEvent(text="Parce que.")

    await ui.explain_question(
        FakeAction(name="quiz_explain", payload={"index": 0}), Eng(), [], session_id="t", user_id="u"
    )
    assert (
        "« Q »" in seen["q"] and [obj for kind, obj in _Sent.log if kind == "message.send"][-1].content == "Parce que."
    )


async def test_quiz_retry_targets_errors_and_adapts_difficulty(ui, monkeypatch):
    async def fake_record(answer):
        return True

    monkeypatch.setattr(ui, "record_answer", fake_record)
    quiz = Quiz(
        topic="ADF",
        questions=[
            QuizQuestion(question=f"Q{i} ?", options=["a", "b"], correct_index=0, explanation="e") for i in range(3)
        ],
        difficulty="moyen",
    )
    await ui.start_quiz(quiz, user_id="alice", thread_id="t")
    for _ in range(3):  # three wrong answers → easier next time
        await ui.answer_quiz(FakeAction(name="quiz_answer", payload={"value": "1"}))
    assert ui.cl.user_session.get("quiz_difficulty") == "facile" and ui.cl.user_session.get("quiz_errors") == [0, 1, 2]
    assert ui.current_difficulty() == "facile"

    calls = {}

    class Eng:
        async def generate_quiz(self, topic, num_questions=3, difficulty="moyen", *, focus=None, **kw):
            calls.update(topic=topic, n=num_questions, difficulty=difficulty, focus=focus)
            return Quiz(
                topic=topic,
                questions=[QuizQuestion(question="R ?", options=["a", "b"], correct_index=0, explanation="e")],
                difficulty=difficulty,
            )

    await ui.retry_errors(FakeAction(name="quiz_retry_errors"), Eng(), user_id="alice", thread_id="t")
    assert calls == {"topic": "ADF", "n": 3, "difficulty": "facile", "focus": ["Q0 ?", "Q1 ?", "Q2 ?"]}
    assert ui.cl.user_session.get("quiz_mode") == "active" and ui.cl.user_session.get("quiz_index") == 0
    last = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert "✅ **Question 1/1**" in last.content


async def test_exercise_flow(ui, monkeypatch):
    recorded = []

    async def fake_record(attempt):
        recorded.append(attempt)
        return True

    monkeypatch.setattr(ui, "record_attempt", fake_record)
    exercise = Exercise(
        topic="AR(1)", title="AR(1)", statement="Soit $X_t$.", hints=["h1", "h2"], solution="sol", sources=["S1"]
    )
    await ui.run_turn(
        FakeEngine([ExerciseEvent(exercise), FinalEvent(text="Voici.")]), "ex", [], [], session_id="t", user_id="alice"
    )
    shown = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert shown.content.startswith("✏️ **Exercice : AR(1)**")
    assert [a.name for a in shown.actions] == ["exercise_step", "exercise_hint", "exercise_solution"]
    assert "2 restant(s)" in shown.actions[1].label

    await ui.reveal_hint(FakeAction(name="exercise_hint"))
    hint = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert hint.content.startswith("💡 **Indice 1/2**") and "1 restant(s)" in hint.actions[1].label

    # Socratic step: button → next message is the attempt
    await ui.ask_for_step(FakeAction(name="exercise_step"))
    assert ui.cl.user_session.get("exercise_mode") == "awaiting_step"

    class Eng:
        calls = []

        async def check_step(self, exercise, attempt, *, revealed_hints=0, previous=None):
            Eng.calls.append((attempt, revealed_hints, previous))
            from ela.pedagogy.schemas import StepFeedback

            if attempt == "fin":
                return StepFeedback(verdict="correct", feedback="Parfait.", exercise_completed=True)
            return StepFeedback(
                verdict="partiel", feedback="Bon début, revois le signe.", guiding_question="Et $\\phi$ ?"
            )

    await ui.check_student_step(Eng(), "X_t = phi X_{t-1}")
    assert ui.cl.user_session.get("exercise_mode") is None
    fb = [obj for kind, obj in _Sent.log if kind == "message.update"][-1]
    assert (
        fb.content.startswith("**🟡 Presque**") and "revois le signe" in fb.content and "➡️ Et $\\phi$ ?" in fb.content
    )
    assert [a.name for a in fb.actions] == ["exercise_step", "exercise_hint", "exercise_solution"]
    assert Eng.calls[-1] == ("X_t = phi X_{t-1}", 1, [])
    assert recorded[-1].verdict == "partiel" and recorded[-1].attempt_index == 1 and recorded[-1].hints_revealed == 1
    assert recorded[-1].user_identifier == "alice" and recorded[-1].exercise_id == exercise.exercise_id

    # "stop" leaves the exercise open without calling the model
    await ui.ask_for_step(FakeAction(name="exercise_step"))
    await ui.check_student_step(Eng(), "stop")
    assert len(Eng.calls) == 1 and ui.cl.user_session.get("exercise") is not None

    # completion closes the exercise (only the solution button remains)
    await ui.ask_for_step(FakeAction(name="exercise_step"))
    await ui.check_student_step(Eng(), "fin")
    assert Eng.calls[-1][2] == [("X_t = phi X_{t-1}", "partiel")]
    fb = [obj for kind, obj in _Sent.log if kind == "message.update"][-1]
    assert "🎉" in fb.content and [a.name for a in fb.actions] == ["exercise_solution"]
    assert ui.cl.user_session.get("exercise_completed") and recorded[-1].completed

    await ui.reveal_solution(FakeAction(name="exercise_solution"))
    sol = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert sol.content.startswith("📝 **Corrigé**")


async def test_check_step_failure_is_reported(ui):
    ui.cl.user_session.set("exercise", Exercise(title="t", statement="s", hints=["h"], solution="x").model_dump())
    ui.cl.user_session.set("exercise_hint_index", 0)

    class Eng:
        async def check_step(self, *a, **k):
            return None

    await ui.check_student_step(Eng(), "tentative")
    fb = [obj for kind, obj in _Sent.log if kind == "message.update"][-1]
    assert fb.content.startswith("⚠️") and [a.name for a in fb.actions] == [
        "exercise_step",
        "exercise_hint",
        "exercise_solution",
    ]


def test_strip_context_header(ui):
    raw = "COURS: UTS (M1)\nCHAPITRE: Chap 2\nSECTION: ARIMAX\nSLIDE: Identification\nCONTENU:\n- Il est possible $x$"
    assert ui.strip_context_header(raw) == "- Il est possible $x$"
    assert ui.strip_context_header("pas d'en-tête") == "pas d'en-tête"


def test_parse_quiz_request(ui):
    assert ui.parse_quiz_request("test ADF, 5 questions") == ("test ADF", 5)
    assert ui.parse_quiz_request("10 questions sur la cointégration") == ("sur la cointégration", 10)
    assert ui.parse_quiz_request("GARCH") == ("GARCH", 3)
    assert ui.parse_quiz_request("VAR 50 questions") == ("VAR", 10)


async def test_run_turn_passes_session_difficulty(ui):
    seen = {}

    class Eng:
        async def ask(
            self, question, chat_history=None, images=None, session_id=None, user_id=None, quiz_difficulty=None
        ):
            seen["d"] = quiz_difficulty
            yield FinalEvent(text="ok")

    ui.cl.user_session.set("quiz_difficulty", "difficile")
    await ui.run_turn(Eng(), "quiz ?", [], [], session_id=None, user_id=None)
    assert seen["d"] == "difficile"


async def test_run_turn_records_activity(ui):
    ref = _ref()
    events = [
        ToolStartEvent("c1", "search_course", {"query": "ADF"}, "🔎"),
        ToolEndEvent("c1", "search_course", "1 extrait(s)"),
        SourcesEvent(cited=[ref], all=[ref]),
        FinalEvent(text="Le test ADF [S1]. [Connaissances générales]", citations_valid=["S1"], n_searches=1),
    ]
    await ui.run_turn(FakeEngine(events), "Explique ADF", [], [], session_id="t1", user_id="alice")
    rec = ui.RECORDED["questions"][-1]
    assert rec.user_identifier == "alice" and rec.course == "UTS" and rec.chapter == "Chapitre 2"
    assert rec.n_searches == 1 and rec.n_sources == 1 and rec.general_knowledge and not rec.refused
    # anonymous session: nothing recorded
    n = len(ui.RECORDED["questions"])
    await ui.run_turn(FakeEngine(events), "Explique ADF", [], [], session_id="t1", user_id=None)
    assert len(ui.RECORDED["questions"]) == n


async def test_show_report_and_follow_suggestion(ui, monkeypatch):
    from ela.db.progress import StudentReport, Suggestion
    from ela.db.quiz import QuizStats

    report = StudentReport(
        identifier="alice",
        activity={"questions": 4, "refused": 0, "general": 1, "threads": 2},
        chapters=[{"course": "UTS", "chapter": "Chapitre 2", "questions": 3, "last_seen": None}],
        quiz=QuizStats(6, 3, 2, [{"topic": "ADF", "n": 4, "correct": 1, "rate": 0.25}]),
        exercises=[{"exercise_id": "e", "topic": "AR(1)", "title": "AR(1)", "attempts": 2, "completed": False}],
        suggestions=[
            Suggestion("quiz", "ADF", "🎓 Quiz : ADF"),
            Suggestion("exercise", "AR(1)", "✏️ Reprendre : AR(1)"),
        ],
    )

    async def fake_report(identifier):
        return report

    monkeypatch.setattr(ui, "student_report", fake_report)
    await ui.show_report("alice")
    msg = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert msg.content.startswith("📈 **Mon bilan**") and "Points faibles : ADF (25 %)" in msg.content
    assert "UTS › Chapitre 2 — 3 question(s)" in msg.content and "⏳ 2 tentative(s)" in msg.content
    assert [a.payload for a in msg.actions] == [
        {"kind": "quiz", "topic": "ADF"},
        {"kind": "exercise", "topic": "AR(1)"},
    ]

    class Eng:
        async def generate_quiz(self, topic, num_questions=3, difficulty="moyen", *, focus=None, **kw):
            return Quiz(
                topic=topic,
                questions=[QuizQuestion(question="Q", options=["a", "b"], correct_index=0, explanation="e")],
            )

        async def generate_exercise(self, topic):
            return Exercise(topic=topic, title=topic, statement="s", hints=["h"], solution="x")

    await ui.follow_suggestion(
        FakeAction(name="report_suggestion", payload={"kind": "quiz", "topic": "ADF"}),
        Eng(),
        user_id="alice",
        thread_id="t",
    )
    assert ui.cl.user_session.get("quiz_mode") == "active"
    await ui.follow_suggestion(
        FakeAction(name="report_suggestion", payload={"kind": "exercise", "topic": "AR(1)"}),
        Eng(),
        user_id="alice",
        thread_id="t",
    )
    assert ui.cl.user_session.get("exercise")["topic"] == "AR(1)"
    await ui.show_report(None)
    assert "compte étudiant" in [obj for kind, obj in _Sent.log if kind == "message.send"][-1].content


async def test_failed_quiz_tool_offers_a_direct_retry(ui):
    events = [
        ToolStartEvent(
            "c1", "generate_quiz", {"topic": "volatilité réalisée", "n": 10, "difficulty": "difficile"}, "🎓"
        ),
        ToolEndEvent("c1", "generate_quiz", "La génération du quiz a échoué (ValueError)", ok=False),
        FinalEvent(text="Il semble y avoir eu un problème. Voulez-vous réessayer ?"),
    ]
    await ui.run_turn(FakeEngine(events), "quiz de 10 questions", [], [], session_id="t", user_id="u")
    msg = next(obj for kind, obj in _Sent.log if kind == "message.send")
    assert [a.name for a in msg.actions] == ["quiz_retry"]
    assert msg.actions[0].payload == {"topic": "volatilité réalisée", "n": 10, "difficulty": "difficile"}
    assert ui.cl.user_session.get("quiz_mode") is None


async def test_retry_quiz_regenerates_with_the_same_parameters(ui):
    class Engine:
        user_level = "M1"
        calls = []

        async def generate_quiz(self, topic, num_questions=3, difficulty="moyen", *, focus=None, **kw):
            self.calls.append((topic, num_questions, difficulty))
            return Quiz(
                topic=topic,
                questions=[QuizQuestion(question="q", options=["a", "b"], correct_index=0, explanation="e")] * 4,
                difficulty=difficulty,
            )

    action = FakeAction(name="quiz_retry", payload={"topic": "VaR", "n": 10, "difficulty": "difficile"})
    await ui.retry_quiz(action, Engine(), user_id="u", thread_id="t")
    assert Engine.calls == [("VaR", 10, "difficile")]
    wait = next(obj for kind, obj in _Sent.log if kind == "message.send")
    assert wait.content.startswith("🎓 Quiz de 4 questions sur **VaR**") and "4 sur les 10 demandées" in wait.content
    assert ui.cl.user_session.get("quiz_mode") == "active"


async def test_launch_quiz_reports_failures_and_missing_topics(ui, monkeypatch):
    from ela.engine import GenerationError

    class Engine:
        user_level = "M1"

        def __init__(self, outcome):
            self.outcome = outcome

        async def generate_quiz(self, topic, num_questions=3, difficulty="moyen", *, focus=None, **kw):
            if self.outcome == "fail":
                raise GenerationError("quiz")
            return None

    assert not await ui.launch_quiz(Engine("fail"), "ADF", n=5, difficulty="moyen", user_id=None, thread_id=None)
    wait = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert wait.content.startswith("⚠️ La génération du quiz a échoué")
    assert wait.actions[0].payload == {"topic": "ADF", "n": 5, "difficulty": "moyen"}

    monkeypatch.setattr(ui, "courses_elsewhere", lambda topic, level: [("Financial_Econometrics", "M2")])
    assert not await ui.launch_quiz(Engine("none"), "GARCH", n=3, difficulty="moyen", user_id=None, thread_id=None)
    wait = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert "n'est pas au programme des cours accessibles à votre niveau (M1)" in wait.content
    assert "**Financial_Econometrics** (niveau M2)" in wait.content

    monkeypatch.setattr(ui, "courses_elsewhere", lambda topic, level: [])
    assert not await ui.launch_quiz(Engine("none"), "zzz", n=3, difficulty="moyen", user_id=None, thread_id=None)
    wait = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert wait.content.startswith("⚠️ Je n'ai rien trouvé sur « zzz »")


# ---------------------------------------------------------------- review sheets & follow-up


def _sheet(course="Univariate_Time_Series", chapter="Chapitre 1 : processus"):
    from ela.pedagogy.schemas import ReviewSection, ReviewSheet

    return ReviewSheet(
        title="Processus stationnaires",
        objectives=["Définir la stationnarité"],
        sections=[ReviewSection(heading="Stationnarité", key_points=["$E(X_t)=\\mu$ constant"])],
        takeaways=["Deux premiers moments invariants"],
        course=course,
        chapter=chapter,
        level="M1",
        slides=["Stationnarité faible"],
    )


async def test_show_review_sheet_message_file_and_activity(ui):
    await ui.show_review_sheet(_sheet(), user_id="alice", thread_id="t1")
    msg = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert msg.content.startswith("# 📄 Processus stationnaires") and "## À retenir" in msg.content
    assert msg.elements[0].name == "fiche_univariate-time-series_chapitre-1-processus.md"
    assert msg.elements[0].content.decode("utf-8").startswith("# 📄 Processus stationnaires")
    assert [a.label for a in msg.actions] == ["🎓 Me tester sur ce chapitre", "✏️ Un exercice"]
    assert msg.actions[0].payload["chapter"] == "Chapitre 1 : processus"
    rec = ui.RECORDED["questions"][-1]
    assert rec.question == "Fiche de révision : Univariate_Time_Series › Chapitre 1 : processus"
    assert rec.course == "Univariate_Time_Series" and rec.chapter == "Chapitre 1 : processus" and rec.n_sources == 1


async def test_run_turn_shows_the_sheet_of_the_agent_and_records_its_chapter(ui):
    from ela.agent.events import ReviewSheetEvent

    events = [
        ToolStartEvent("c1", "generate_review_sheet", {"course": "UTS", "chapter": "1"}, "📄"),
        ToolEndEvent("c1", "generate_review_sheet", "Fiche « Processus stationnaires » générée"),
        ReviewSheetEvent(_sheet()),
        SourcesEvent(cited=[], all=[]),
        FinalEvent(text="Voici la fiche du chapitre 1."),
    ]
    await ui.run_turn(FakeEngine(events), "fiche du chapitre 1", [], [], session_id="t", user_id="alice")
    answer = [obj for kind, obj in _Sent.log if kind == "message.send"][0]
    assert answer.content == "Voici la fiche du chapitre 1." and answer.actions == []  # no follow-up on a sheet turn
    sheet_msg = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert sheet_msg.content.startswith("# 📄 Processus stationnaires")
    turn_rec = ui.RECORDED["questions"][0]
    assert turn_rec.question == "fiche du chapitre 1" and turn_rec.course == "Univariate_Time_Series"
    assert turn_rec.chapter == "Chapitre 1 : processus"


def test_followup_actions_rules(ui):
    final = FinalEvent(text="Le test ADF [S1].")
    sources = SourcesEvent(cited=[_ref()], all=[_ref()])
    labels = [a.label for a in ui.followup_actions("Comment tester en Python ?", final, sources)]
    assert labels == ["🎓 Me tester : adf", "✏️ Exercice : adf", "📄 Fiche : Chapitre 2", "💻 Exemple Python"]
    assert ui.followup_actions("ADF ?", final, SourcesEvent(cited=[], all=[_ref()])) == []
    refusal = FinalEvent(
        text="Je suis spécialisé en **Séries Temporelles** et **Économétrie Financière**. "
        "Ce sujet sort du cadre du cours."
    )
    assert ui.followup_actions("foot ?", refusal, sources) == []
    ui.cl.user_session.set("quiz_mode", "active")
    assert ui.followup_actions("ADF ?", final, sources) == []
    ui.cl.user_session.set("quiz_mode", None)
    no_query = SourceRef(
        "S1", "c1", "UTS.tex", "UTS", "M1", "Chapitre 2", "", "Test ADF", "x", None, None, "slide", None
    )
    labels = [
        a.label
        for a in ui.followup_actions(
            "Explique le test ADF en détail s'il te plaît", final, SourcesEvent([no_query], [no_query])
        )
    ]
    assert labels[0] == "🎓 Me tester : Explique le test ADF en détail s'il te…"


async def test_follow_suggestion_routes_sheet_quiz_and_code(ui):
    class Engine:
        user_level = "M1"
        calls = []

        async def generate_review_sheet(self, course, chapter):
            self.calls.append(("sheet", course, chapter))
            return _sheet()

        async def generate_quiz(self, topic, num_questions=3, difficulty="moyen", *, focus=None, **kw):
            self.calls.append(("quiz", topic, kw))
            return None

        async def generate_code(self, topic, language):
            self.calls.append(("code", topic, language))
            return "```python\nprint(1)\n```"

    eng = Engine()
    sheet = FakeAction(name="report_suggestion", payload={"kind": "sheet", "course": "UTS", "chapter": "1"})
    await ui.follow_suggestion(sheet, eng, user_id=None, thread_id=None)
    quiz = FakeAction(
        name="report_suggestion", payload={"kind": "quiz", "topic": "adf", "course": "UTS", "chapter": "Ch 2"}
    )
    await ui.follow_suggestion(quiz, eng, user_id=None, thread_id=None)
    code = FakeAction(name="report_suggestion", payload={"kind": "code", "topic": "adf", "language": "Python"})
    await ui.follow_suggestion(code, eng, user_id=None, thread_id=None)
    assert Engine.calls == [
        ("sheet", "UTS", "1"),
        ("quiz", "adf", {"course": "UTS", "chapter": "Ch 2"}),
        ("code", "adf", "Python"),
    ]
    sent = [obj.content for kind, obj in _Sent.log if kind == "message.send"]
    assert any(c.startswith("# 📄 Processus stationnaires") for c in sent)
    assert "print(1)" in sent[-1]


async def test_sheet_picker_flow(ui, monkeypatch):
    from ela.agent.outline import build_outlines
    from test_agent_tools import CORPUS

    outlines = build_outlines(CORPUS)
    visible = {k: v for k, v in outlines.items() if v.level == "M1"}
    monkeypatch.setattr(ui, "get_outlines", lambda level: visible)

    class Engine:
        user_level = "M1"

        async def generate_review_sheet(self, course, chapter):
            from ela.pedagogy.review import resolve_target

            outline, ch = resolve_target("M1", course, chapter) if False else (None, None)
            return _sheet(course, chapter)

    await ui.start_sheet_picker(Engine())
    assert ui.cl.user_session.get("sheet_mode") == "waiting"
    picker = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert [a.payload["course"] for a in picker.actions] == ["Multivariate_Time_Series", "Univariate_Time_Series"]
    await ui.pick_sheet_course(FakeAction(name="sheet_course", payload={"course": "Univariate_Time_Series"}), Engine())
    chapters = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert [a.label for a in chapters.actions] == ["1. Chapitre 1 : processus", "2. Chapitre 2 : racines unitaires"]
    await ui.pick_sheet_chapter(
        FakeAction(name="sheet_chapter", payload=chapters.actions[1].payload), Engine(), user_id=None, thread_id=None
    )
    assert ui.cl.user_session.get("sheet_mode") is None
    last = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert "Chapitre 2 : racines unitaires" in last.content

    assert ui.parse_sheet_request("UTS, chapitre 2", visible) == ("Univariate_Time_Series", "chapitre 2")
    assert ui.parse_sheet_request("racines unitaires (UTS)", visible) == (
        "Univariate_Time_Series",
        "racines unitaires",
    )
    assert ui.parse_sheet_request("chapitre 2", visible) is None  # two visible courses: ambiguous
    assert ui.parse_sheet_request("chapitre 2", {"Univariate_Time_Series": visible["Univariate_Time_Series"]}) == (
        "Univariate_Time_Series",
        "chapitre 2",
    )
    await ui.handle_sheet_request(Engine(), "chapitre 2", user_id=None, thread_id=None)
    assert [obj for kind, obj in _Sent.log if kind == "message.send"][-1].content.startswith("⚠️ Indiquez le cours")


async def test_launch_review_sheet_reports_lookup_and_generation_errors(ui):
    from ela.engine import GenerationError
    from ela.pedagogy.review import ReviewLookupError

    class Engine:
        user_level = "M1"

        def __init__(self, exc):
            self.exc = exc

        async def generate_review_sheet(self, course, chapter):
            raise self.exc

    assert not await ui.launch_review_sheet(
        Engine(ReviewLookupError("Chapitre « 9 » inconnu")), "UTS", "9", user_id=None, thread_id=None
    )
    assert [obj for kind, obj in _Sent.log if kind == "message.send"][-1].content == "⚠️ Chapitre « 9 » inconnu"
    assert not await ui.launch_review_sheet(Engine(GenerationError("fiche")), "UTS", "1", user_id=None, thread_id=None)
    assert [obj for kind, obj in _Sent.log if kind == "message.send"][-1].content.startswith(
        "⚠️ La génération de la fiche a échoué"
    )


# ---------------------------------------------------------------- code execution (Atelier Code 2.0)


@pytest.fixture
def sandbox_on(monkeypatch):
    from ela.config import get_settings

    monkeypatch.setenv("SANDBOX_URL", "http://sandbox:8080")
    monkeypatch.setenv("CODE_AUTO_FIX", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def test_launch_code_offers_to_run_python_only_when_the_sandbox_is_on(ui, sandbox_on):
    class Engine:
        async def generate_code(self, topic, language):
            return f"Intro\n```{language.lower()}\nprint('hi')\n```\nFin"

    await ui.launch_code(Engine(), "AR(1)", "Python")
    msg = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert "print('hi')" in msg.content and [a.name for a in msg.actions] == ["code_run"]
    assert ui.cl.user_session.get("code_last") == {
        "code": "print('hi')",
        "topic": "AR(1)",
        "language": "Python",
        "attempts": [],
    }
    await ui.launch_code(Engine(), "AR(1)", "R")
    msg = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert msg.actions == []  # R is not executed


async def test_launch_code_without_sandbox_has_no_button(ui, monkeypatch):
    from ela.config import get_settings

    monkeypatch.delenv("SANDBOX_URL", raising=False)
    get_settings.cache_clear()

    class Engine:
        async def generate_code(self, topic, language):
            return "```python\nprint(1)\n```"

    await ui.launch_code(Engine(), "x", "Python")
    assert [obj for kind, obj in _Sent.log if kind == "message.send"][-1].actions == []


async def test_run_last_code_shows_output_figures_and_fix_button(ui, sandbox_on, monkeypatch):
    from ela.sandbox import service
    from ela.sandbox.client import RunResult

    async def _remaining(user_id):
        return 5

    monkeypatch.setattr(service, "quota_remaining", _remaining)

    class Engine:
        calls = []

        def __init__(self, result):
            self.result = result

        async def run_code(self, code, **kw):
            self.calls.append((code, kw))
            return self.result

        async def fix_code(self, code, error, *, topic="", language="Python", history=None):
            self.calls.append(("fix", code, error[-16:], topic, list(history or [])))
            return "print('fixed')"

        async def interpret_run(self, code, result, *, topic="", language="Python"):
            from ela.sandbox.service import Interpretation

            self.calls.append(("interpret", code, result.stdout, topic))
            return Interpretation("| π | 3.14 |\n\nCommentaire [S1].", cited=[_ref()])

    ui.remember_code("print(1/0)", topic="division", language="Python")
    err = RunResult("error", stderr="Traceback\nZeroDivisionError: division by zero", exit_code=1)
    eng = Engine(err)
    await ui.run_last_code(FakeAction(name="code_run"), eng, user_id="alice", thread_id="t")
    assert eng.calls[0] == (
        "print(1/0)",
        {"user_id": "alice", "thread_id": "t", "origin": "workshop", "topic": "division", "language": "Python"},
    )
    step = [obj for kind, obj in _Sent.log if kind == "step.send"][-1]
    assert step.output.startswith("⚠️ error en 0.0 s — 4 exécution(s) restante(s)")
    out = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert out.content.startswith("❌ **Le script a échoué**") and [a.name for a in out.actions] == ["code_fix"]

    await ui.fix_last_code(FakeAction(name="code_fix"), eng)
    assert eng.calls[-1] == ("fix", "print(1/0)", "division by zero", "division", [])
    fixed = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert "print('fixed')" in fixed.content and [a.name for a in fixed.actions] == ["code_run"]
    assert ui.cl.user_session.get("code_last")["code"] == "print('fixed')"
    assert ui.cl.user_session.get("code_last")["origin"] == "fix"

    ok = RunResult("ok", stdout="3.14", elapsed_ms=2100, images=[b"\x89PNG..."])
    eng = Engine(ok)
    await ui.run_last_code(FakeAction(name="code_run"), eng, user_id="alice", thread_id="t")
    out, interp = [obj for kind, obj in _Sent.log if kind == "message.send"][-2:]
    assert out.content.startswith("✅ **Exécution terminée** en 2.1 s") and out.actions == []
    assert out.elements[0].name == "figure_1" and out.elements[0].content == b"\x89PNG..."
    assert eng.calls[-1] == ("interpret", "print('fixed')", "3.14", "division")
    assert interp.content.startswith("| π | 3.14 |") and [e.name for e in interp.elements] == ["S1"]
    step = [obj for kind, obj in _Sent.log if kind == "step.send"][-1]
    assert step.name == "🧠 Interprétation des résultats" and step.output == "1 extrait(s) du cours cité(s)"


async def test_run_last_code_reports_quota_and_outage(ui, sandbox_on, monkeypatch):
    from ela.sandbox import service
    from ela.sandbox.client import SandboxUnavailableError

    async def _remaining(user_id):
        return 0

    monkeypatch.setattr(service, "quota_remaining", _remaining)

    class Engine:
        def __init__(self, exc):
            self.exc = exc

        async def run_code(self, code, **kw):
            raise self.exc

    ui.remember_code("print(1)", topic="t", language="Python")
    await ui.run_last_code(
        FakeAction(name="code_run"), Engine(service.QuotaExceededError(10)), user_id="a", thread_id=None
    )
    assert "quota de **10 exécutions par jour**" in [o for k, o in _Sent.log if k == "message.send"][-1].content
    await ui.run_last_code(
        FakeAction(name="code_run"), Engine(SandboxUnavailableError("down")), user_id="a", thread_id=None
    )
    assert "indisponible" in [o for k, o in _Sent.log if k == "message.send"][-1].content
    ui.cl.user_session.set("code_last", None)
    await ui.run_last_code(FakeAction(name="code_run"), Engine(None), user_id="a", thread_id=None)
    assert "Aucun script" in [o for k, o in _Sent.log if k == "message.send"][-1].content


async def test_run_turn_shows_agent_code_runs(ui, sandbox_on):
    from ela.agent.events import CodeRunEvent
    from ela.sandbox.client import RunResult

    events = [
        ToolStartEvent("c1", "run_python", {"code": "print(42)"}, "💻"),
        ToolEndEvent("c1", "run_python", "Statut : ok (5 ms)."),
        CodeRunEvent("print(42)", RunResult("ok", stdout="42", elapsed_ms=5)),
        SourcesEvent(cited=[], all=[]),
        FinalEvent(text="La simulation donne 42."),
    ]
    await ui.run_turn(FakeEngine(events), "simule", [], [], session_id="t", user_id="alice")
    sent = [obj for kind, obj in _Sent.log if kind == "message.send"]
    assert sent[0].content == "La simulation donne 42." and sent[-1].content.startswith("✅ **Exécution terminée**")
    assert "```text\n42\n```" in sent[-1].content


async def test_interpretation_without_numbers_offers_to_complete_the_script(ui, sandbox_on):
    from ela.sandbox import service
    from ela.sandbox.client import RunResult

    class Engine:
        calls = []

        async def interpret_run(self, code, result, *, topic="", language="Python"):
            return service.Interpretation("ℹ️ rien à interpréter", needs_output=True)

        async def fix_code(self, code, error, *, topic="", language="Python", history=None):
            self.calls.append((code, error, topic))
            return "print('now with numbers')"

    eng = Engine()
    await ui.interpret_run(eng, "plt.plot([1])", RunResult("ok", images=[b"png"]), topic="AR(1)", language="Python")
    msg = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert msg.content == "ℹ️ rien à interpréter" and [a.label for a in msg.actions] == ["🔧 Compléter le script"]
    assert ui.cl.user_session.get("code_last_error") == service.NO_OUTPUT_HINT
    await ui.fix_last_code(FakeAction(name="code_fix"), eng)
    assert eng.calls == [("plt.plot([1])", service.NO_OUTPUT_HINT, "AR(1)")]
    msg = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert msg.content.startswith("🔧 **Script complété**") and [a.name for a in msg.actions] == ["code_run"]


async def test_interpretation_failure_is_reported(ui, sandbox_on):
    from ela.engine import GenerationError
    from ela.sandbox.client import RunResult

    class Engine:
        async def interpret_run(self, code, result, *, topic="", language="Python"):
            raise GenerationError("interprétation")

    await ui.interpret_run(Engine(), "print(1)", RunResult("ok", stdout="1"), topic="t", language="Python")
    assert [o for k, o in _Sent.log if k == "message.send"][-1].content.startswith("⚠️ Je n'ai pas réussi à interpréter")


async def test_launch_code_fixes_syntax_before_display(ui, sandbox_on):
    class Engine:
        calls = []

        async def generate_code(self, topic, language):
            return "Intro\n```python\nprint('hi'\n```\nFin"  # truncated / unbalanced

        async def fix_code(self, code, error, *, topic="", language="Python", history=None):
            self.calls.append((code, error.splitlines()[0], list(history or [])))
            return "print('hi')"

    eng = Engine()
    await ui.launch_code(eng, "AR(1)", "Python")
    assert len(eng.calls) == 1 and eng.calls[0][0] == "print('hi'" and eng.calls[0][1].startswith("SyntaxError")
    msg = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert "🔧 **Script vérifié et corrigé (syntaxe)**" in msg.content and "print('hi')\n```" in msg.content
    assert ui.cl.user_session.get("code_last")["code"] == "print('hi')"
    assert [a.name for a in msg.actions] == ["code_run"]


async def test_run_last_code_auto_fixes_and_reruns_once(ui, sandbox_on, monkeypatch):
    from ela.config import get_settings
    from ela.sandbox import service
    from ela.sandbox.client import RunResult

    monkeypatch.setenv("CODE_AUTO_FIX", "true")
    get_settings.cache_clear()

    async def _remaining(user_id):
        return 5

    monkeypatch.setattr(service, "quota_remaining", _remaining)

    class Engine:
        calls = []
        results = [
            RunResult("error", stderr="NameError: name 'np' is not defined", exit_code=1),
            RunResult("ok", stdout="3.14", elapsed_ms=10),
        ]

        async def run_code(self, code, **kw):
            self.calls.append(("run", code, kw["origin"]))
            return self.results.pop(0)

        async def fix_code(self, code, error, *, topic="", language="Python", history=None):
            self.calls.append(("fix", code, error, list(history or [])))
            return "import numpy as np\nprint(np.pi)"

        async def interpret_run(self, code, result, *, topic="", language="Python"):
            self.calls.append(("interpret", code))
            return service.Interpretation("| π | 3.14 |")

    ui.remember_code("print(np.pi)", topic="pi", language="Python")
    eng = Engine()
    await ui.run_last_code(FakeAction(name="code_run"), eng, user_id="alice", thread_id="t")
    assert eng.calls == [
        ("run", "print(np.pi)", "workshop"),
        ("fix", "print(np.pi)", "NameError: name 'np' is not defined", []),
        ("run", "import numpy as np\nprint(np.pi)", "fix"),
        ("interpret", "import numpy as np\nprint(np.pi)"),
    ]
    steps = [obj.name for kind, obj in _Sent.log if kind == "step.send"]
    assert steps == [
        "▶️ Exécution du script dans le bac à sable",
        "🔧 Correction automatique et nouvelle exécution",
        "🧠 Interprétation des résultats",
    ]
    sent = [obj.content for kind, obj in _Sent.log if kind == "message.send"]
    assert sent[-3].startswith("🔧 **Script corrigé automatiquement**")
    assert sent[-2].startswith("✅ **Exécution terminée**") and sent[-1] == "| π | 3.14 |"
    last = ui.cl.user_session.get("code_last")
    assert last["auto_fixed"] and last["attempts"] == ["NameError: name 'np' is not defined"]

    # A second failure of the fixed script is shown with the manual button, no second automatic pass
    class Engine2(Engine):
        results = [RunResult("error", stderr="ValueError: boom", exit_code=1)]

    eng2 = Engine2()
    eng2.calls = []
    await ui.run_last_code(FakeAction(name="code_run"), eng2, user_id="alice", thread_id="t")
    assert [c[0] for c in eng2.calls] == ["run"]
    out = [obj for kind, obj in _Sent.log if kind == "message.send"][-1]
    assert out.content.startswith("❌") and [a.name for a in out.actions] == ["code_fix"]
    # the manual fix then receives the history of both errors except the current one
    await ui.fix_last_code(FakeAction(name="code_fix"), eng2)
    fix = [c for c in eng2.calls if c[0] == "fix"][-1]
    assert fix[2] == "ValueError: boom" and fix[3] == ["NameError: name 'np' is not defined"]
