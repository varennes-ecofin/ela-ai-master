# admin/views/pedagogy.py
"""
Pedagogy page for the ELA backoffice (Phase 4c).

What the students work on and where they struggle, from the tables fed by
the chat: ``student_activity`` (questions answered by the agent),
``quiz_results`` (one row per answered quiz question) and
``exercise_attempts`` (Socratic mode) and ``code_runs`` (Atelier Code 2.0).
Signals for the teacher: questions per chapter, notions failed in quizzes,
subjects asked but absent from the course, scripts that fail.
"""

from datetime import date, timedelta

import streamlit as st
from db import execute_query

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _period_clause(column: str, start: date, end: date) -> tuple[str, dict]:
    return f"{column}::DATE BETWEEN %(start)s AND %(end)s", {"start": str(start), "end": str(end)}


def _quiz_kpis(start: date, end: date) -> dict:
    where, params = _period_clause("answered_at", start, end)
    rows = execute_query(
        f"""
        SELECT count(*) AS answers, count(*) FILTER (WHERE correct) AS correct,
               count(DISTINCT quiz_id) AS quizzes, count(DISTINCT user_identifier) AS students
        FROM quiz_results WHERE {where}
        """,
        params,
    )
    return rows[0] if rows else {"answers": 0, "correct": 0, "quizzes": 0, "students": 0}


def _quiz_by_topic(start: date, end: date) -> list[dict]:
    where, params = _period_clause("answered_at", start, end)
    return execute_query(
        f"""
        SELECT topic, course, count(*) AS answers, count(*) FILTER (WHERE correct) AS correct,
               round(100.0 * count(*) FILTER (WHERE correct) / count(*), 0) AS success_pct,
               count(DISTINCT user_identifier) AS students
        FROM quiz_results WHERE {where}
        GROUP BY topic, course ORDER BY success_pct ASC, answers DESC
        """,
        params,
    )


def _failed_questions(start: date, end: date, limit: int = 30) -> list[dict]:
    where, params = _period_clause("answered_at", start, end)
    params["limit"] = limit
    return execute_query(
        f"""
        SELECT question, topic, count(*) AS failures,
               count(DISTINCT user_identifier) AS students, max(answered_at) AS last_seen
        FROM quiz_results WHERE {where} AND NOT correct
        GROUP BY question, topic ORDER BY failures DESC, last_seen DESC LIMIT %(limit)s
        """,
        params,
    )


def _exercise_stats(start: date, end: date) -> list[dict]:
    where, params = _period_clause("created_at", start, end)
    return execute_query(
        f"""
        SELECT topic, count(DISTINCT exercise_id) AS exercises, count(*) AS attempts,
               count(DISTINCT user_identifier) AS students,
               count(DISTINCT exercise_id) FILTER (WHERE completed) AS completed,
               round(100.0 * count(*) FILTER (WHERE verdict = 'correct') / count(*), 0) AS correct_steps_pct
        FROM exercise_attempts WHERE {where}
        GROUP BY topic ORDER BY attempts DESC
        """,
        params,
    )


def _questions_by_chapter(start: date, end: date) -> list[dict]:
    where, params = _period_clause("created_at", start, end)
    return execute_query(
        f"""
        SELECT coalesce(course, '(hors cours)') AS course, coalesce(chapter, '—') AS chapter,
               count(*) AS questions, count(DISTINCT user_identifier) AS students
        FROM student_activity WHERE {where} AND NOT refused
        GROUP BY course, chapter ORDER BY questions DESC
        """,
        params,
    )


def _recent_questions(start: date, end: date, course: str | None, limit: int = 50) -> list[dict]:
    where, params = _period_clause("created_at", start, end)
    params["limit"] = limit
    course_clause = ""
    if course and course != "Tous":
        course_clause = " AND course = %(course)s"
        params["course"] = course
    return execute_query(
        f"""
        SELECT created_at, user_identifier, course, chapter, left(question, 160) AS question,
               n_searches, n_sources, general_knowledge, refused
        FROM student_activity WHERE {where}{course_clause}
        ORDER BY created_at DESC LIMIT %(limit)s
        """,
        params,
    )


def _out_of_corpus(start: date, end: date, limit: int = 50) -> list[dict]:
    """Questions answered from general knowledge or without any course passage (signal for the course)."""
    where, params = _period_clause("created_at", start, end)
    params["limit"] = limit
    return execute_query(
        f"""
        SELECT created_at, user_identifier, left(question, 160) AS question,
               CASE WHEN refused THEN 'refusée (hors domaine)'
                    WHEN general_knowledge THEN 'connaissances générales'
                    ELSE 'aucun extrait cité' END AS raison
        FROM student_activity
        WHERE {where} AND (refused OR general_knowledge OR n_sources = 0)
        ORDER BY created_at DESC LIMIT %(limit)s
        """,
        params,
    )


def _code_runs(start: date, end: date) -> list[dict]:
    where, params = _period_clause("created_at", start, end)
    return execute_query(
        f"""
        SELECT status, origin, count(*) AS runs, count(DISTINCT user_identifier) AS students,
               round(avg(elapsed_ms) / 1000.0, 1) AS avg_seconds, sum(n_images) AS figures
        FROM code_runs WHERE {where}
        GROUP BY status, origin ORDER BY runs DESC
        """,
        params,
    )


def _code_errors(start: date, end: date, limit: int = 30) -> list[dict]:
    where, params = _period_clause("created_at", start, end)
    params["limit"] = limit
    return execute_query(
        f"""
        SELECT left(error, 160) AS error, count(*) AS runs, max(created_at) AS last_seen
        FROM code_runs WHERE {where} AND status IN ('error', 'timeout') AND error IS NOT NULL
        GROUP BY left(error, 160) ORDER BY runs DESC, last_seen DESC LIMIT %(limit)s
        """,
        params,
    )


def _code_by_student(start: date, end: date) -> list[dict]:
    where, params = _period_clause("created_at", start, end)
    return execute_query(
        f"""
        SELECT user_identifier, count(*) AS runs, count(*) FILTER (WHERE status = 'ok') AS ok,
               count(*) FILTER (WHERE created_at >= date_trunc('day', now())) AS today
        FROM code_runs WHERE {where}
        GROUP BY user_identifier ORDER BY runs DESC
        """,
        params,
    )


def _courses() -> list[str]:
    rows = execute_query("SELECT DISTINCT course FROM student_activity WHERE course IS NOT NULL ORDER BY course")
    return ["Tous"] + [r["course"] for r in rows]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_filters() -> tuple[date, date]:
    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Du", value=date.today() - timedelta(days=30), key="ped_start")
    with col2:
        end = st.date_input("Au", value=date.today(), key="ped_end")
    return start, end


def _render_quiz(start: date, end: date) -> None:
    k = _quiz_kpis(start, end)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quiz", k["quizzes"])
    c2.metric("Réponses", k["answers"])
    rate = f"{round(100 * k['correct'] / k['answers'])} %" if k["answers"] else "—"
    c3.metric("Taux de réussite", rate)
    c4.metric("Étudiants", k["students"])

    st.markdown("#### Réussite par sujet (les plus faibles d'abord)")
    rows = _quiz_by_topic(start, end)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Aucun quiz sur la période.")

    st.markdown("#### Questions les plus ratées")
    failed = _failed_questions(start, end)
    if failed:
        st.dataframe(failed, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune erreur enregistrée sur la période.")


def _render_exercises(start: date, end: date) -> None:
    rows = _exercise_stats(start, end)
    if not rows:
        st.info("Aucun exercice sur la période.")
        return
    total_attempts = sum(r["attempts"] for r in rows)
    total_ex = sum(r["exercises"] for r in rows)
    done = sum(r["completed"] for r in rows)
    c1, c2, c3 = st.columns(3)
    c1.metric("Exercices commencés", total_ex)
    c2.metric("Terminés", f"{done} ({round(100 * done / total_ex)} %)" if total_ex else "0")
    c3.metric("Étapes soumises", total_attempts)
    st.markdown("#### Par sujet")
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_questions(start: date, end: date) -> None:
    st.markdown("#### Questions par cours et chapitre")
    rows = _questions_by_chapter(start, end)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune question enregistrée sur la période (les questions sont enregistrées depuis la phase 4c).")
        return
    course = st.selectbox("Cours", _courses(), key="ped_course")
    st.markdown("#### Dernières questions")
    st.dataframe(_recent_questions(start, end, course), use_container_width=True, hide_index=True)


def _render_out_of_corpus(start: date, end: date) -> None:
    st.caption(
        "Questions refusées, répondues avec des connaissances générales ou sans aucun extrait du cours : "
        "des sujets que les étudiants cherchent et que les supports ne couvrent pas (ou mal)."
    )
    rows = _out_of_corpus(start, end)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.success("Toutes les questions de la période ont trouvé des extraits dans le cours.")


def _render_code(start: date, end: date) -> None:
    rows = _code_runs(start, end)
    if not rows:
        st.info("Aucune exécution de script sur la période (Atelier Code 2.0 : SANDBOX_URL doit être configuré).")
        return
    total = sum(r["runs"] for r in rows)
    ok = sum(r["runs"] for r in rows if r["status"] == "ok")
    c1, c2, c3 = st.columns(3)
    c1.metric("Exécutions", total)
    c2.metric("Réussies", f"{ok} ({round(100 * ok / total)} %)" if total else "0")
    c3.metric("Étudiants", max(r["students"] for r in rows))
    st.markdown("#### Par statut et origine")
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.markdown("#### Erreurs les plus fréquentes")
    errors = _code_errors(start, end)
    if errors:
        st.dataframe(errors, use_container_width=True, hide_index=True)
    else:
        st.success("Aucune erreur d'exécution sur la période.")
    st.markdown("#### Par étudiant (quota journalier)")
    st.dataframe(_code_by_student(start, end), use_container_width=True, hide_index=True)


def show_pedagogy_page():
    """Render the pedagogy dashboard."""
    st.markdown("## 🎓 Pédagogie")
    st.caption("Ce que les étudiants travaillent, où ils échouent, ce qui manque au cours.")
    start, end = _render_filters()
    if start > end:
        st.warning("La date de début doit être antérieure à la date de fin.")
        return
    st.divider()
    tab_quiz, tab_ex, tab_q, tab_ooc, tab_code = st.tabs(
        ["🎓 Quiz", "✏️ Exercices", "💬 Questions", "🧭 Hors corpus", "💻 Code"]
    )
    with tab_quiz:
        _render_quiz(start, end)
    with tab_ex:
        _render_exercises(start, end)
    with tab_q:
        _render_questions(start, end)
    with tab_ooc:
        _render_out_of_corpus(start, end)
    with tab_code:
        _render_code(start, end)
