# admin/views/feedbacks.py
"""
Feedbacks page for the ELA backoffice.

Displays user feedback (thumbs up/down) with the associated
assistant response and conversation context. Supports filtering
by user, date range, and feedback score.

Data sources: feedbacks, steps, threads, users tables.
"""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from db import execute_query


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_feedbacks(
    user_filter: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    score_filter: int | None = None,
    limit: int = 100,
) -> list[dict]:
    """Fetch feedbacks joined with step and thread context.

    Args:
        user_filter: Filter by user identifier (exact match).
        date_from: Include feedbacks from this date onward.
        date_to: Include feedbacks up to this date.
        score_filter: Filter by feedback value (1 = positive, 0 = negative).
        limit: Maximum number of rows to return.

    Returns:
        List of dicts with feedback + context fields.
    """
    conditions = []
    params: dict = {"limit": limit}

    if user_filter and user_filter != "Tous":
        conditions.append('t."userIdentifier" = %(user)s')
        params["user"] = user_filter

    if date_from:
        conditions.append('s."createdAt"::DATE >= %(date_from)s')
        params["date_from"] = str(date_from)

    if date_to:
        conditions.append('s."createdAt"::DATE <= %(date_to)s')
        params["date_to"] = str(date_to)

    if score_filter is not None:
        conditions.append("f.value = %(score)s")
        params["score"] = score_filter

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

    query = f"""
        SELECT
            f.id AS feedback_id,
            f.value AS score,
            f.comment,
            s.id AS step_id,
            s.output AS assistant_response,
            s."createdAt" AS step_date,
            t.id AS thread_id,
            t.name AS thread_name,
            t."userIdentifier" AS user_identifier
        FROM feedbacks f
        JOIN steps s ON f."forId" = s.id
        JOIN threads t ON s."threadId" = t.id
        {where_clause}
        ORDER BY s."createdAt" DESC
        LIMIT %(limit)s
    """

    return execute_query(query, params)


def _load_user_list() -> list[str]:
    """Fetch distinct user identifiers that have at least one feedback.

    Returns:
        Sorted list of user identifiers.
    """
    rows = execute_query("""
        SELECT DISTINCT t."userIdentifier" AS uid
        FROM feedbacks f
        JOIN steps s ON f."forId" = s.id
        JOIN threads t ON s."threadId" = t.id
        WHERE t."userIdentifier" IS NOT NULL
        ORDER BY uid
    """)
    return [r["uid"] for r in rows]


def _load_feedback_stats() -> dict:
    """Compute aggregate feedback statistics.

    Returns:
        Dict with total, positive, negative counts.
    """
    rows = execute_query("""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE value = 1) AS positive,
            COUNT(*) FILTER (WHERE value = 0) AS negative
        FROM feedbacks
    """)
    return rows[0] if rows else {"total": 0, "positive": 0, "negative": 0}


def _get_conversation_context(thread_id: str, step_id: str) -> list[dict]:
    """Fetch a few steps around the rated step for context.

    Retrieves up to 3 steps before and 1 step after the rated step
    within the same thread, giving the admin enough context to
    understand the exchange.

    Args:
        thread_id: The thread UUID.
        step_id: The rated step UUID.

    Returns:
        List of step dicts ordered chronologically.
    """
    return execute_query(
        """
        WITH target AS (
            SELECT "createdAt" FROM steps WHERE id = %(step_id)s
        )
        SELECT
            s.id,
            s.type,
            s.name,
            s.output,
            s.input,
            s."createdAt"
        FROM steps s, target
        WHERE s."threadId" = %(thread_id)s
          AND s.type IN ('user_message', 'assistant_message')
          AND s."createdAt" >= (target."createdAt"::TIMESTAMP - INTERVAL '10 minutes')
          AND s."createdAt" <= (target."createdAt"::TIMESTAMP + INTERVAL '2 minutes')
        ORDER BY s."createdAt" ASC
        LIMIT 8
        """,
        {"thread_id": thread_id, "step_id": step_id},
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_filters() -> tuple:
    """Render the filter bar and return selected filter values.

    Returns:
        Tuple of (user_filter, date_from, date_to, score_filter).
    """
    users = ["Tous"] + _load_user_list()

    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

    with col1:
        user_filter = st.selectbox("Utilisateur", users, key="fb_user")

    with col2:
        date_from = st.date_input(
            "Du",
            value=date.today() - timedelta(days=30),
            key="fb_from",
        )

    with col3:
        date_to = st.date_input("Au", value=date.today(), key="fb_to")

    with col4:
        score_options = {"Tous": None, "👍 Positif": 1, "👎 Négatif": 0}
        score_label = st.selectbox("Score", list(score_options.keys()), key="fb_score")
        score_filter = score_options[score_label]

    return (
        user_filter if user_filter != "Tous" else None,
        date_from,
        date_to,
        score_filter,
    )


def _render_stats(stats: dict):
    """Render the top-level feedback metrics."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total feedbacks", stats["total"])
    col2.metric("👍 Positifs", stats["positive"])
    col3.metric("👎 Négatifs", stats["negative"])

    if stats["total"] > 0:
        rate = round(stats["positive"] / stats["total"] * 100, 1)
        col4.metric("Taux satisfaction", f"{rate}%")
    else:
        col4.metric("Taux satisfaction", "—")


def _render_feedback_list(feedbacks: list[dict]):
    """Render the feedback entries as expandable cards."""
    if not feedbacks:
        st.info("Aucun feedback trouvé pour ces critères.")
        return

    for fb in feedbacks:
        icon = "👍" if fb["score"] == 1 else "👎"
        user = fb.get("user_identifier", "?")
        date_str = str(fb.get("step_date", ""))[:19]
        thread_name = fb.get("thread_name", "Sans nom") or "Sans nom"

        header = f"{icon} **{user}** — {thread_name} — `{date_str}`"

        with st.expander(header, expanded=False):
            # Assistant response that was rated
            response_text = fb.get("assistant_response", "")
            if response_text:
                st.markdown("**Réponse ELA :**")
                st.markdown(
                    response_text[:1500]
                    + ("…" if len(response_text or "") > 1500 else "")
                )
            else:
                st.caption("(Réponse non disponible)")

            # Optional comment
            if fb.get("comment"):
                st.markdown(f"**Commentaire :** {fb['comment']}")

            # Context button
            if st.button(
                "🔍 Voir le contexte",
                key=f"ctx_{fb['feedback_id']}",
            ):
                context = _get_conversation_context(
                    fb["thread_id"], fb["step_id"],
                )
                if context:
                    st.markdown("---")
                    st.markdown("**Échange autour de cette réponse :**")
                    for step in context:
                        if step["type"] == "user_message":
                            content = step.get("input") or step.get("output", "")
                            st.chat_message("user").markdown(content[:800])
                        else:
                            content = step.get("output", "")
                            is_rated = step["id"] == fb["step_id"]
                            prefix = "⭐ " if is_rated else ""
                            st.chat_message("assistant").markdown(
                                prefix + content[:800]
                            )
                else:
                    st.caption("Contexte non disponible.")


# ---------------------------------------------------------------------------
# Main page entry point
# ---------------------------------------------------------------------------

def show_feedbacks_page():
    """Render the full feedbacks page."""
    st.markdown("## ⭐ Feedbacks utilisateurs")
    st.caption(
        "Consultez les retours (👍/👎) laissés par les étudiants "
        "sur les réponses d'ELA."
    )
    st.divider()

    # Stats
    stats = _load_feedback_stats()
    _render_stats(stats)

    st.divider()

    # Filters
    user_filter, date_from, date_to, score_filter = _render_filters()

    st.divider()

    # Feedback list
    feedbacks = _load_feedbacks(
        user_filter=user_filter,
        date_from=date_from,
        date_to=date_to,
        score_filter=score_filter,
    )

    st.caption(f"{len(feedbacks)} feedback(s) trouvé(s)")
    _render_feedback_list(feedbacks)
