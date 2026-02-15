# admin/views/conversations.py
"""
Conversations page for the ELA backoffice.

Provides a read-only viewer for ELA conversations:
- List of threads with filtering by user, date, and type (Chat/Quiz/Code).
- Full conversation reader with ordered steps.
- Quick jump to related feedbacks.

Data sources: threads, steps, users tables.
"""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from db import execute_query


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Thread type detection based on naming conventions set in app.py
_THREAD_TYPE_PATTERNS = {
    "🎓 Quiz": "Quiz",
    "💻 Code": "Code",
    "💻 Atelier": "Code",
}


def _classify_thread(thread_name: str | None) -> str:
    """Classify a thread by its name prefix.

    Args:
        thread_name: The thread display name (may contain emoji prefix).

    Returns:
        One of 'Quiz', 'Code', or 'Chat'.
    """
    if not thread_name:
        return "Chat"
    for prefix, label in _THREAD_TYPE_PATTERNS.items():
        if thread_name.startswith(prefix):
            return label
    return "Chat"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_threads(
    user_filter: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    type_filter: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Fetch threads with message counts and optional filters.

    Args:
        user_filter: Filter by user identifier.
        date_from: Include threads created from this date.
        date_to: Include threads created up to this date.
        type_filter: Filter by thread type (Quiz, Code, Chat).
        limit: Maximum rows.

    Returns:
        List of thread dicts with computed fields.
    """
    conditions = []
    params: dict = {"limit": limit}

    if user_filter and user_filter != "Tous":
        conditions.append('t."userIdentifier" = %(user)s')
        params["user"] = user_filter

    if date_from:
        conditions.append('t."createdAt"::DATE >= %(date_from)s')
        params["date_from"] = str(date_from)

    if date_to:
        conditions.append('t."createdAt"::DATE <= %(date_to)s')
        params["date_to"] = str(date_to)

    # Type filtering is done post-query (based on thread name pattern)
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

    query = f"""
        SELECT
            t.id,
            t.name,
            t."userIdentifier" AS user_identifier,
            t."createdAt" AS created_at,
            COUNT(s.id) AS step_count,
            COUNT(s.id) FILTER (WHERE s.type = 'user_message') AS user_messages,
            COUNT(f.id) AS feedback_count
        FROM threads t
        LEFT JOIN steps s ON s."threadId" = t.id
        LEFT JOIN feedbacks f ON f."forId" = s.id
        {where_clause}
        GROUP BY t.id, t.name, t."userIdentifier", t."createdAt"
        ORDER BY t."createdAt" DESC
        LIMIT %(limit)s
    """

    rows = execute_query(query, params)

    # Classify and optionally filter by type
    for row in rows:
        row["type"] = _classify_thread(row.get("name"))

    if type_filter and type_filter != "Tous":
        rows = [r for r in rows if r["type"] == type_filter]

    return rows


def _load_user_list() -> list[str]:
    """Fetch distinct user identifiers that have conversations.

    Returns:
        Sorted list of user identifiers.
    """
    rows = execute_query("""
        SELECT DISTINCT "userIdentifier" AS uid
        FROM threads
        WHERE "userIdentifier" IS NOT NULL
        ORDER BY uid
    """)
    return [r["uid"] for r in rows]


def _load_conversation_steps(thread_id: str) -> list[dict]:
    """Fetch all steps for a thread, ordered chronologically.

    Args:
        thread_id: The thread UUID.

    Returns:
        List of step dicts with type, content, and feedback info.
    """
    return execute_query(
        """
        SELECT
            s.id,
            s.type,
            s.name AS step_name,
            s.input,
            s.output,
            s."createdAt" AS created_at,
            s."isError" AS is_error,
            f.value AS feedback_score,
            f.comment AS feedback_comment
        FROM steps s
        LEFT JOIN feedbacks f ON f."forId" = s.id
        WHERE s."threadId" = %(thread_id)s
        ORDER BY s."createdAt" ASC
        """,
        {"thread_id": thread_id},
    )


def _load_thread_info(thread_id: str) -> dict | None:
    """Fetch metadata for a single thread.

    Args:
        thread_id: The thread UUID.

    Returns:
        Thread dict or None.
    """
    rows = execute_query(
        """
        SELECT
            t.id,
            t.name,
            t."userIdentifier" AS user_identifier,
            t."createdAt" AS created_at,
            t.tags,
            t.metadata
        FROM threads t
        WHERE t.id = %(thread_id)s
        """,
        {"thread_id": thread_id},
    )
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_filters() -> tuple:
    """Render the filter bar for the thread list.

    Returns:
        Tuple of (user_filter, date_from, date_to, type_filter).
    """
    users = ["Tous"] + _load_user_list()

    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

    with col1:
        user_filter = st.selectbox("Utilisateur", users, key="conv_user")

    with col2:
        date_from = st.date_input(
            "Du",
            value=date.today() - timedelta(days=30),
            key="conv_from",
        )

    with col3:
        date_to = st.date_input("Au", value=date.today(), key="conv_to")

    with col4:
        type_filter = st.selectbox(
            "Type",
            ["Tous", "Chat", "Quiz", "Code"],
            key="conv_type",
        )

    return (
        user_filter if user_filter != "Tous" else None,
        date_from,
        date_to,
        type_filter if type_filter != "Tous" else None,
    )


def _render_thread_list(threads: list[dict]):
    """Render the thread list as a selectable table.

    Args:
        threads: List of thread dicts from _load_threads().
    """
    if not threads:
        st.info("Aucune conversation trouvée pour ces critères.")
        return

    # Build display dataframe
    display_data = []
    for t in threads:
        type_icon = {"Quiz": "🎓", "Code": "💻", "Chat": "💬"}.get(t["type"], "💬")
        display_data.append({
            "Type": f"{type_icon} {t['type']}",
            "Nom": t.get("name", "Sans nom") or "Sans nom",
            "Utilisateur": t.get("user_identifier", "?"),
            "Messages": t.get("user_messages", 0),
            "Feedbacks": t.get("feedback_count", 0),
            "Date": str(t.get("created_at", ""))[:16],
            "_id": t["id"],
        })

    df = pd.DataFrame(display_data)

    # Show table (without _id column)
    st.dataframe(
        df.drop(columns=["_id"]),
        use_container_width=True,
        hide_index=True,
    )

    # Thread selector
    thread_options = {
        f"{d['Utilisateur']} — {d['Nom']} ({d['Date']})": d["_id"]
        for d in display_data
    }

    selected_label = st.selectbox(
        "Sélectionner une conversation à lire",
        ["—"] + list(thread_options.keys()),
        key="conv_select",
    )

    if selected_label != "—":
        thread_id = thread_options[selected_label]
        st.session_state["selected_thread_id"] = thread_id


def _render_conversation_reader(thread_id: str):
    """Render a full conversation in chat-like format.

    Args:
        thread_id: The thread UUID to display.
    """
    info = _load_thread_info(thread_id)
    if not info:
        st.error("Conversation introuvable.")
        return

    # Header
    thread_name = info.get("name", "Sans nom") or "Sans nom"
    user = info.get("user_identifier", "?")
    created = str(info.get("created_at", ""))[:16]

    st.markdown(f"### 💬 {thread_name}")
    st.caption(f"Utilisateur : **{user}** · Créée le : {created}")
    st.divider()

    # Steps
    steps = _load_conversation_steps(thread_id)

    if not steps:
        st.info("Cette conversation ne contient aucun message.")
        return

    for step in steps:
        step_type = step.get("type", "")

        # User message
        if step_type == "user_message":
            content = step.get("input") or step.get("output", "")
            if content:
                st.chat_message("user").markdown(content[:3000])

        # Assistant message
        elif step_type == "assistant_message":
            content = step.get("output", "")
            if content:
                msg = st.chat_message("assistant")
                msg.markdown(content[:3000])

                # Show feedback badge if present
                fb_score = step.get("feedback_score")
                if fb_score is not None:
                    badge = "👍" if fb_score == 1 else "👎"
                    fb_comment = step.get("feedback_comment", "")
                    caption = f"{badge} Feedback"
                    if fb_comment:
                        caption += f" — _{fb_comment}_"
                    msg.caption(caption)

        # Error steps
        elif step.get("is_error"):
            st.error(f"❌ Erreur : {step.get('output', 'Détails non disponibles')[:500]}")

    st.divider()
    st.caption(f"Total : {len(steps)} étape(s)")

    # Back button
    if st.button("← Retour à la liste", key="conv_back"):
        del st.session_state["selected_thread_id"]
        st.rerun()


# ---------------------------------------------------------------------------
# Main page entry point
# ---------------------------------------------------------------------------

def show_conversations_page():
    """Render the full conversations page."""
    st.markdown("## 💬 Conversations")
    st.caption(
        "Consultez les échanges entre les étudiants et ELA. "
        "Sélectionnez une conversation pour la lire en détail."
    )
    st.divider()

    # If a thread is selected, show the reader
    selected_id = st.session_state.get("selected_thread_id")
    if selected_id:
        _render_conversation_reader(selected_id)
        return

    # Otherwise, show list with filters
    user_filter, date_from, date_to, type_filter = _render_filters()

    st.divider()

    threads = _load_threads(
        user_filter=user_filter,
        date_from=date_from,
        date_to=date_to,
        type_filter=type_filter,
    )

    st.caption(f"{len(threads)} conversation(s) trouvée(s)")
    _render_thread_list(threads)
