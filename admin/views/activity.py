# admin/pages/activity.py
"""
Activity monitoring dashboard for the ELA backoffice.

Provides:
- KPI cards: total users, conversations, messages today.
- Daily message volume chart (Plotly).
- Activity type breakdown: Chat vs Quiz vs Code Workshop.
- Per-user quota consumption with progress bars.
- Filterable by date range and user.

Data sources: tables `users`, `threads`, `steps` (Chainlit schema).
"""

import os
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
import psycopg2
import psycopg2.extras


# ---------------------------------------------------------------------------
# Database helper
# ---------------------------------------------------------------------------

def _get_conn():
    """Return a psycopg2 connection using Streamlit secrets or env var.

    Tries st.secrets["ADMIN_DATABASE_URL"] first, then falls back to the
    environment variable.  The URL must use the synchronous driver
    (postgresql://..., NOT postgresql+asyncpg://...).
    """
    db_url = st.secrets.get("ADMIN_DATABASE_URL") or os.getenv("ADMIN_DATABASE_URL")
    if not db_url:
        st.error("❌ `ADMIN_DATABASE_URL` non configurée (secrets.toml ou env).")
        st.stop()
    return psycopg2.connect(db_url)


def _query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Execute a read query and return a DataFrame.

    Args:
        sql: SQL query with %(name)s-style placeholders.
        params: Dict of parameter values.

    Returns:
        pandas DataFrame of results (empty if no rows).
    """
    try:
        conn = _get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or {})
            rows = cur.fetchall()
        conn.close()
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur DB : {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Data queries
# ---------------------------------------------------------------------------

def _fetch_kpis() -> dict:
    """Fetch top-level KPI values.

    Returns:
        Dict with keys: active_users, total_threads, messages_today,
        messages_yesterday.
    """
    df = _query("""
        SELECT
            (SELECT COUNT(*) FROM users WHERE is_active = TRUE)
                AS active_users,
            (SELECT COUNT(*) FROM threads)
                AS total_threads,
            (SELECT COUNT(*) FROM steps
             WHERE type = 'user_message'
             AND "createdAt"::date = CURRENT_DATE)
                AS messages_today,
            (SELECT COUNT(*) FROM steps
             WHERE type = 'user_message'
             AND "createdAt"::date = CURRENT_DATE - INTERVAL '1 day')
                AS messages_yesterday
    """)
    if df.empty:
        return {"active_users": 0, "total_threads": 0,
                "messages_today": 0, "messages_yesterday": 0}
    return df.iloc[0].to_dict()


def _fetch_daily_activity(
    start: date, end: date, user: str | None = None,
) -> pd.DataFrame:
    """Fetch daily message counts within a date range.

    Args:
        start: Start date (inclusive).
        end: End date (inclusive).
        user: Optional user identifier to filter on.

    Returns:
        DataFrame with columns: day, msg_count.
    """
    user_clause = ""
    params: dict = {"start": start, "end": end}
    if user:
        user_clause = 'AND t."userIdentifier" = %(user)s'
        params["user"] = user

    return _query(f"""
        SELECT
            s."createdAt"::date AS day,
            COUNT(*) AS msg_count
        FROM steps s
        JOIN threads t ON s."threadId" = t.id
        WHERE s.type = 'user_message'
          AND s."createdAt"::date BETWEEN %(start)s AND %(end)s
          {user_clause}
        GROUP BY day
        ORDER BY day
    """, params)


def _fetch_activity_by_user(start: date, end: date) -> pd.DataFrame:
    """Fetch message counts per user within a date range.

    Args:
        start: Start date (inclusive).
        end: End date (inclusive).

    Returns:
        DataFrame with columns: user_identifier, msg_count.
    """
    return _query("""
        SELECT
            t."userIdentifier" AS user_identifier,
            COUNT(s.id) AS msg_count
        FROM steps s
        JOIN threads t ON s."threadId" = t.id
        WHERE s.type = 'user_message'
          AND s."createdAt"::date BETWEEN %(start)s AND %(end)s
        GROUP BY t."userIdentifier"
        ORDER BY msg_count DESC
    """, {"start": start, "end": end})


def _fetch_activity_type(
    start: date, end: date, user: str | None = None,
) -> pd.DataFrame:
    """Classify threads into Chat / Quiz / Code by thread name pattern.

    Detection heuristic (matches app.py rename conventions):
    - Thread name starts with '🎓' → Quiz
    - Thread name starts with '💻' → Code Workshop
    - Everything else → Chat

    Args:
        start: Start date (inclusive).
        end: End date (inclusive).
        user: Optional user identifier to filter on.

    Returns:
        DataFrame with columns: activity_type, thread_count.
    """
    user_clause = ""
    params: dict = {"start": start, "end": end}
    if user:
        user_clause = 'AND t."userIdentifier" = %(user)s'
        params["user"] = user

    return _query(f"""
        SELECT
            CASE
                WHEN t.name LIKE '🎓%%' THEN 'Quiz'
                WHEN t.name LIKE '💻%%' THEN 'Code'
                ELSE 'Chat'
            END AS activity_type,
            COUNT(DISTINCT t.id) AS thread_count
        FROM threads t
        WHERE t."createdAt"::date BETWEEN %(start)s AND %(end)s
          {user_clause}
        GROUP BY activity_type
        ORDER BY thread_count DESC
    """, params)


def _fetch_daily_by_type(
    start: date, end: date, user: str | None = None,
) -> pd.DataFrame:
    """Fetch daily thread counts broken down by activity type.

    Args:
        start: Start date (inclusive).
        end: End date (inclusive).
        user: Optional user identifier to filter on.

    Returns:
        DataFrame with columns: day, activity_type, thread_count.
    """
    user_clause = ""
    params: dict = {"start": start, "end": end}
    if user:
        user_clause = 'AND t."userIdentifier" = %(user)s'
        params["user"] = user

    return _query(f"""
        SELECT
            t."createdAt"::date AS day,
            CASE
                WHEN t.name LIKE '🎓%%' THEN 'Quiz'
                WHEN t.name LIKE '💻%%' THEN 'Code'
                ELSE 'Chat'
            END AS activity_type,
            COUNT(DISTINCT t.id) AS thread_count
        FROM threads t
        WHERE t."createdAt"::date BETWEEN %(start)s AND %(end)s
          {user_clause}
        GROUP BY day, activity_type
        ORDER BY day
    """, params)


def _fetch_quota_usage() -> pd.DataFrame:
    """Fetch today's quota consumption for all active users.

    Returns:
        DataFrame with columns: identifier, role, level, daily_quota,
        used_today.
    """
    return _query("""
        SELECT
            u.identifier,
            u.role,
            u.level,
            u.daily_quota,
            COALESCE(sq.used_today, 0) AS used_today
        FROM users u
        LEFT JOIN (
            SELECT
                t."userIdentifier",
                COUNT(s.id) AS used_today
            FROM steps s
            JOIN threads t ON s."threadId" = t.id
            WHERE s.type = 'user_message'
              AND s."createdAt"::date = CURRENT_DATE
            GROUP BY t."userIdentifier"
        ) sq ON sq."userIdentifier" = u.identifier
        WHERE u.is_active = TRUE
        ORDER BY u.role, u.identifier
    """)


def _fetch_user_list() -> list[str]:
    """Fetch list of active user identifiers for filtering.

    Returns:
        Sorted list of identifier strings.
    """
    df = _query("""
        SELECT identifier FROM users
        WHERE is_active = TRUE
        ORDER BY identifier
    """)
    if df.empty:
        return []
    return df["identifier"].tolist()


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------

def _render_kpis():
    """Render the top-level KPI metric cards."""
    kpis = _fetch_kpis()

    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Utilisateurs actifs", kpis["active_users"])
    col2.metric("💬 Conversations totales", kpis["total_threads"])

    # Delta: compare today vs yesterday
    today = kpis["messages_today"]
    yesterday = kpis["messages_yesterday"]
    delta = today - yesterday if yesterday else None
    col3.metric("📨 Messages aujourd'hui", today, delta=delta)


def _render_filters() -> tuple[date, date, str | None]:
    """Render date range and optional user filter.

    Returns:
        Tuple of (start_date, end_date, selected_user_or_None).
    """
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        start = st.date_input(
            "Du", value=date.today() - timedelta(days=30),
            key="activity_start",
        )
    with col2:
        end = st.date_input(
            "Au", value=date.today(),
            key="activity_end",
        )
    with col3:
        users = ["Tous"] + _fetch_user_list()
        selected = st.selectbox("Utilisateur", users, key="activity_user")

    user_filter = None if selected == "Tous" else selected
    return start, end, user_filter


def _render_daily_chart(start: date, end: date, user: str | None = None):
    """Render the daily message volume line chart."""
    st.markdown("#### 📈 Volume de messages par jour")

    df = _fetch_daily_activity(start, end, user)
    if df.empty:
        st.info("Aucune activité sur cette période.")
        return

    fig = px.line(
        df, x="day", y="msg_count",
        labels={"day": "Date", "msg_count": "Messages"},
        markers=True,
    )
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_type_breakdown(
    start: date, end: date, user: str | None = None,
):
    """Render the activity type pie chart and stacked area chart."""
    st.markdown("#### 🎯 Répartition par type d'activité")

    df_pie = _fetch_activity_type(start, end, user)
    df_area = _fetch_daily_by_type(start, end, user)

    if df_pie.empty:
        st.info("Aucune conversation sur cette période.")
        return

    col_pie, col_area = st.columns([1, 2])

    with col_pie:
        colors = {"Chat": "#636EFA", "Quiz": "#00CC96", "Code": "#EF553B"}
        fig_pie = px.pie(
            df_pie, names="activity_type", values="thread_count",
            color="activity_type",
            color_discrete_map=colors,
        )
        fig_pie.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_area:
        if not df_area.empty:
            fig_area = px.area(
                df_area, x="day", y="thread_count",
                color="activity_type",
                color_discrete_map=colors,
                labels={"day": "Date", "thread_count": "Conversations",
                        "activity_type": "Type"},
            )
            fig_area.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=10, b=20),
                xaxis_title=None,
            )
            st.plotly_chart(fig_area, use_container_width=True)


def _render_user_ranking(start: date, end: date):
    """Render the per-user message count bar chart."""
    st.markdown("#### 👤 Activité par utilisateur")

    df = _fetch_activity_by_user(start, end)
    if df.empty:
        st.info("Aucune activité utilisateur sur cette période.")
        return

    fig = px.bar(
        df, x="user_identifier", y="msg_count",
        labels={"user_identifier": "Utilisateur", "msg_count": "Messages"},
        color="msg_count",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis_title=None,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_quota_status():
    """Render today's quota consumption per user with progress bars."""
    st.markdown("#### ⏳ Consommation des quotas (aujourd'hui)")

    df = _fetch_quota_usage()
    if df.empty:
        st.info("Aucun utilisateur actif trouvé.")
        return

    for _, row in df.iterrows():
        quota = row["daily_quota"]
        used = int(row["used_today"])
        identifier = row["identifier"]
        role = row["role"]
        level = row["level"]

        role_icon = {"admin": "🔴", "supervisor": "🟠", "student": "🟢"}.get(role, "⚪")
        label = f"{role_icon} **{identifier}** ({level})"

        if quota is None:
            # Unlimited (admin/supervisor)
            st.markdown(f"{label} — {used} messages · _illimité_")
        else:
            ratio = min(used / quota, 1.0) if quota > 0 else 0.0
            status = f"{used}/{quota}"
            if ratio >= 1.0:
                status += " 🔒"

            col_label, col_bar = st.columns([1, 2])
            col_label.markdown(label)
            col_bar.progress(ratio, text=status)


# ---------------------------------------------------------------------------
# Main page entry point
# ---------------------------------------------------------------------------

def show_activity_page():
    """Render the full activity monitoring dashboard."""
    st.markdown("## 📊 Monitoring d'Activité")
    st.caption("Suivi de l'utilisation d'ELA AI en temps réel.")
    st.divider()

    # KPIs
    _render_kpis()
    st.divider()

    # Filters
    start, end, user_filter = _render_filters()

    if start > end:
        st.warning("La date de début doit être antérieure à la date de fin.")
        return

    st.divider()

    # Main charts in tabs
    tab_volume, tab_types, tab_users, tab_quotas = st.tabs([
        "📈 Volume",
        "🎯 Types",
        "👤 Utilisateurs",
        "⏳ Quotas",
    ])

    with tab_volume:
        _render_daily_chart(start, end, user_filter)

    with tab_types:
        _render_type_breakdown(start, end, user_filter)

    with tab_users:
        _render_user_ranking(start, end)

    with tab_quotas:
        _render_quota_status()
