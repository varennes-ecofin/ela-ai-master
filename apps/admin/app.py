# admin/app.py

# cd admin
# streamlit run app.py
"""
ELA AI Backoffice — Main entry point.

Streamlit multipage app with admin authentication.
Pages are auto-discovered from the pages/ directory.
"""

import bcrypt
import streamlit as st
from db import execute_query

# --- Page config (must be first Streamlit call) ---
st.set_page_config(
    page_title="ELA AI — Backoffice",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def verify_admin_login(username: str, password: str) -> bool:
    """Verify admin credentials against the users table.

    Only users with role 'admin' can access the backoffice.

    Args:
        username: The login identifier.
        password: The plaintext password to verify.

    Returns:
        True if credentials are valid and user has admin role.
    """
    rows = execute_query(
        """
        SELECT identifier, password_hash, role, is_active
        FROM users
        WHERE identifier = %(username)s
        """,
        {"username": username},
    )

    if not rows:
        return False

    user = rows[0]

    # Must be active and admin
    if not user["is_active"] or user["role"] != "admin":
        return False

    # Verify bcrypt hash
    if not user["password_hash"]:
        return False

    return bcrypt.checkpw(
        password.encode("utf-8"),
        user["password_hash"].encode("utf-8"),
    )


def show_login_form():
    """Display the admin login form."""
    st.markdown("## 🔐 ELA AI — Backoffice")
    st.markdown("Accès réservé aux administrateurs.")

    with st.form("login_form"):
        username = st.text_input("Identifiant")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter", use_container_width=True)

    if submitted:
        if verify_admin_login(username, password):
            st.session_state["authenticated"] = True
            st.session_state["admin_user"] = username
            st.rerun()
        else:
            st.error("Identifiants invalides ou accès non autorisé.")


def show_sidebar():
    """Display the sidebar with navigation and logout."""
    with st.sidebar:
        st.markdown("### 🛠️ Backoffice ELA")
        st.markdown(f"Connecté : **{st.session_state.get('admin_user', '?')}**")
        st.divider()

        if st.button("🚪 Déconnexion", use_container_width=True):
            st.session_state.clear()
            st.rerun()


def show_home():
    """Display the home dashboard with quick stats."""
    st.markdown("## 📊 Tableau de bord")

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)

    users_stats = execute_query(
        """
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE is_active = TRUE) AS active,
            COUNT(*) FILTER (WHERE role = 'student') AS students,
            COUNT(*) FILTER (WHERE role = 'admin') AS admins
        FROM users
        """
    )

    thread_count = execute_query("SELECT COUNT(*) AS total FROM threads")

    if users_stats:
        s = users_stats[0]
        col1.metric("Utilisateurs", s["total"])
        col2.metric("Actifs", s["active"])
        col3.metric("Étudiants", s["students"])

    if thread_count:
        col4.metric("Conversations", thread_count[0]["total"])

    st.divider()

    # Recent activity preview
    st.markdown("### 📈 Activité récente")
    recent = execute_query(
        """
        SELECT
            u.identifier,
            t.name AS thread_name,
            t."createdAt" AS created_at
        FROM threads t
        JOIN users u ON t."userId" = u.id
        ORDER BY t."createdAt" DESC
        LIMIT 10
        """
    )

    if recent:
        st.dataframe(recent, use_container_width=True, hide_index=True)
    else:
        st.info("Aucune activité enregistrée.")


# --- Main app logic ---
def main():
    """Main entry point — routes to login or dashboard."""
    if not st.session_state.get("authenticated"):
        show_login_form()
        return

    show_sidebar()

    # Navigation (Phase 5: added RAG management)

    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Accueil",
            "👥 Utilisateurs",
            "📚 Base RAG",
            "📊 Activité",
            "🎓 Pédagogie",
            "⭐ Feedbacks",
            "💬 Conversations",
        ],
        label_visibility="collapsed",
    )

    if page == "🏠 Accueil":
        show_home()
    elif page == "👥 Utilisateurs":
        from views.users import show_users_page

        show_users_page()
    elif page == "📚 Base RAG":
        from views.rag_management import show_rag_management_page

        show_rag_management_page()
    elif page == "📊 Activité":
        from views.activity import show_activity_page

        show_activity_page()
    elif page == "🎓 Pédagogie":
        from views.pedagogy import show_pedagogy_page

        show_pedagogy_page()
    elif page == "⭐ Feedbacks":
        from views.feedbacks import show_feedbacks_page

        show_feedbacks_page()
    elif page == "💬 Conversations":
        from views.conversations import show_conversations_page

        show_conversations_page()


if __name__ == "__main__":
    main()
