# admin/pages/users.py
"""
User management page for the ELA backoffice.

Provides full CRUD operations:
- List all users with status indicators
- Create new users (with bcrypt hashing)
- Edit role, level, quota, active status
- Soft-delete (deactivate) users
"""

import streamlit as st
import bcrypt
import pandas as pd
from db import execute_query


# --- Constants ---
ROLES = ["student", "supervisor", "admin"]
LEVELS = ["M1", "M2", "ALL"]


def _hash_password(password: str) -> str:
    """Generate a bcrypt hash from a plaintext password.

    Args:
        password: Plaintext password.

    Returns:
        Bcrypt hash string.
    """
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def _load_users() -> list:
    """Fetch all users from the database.

    Returns:
        List of user dicts ordered by role then identifier.
    """
    return execute_query(
        """
        SELECT
            id,
            identifier,
            role,
            level,
            is_active,
            daily_quota,
            last_login,
            password_hash IS NOT NULL AS has_password,
            "createdAt" AS created_at
        FROM users
        ORDER BY
            CASE role WHEN 'admin' THEN 0 WHEN 'supervisor' THEN 1 ELSE 2 END,
            identifier
        """
    )


def _show_users_table(users: list):
    """Display the users table with status indicators.

    Args:
        users: List of user dicts from _load_users().
    """
    if not users:
        st.info("Aucun utilisateur en base.")
        return

    df = pd.DataFrame(users)

    # Format for display
    df["statut"] = df["is_active"].map({True: "✅ Actif", False: "❌ Inactif"})
    df["mot de passe"] = df["has_password"].map({True: "🔒 Défini", False: "⚠️ Absent"})
    df["quota"] = df["daily_quota"].apply(lambda x: "♾️ Illimité" if x is None else f"{x}/jour")
    df["dernière connexion"] = df["last_login"].apply(
        lambda x: str(x)[:16] if x else "Jamais"
    )

    display_df = df[[
        "identifier", "role", "level", "statut",
        "quota", "mot de passe", "dernière connexion"
    ]].rename(columns={
        "identifier": "Identifiant",
        "role": "Rôle",
        "level": "Niveau",
        "statut": "Statut",
        "quota": "Quota",
        "mot de passe": "Mot de passe",
        "dernière connexion": "Dernière connexion",
    })

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def _show_create_form():
    """Display the user creation form."""
    st.markdown("#### ➕ Créer un utilisateur")

    with st.form("create_user_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            new_identifier = st.text_input("Identifiant (login)")
            new_password = st.text_input("Mot de passe", type="password")
        with col2:
            new_role = st.selectbox("Rôle", ROLES, index=0)
            new_level = st.selectbox("Niveau", LEVELS, index=0)

        col3, col4 = st.columns(2)
        with col3:
            new_quota = st.number_input(
                "Quota journalier",
                min_value=0, max_value=500, value=50,
                help="0 = illimité (pour les admins)"
            )

        submitted = st.form_submit_button("Créer l'utilisateur", use_container_width=True)

    if submitted:
        if not new_identifier or not new_password:
            st.error("L'identifiant et le mot de passe sont obligatoires.")
            return

        if len(new_password) < 6:
            st.error("Le mot de passe doit contenir au moins 6 caractères.")
            return

        # Check if identifier already exists
        existing = execute_query(
            "SELECT id FROM users WHERE identifier = %(id)s",
            {"id": new_identifier},
        )
        if existing:
            st.error(f"L'identifiant '{new_identifier}' existe déjà.")
            return

        # Insert new user
        password_hash = _hash_password(new_password)
        quota_value = None if new_quota == 0 else new_quota

        execute_query(
            """
            INSERT INTO users (id, identifier, "createdAt", password_hash, role, level, is_active, daily_quota)
            VALUES (gen_random_uuid(), %(identifier)s, NOW()::TEXT, %(hash)s, %(role)s, %(level)s, TRUE, %(quota)s)
            """,
            {
                "identifier": new_identifier,
                "hash": password_hash,
                "role": new_role,
                "level": new_level,
                "quota": quota_value,
            },
            fetch=False,
        )
        st.success(f"Utilisateur **{new_identifier}** créé avec succès.")
        st.rerun()


def _show_edit_section(users: list):
    """Display the user edit/deactivate section.

    Args:
        users: List of user dicts from _load_users().
    """
    st.markdown("#### ✏️ Modifier un utilisateur")

    # User selector (exclude current admin to prevent self-lockout)
    admin_user = st.session_state.get("admin_user", "")
    user_options = {u["identifier"]: u for u in users}
    selected_id = st.selectbox(
        "Sélectionner un utilisateur",
        options=list(user_options.keys()),
        format_func=lambda x: f"{x} ({user_options[x]['role']})",
    )

    if not selected_id:
        return

    user = user_options[selected_id]
    is_self = selected_id == admin_user

    with st.form("edit_user_form"):
        col1, col2 = st.columns(2)
        with col1:
            edit_role = st.selectbox(
                "Rôle",
                ROLES,
                index=ROLES.index(user["role"]) if user["role"] in ROLES else 0,
                disabled=is_self,
                help="Vous ne pouvez pas modifier votre propre rôle." if is_self else None,
            )
            edit_level = st.selectbox(
                "Niveau",
                LEVELS,
                index=LEVELS.index(user["level"]) if user["level"] in LEVELS else 0,
            )
        with col2:
            current_quota = user["daily_quota"] if user["daily_quota"] is not None else 0
            edit_quota = st.number_input(
                "Quota journalier",
                min_value=0, max_value=500,
                value=current_quota,
                help="0 = illimité",
            )
            edit_active = st.checkbox(
                "Compte actif",
                value=user["is_active"],
                disabled=is_self,
                help="Vous ne pouvez pas vous désactiver." if is_self else None,
            )

        st.markdown("---")
        new_password = st.text_input(
            "Nouveau mot de passe (laisser vide pour ne pas changer)",
            type="password",
        )

        submitted = st.form_submit_button("Enregistrer les modifications", use_container_width=True)

    if submitted:
        quota_value = None if edit_quota == 0 else edit_quota
        role_value = user["role"] if is_self else edit_role
        active_value = user["is_active"] if is_self else edit_active

        # Build update query dynamically
        update_fields = {
            "role": role_value,
            "level": edit_level,
            "daily_quota": quota_value,
            "is_active": active_value,
        }

        if new_password:
            if len(new_password) < 6:
                st.error("Le mot de passe doit contenir au moins 6 caractères.")
                return
            update_fields["password_hash"] = _hash_password(new_password)

        set_clause = ", ".join(f'"{k}" = %({k})s' for k in update_fields)
        update_fields["user_id"] = user["id"]

        execute_query(
            f'UPDATE users SET {set_clause} WHERE id = %(user_id)s',
            update_fields,
            fetch=False,
        )
        st.success(f"Utilisateur **{selected_id}** mis à jour.")
        st.rerun()


# --- Main page function ---
def show_users_page():
    """Render the full users management page."""
    st.markdown("## 👥 Gestion des utilisateurs")

    users = _load_users()

    # Users table
    _show_users_table(users)

    st.divider()

    # Tabs for create / edit
    tab_create, tab_edit = st.tabs(["➕ Créer", "✏️ Modifier"])

    with tab_create:
        _show_create_form()

    with tab_edit:
        if users:
            _show_edit_section(users)
        else:
            st.info("Aucun utilisateur à modifier.")
