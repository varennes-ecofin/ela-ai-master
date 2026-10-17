# admin/pages/rag_management.py
"""
RAG Management page for the ELA backoffice.

Allows the admin to:
- Browse the latex/ directory tree.
- Upload .tex files with level/course assignment.
- Delete .tex files from the knowledge base.
- Run the ingestion (`ela-ingest`): dry run, incremental update or full rebuild.
- View ingestion logs and statistics.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Configuration — paths are relative to the project root.
# In Docker the project is installed under /app ; locally this file lives in
# apps/admin/views/ so the root is three levels up.
# ---------------------------------------------------------------------------

_DOCKER_BASE = Path("/app")
_LOCAL_BASE = Path(__file__).resolve().parents[3]  # apps/admin/views/x.py → project root

if (_DOCKER_BASE / "pyproject.toml").exists():
    BASE_DIR = _DOCKER_BASE
else:
    BASE_DIR = _LOCAL_BASE

LATEX_DIR = BASE_DIR / "data" / "latex"
# Ingestion runs as a module of the installed `ela` package (same as `ela-ingest`).
# `-I` (isolated mode) keeps the current directory out of sys.path so a stale
# `ela/` folder at the project root can never shadow the installed package.
# `-X utf8` forces UTF-8 for stdout/stderr (Windows uses cp1252 when the output
# is captured, which breaks the emojis of the ingestion report).
INGEST_CMD = [sys.executable, "-I", "-X", "utf8", "-m", "ela.ingestion.pipeline"]

VALID_LEVELS = ["M1", "M2", "Commun"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_latex_tree() -> dict:
    """Scan the latex/ directory and return a nested dict of its structure.

    Returns:
        Dict with structure:
        {
            "M1": {
                "Series_Temporelles": ["cours_var.tex", "cours_arima.tex"],
                "Econometrie_Base": ["cours_mco.tex"],
            },
            "M2": { ... },
            "Commun": { ... },
        }
        Plus a "_root_files" key for .tex files directly under latex/.
    """
    tree: dict = {}

    if not LATEX_DIR.exists():
        return tree

    for entry in sorted(LATEX_DIR.iterdir()):
        if entry.is_file() and entry.suffix == ".tex":
            tree.setdefault("_root_files", []).append(entry.name)
        elif entry.is_dir():
            level_name = entry.name
            level_dict: dict = {}
            for course_entry in sorted(entry.iterdir()):
                if course_entry.is_dir():
                    tex_files = sorted(f.name for f in course_entry.iterdir() if f.suffix == ".tex")
                    if tex_files:
                        level_dict[course_entry.name] = tex_files
                elif course_entry.suffix == ".tex":
                    level_dict.setdefault("_loose_files", []).append(
                        course_entry.name,
                    )
            if level_dict:
                tree[level_name] = level_dict

    return tree


def _count_tex_files() -> int:
    """Count total .tex files under latex/."""
    if not LATEX_DIR.exists():
        return 0
    return sum(1 for _ in LATEX_DIR.rglob("*.tex"))


def _get_store_stats() -> dict:
    """Backend and number of chunks of the knowledge base (``RETRIEVAL_BACKEND``)."""
    stats = {"backend": "?", "count": None, "error": ""}
    try:
        from ela.config import get_settings
        from ela.retrieval.store import build_store

        settings = get_settings()
        stats["backend"] = settings.retrieval_backend
        stats["count"] = build_store(settings).count()
    except Exception as exc:  # store unreachable (no DB, no pgvector, no chroma dir...)
        stats["error"] = f"{exc.__class__.__name__}: {exc}"
    return stats


def _get_existing_courses(level: str) -> list[str]:
    """List existing course folders for a given level.

    Args:
        level: Academic level (M1, M2, Commun).

    Returns:
        Sorted list of course folder names.
    """
    level_dir = LATEX_DIR / level
    if not level_dir.exists():
        return []
    return sorted(d.name for d in level_dir.iterdir() if d.is_dir())


def _ensure_directory(path: Path):
    """Create directory (and parents) if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _run_ingestion(mode: str = "incremental") -> tuple[bool, str, float]:
    """Run ``ela-ingest`` as a subprocess.

    Args:
        mode: ``dry_run`` (parse only), ``incremental`` (default: only new or
            changed chunks are embedded) or ``full`` (store cleared then rebuilt).

    Returns:
        Tuple of (success, log_output, duration_seconds).
    """
    cmd = [*INGEST_CMD, "--source-dir", str(LATEX_DIR)]
    if mode == "dry_run":
        cmd.append("--dry-run")
    elif mode == "full":
        cmd.append("--full")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,  # the first full run downloads the models and embeds everything
            cwd=str(BASE_DIR),
        )
        duration = time.time() - start
        log = result.stdout + ("\n" + result.stderr if result.stderr else "")
        if result.returncode != 0:
            return False, f"❌ Erreur (code {result.returncode}):\n{log}", duration
        return True, log, duration
    except subprocess.TimeoutExpired:
        return False, "❌ Timeout : l'ingestion a dépassé 60 minutes.", time.time() - start
    except Exception as e:
        return False, f"❌ Exception : {e}", time.time() - start


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------


def _render_tree_view():
    """Render the latex/ directory tree as expandable sections."""
    st.markdown("### 📂 Arborescence `latex/`")

    tree = _get_latex_tree()

    if not tree:
        st.warning(
            f"Le dossier `{LATEX_DIR}` est vide ou introuvable. "
            "Créez la structure `latex/M1/NomCours/fichier.tex` pour commencer."
        )
        return

    # Root-level .tex files (legacy)
    root_files = tree.pop("_root_files", [])
    if root_files:
        with st.expander("⚠️ Fichiers racine (legacy)", expanded=False):
            st.caption(
                "Ces fichiers seront ingérés avec `level=legacy`. "
                "Déplacez-les dans un sous-dossier pour un cloisonnement correct."
            )
            for f in root_files:
                st.text(f"  📄 {f}")

    # Level folders
    for level_name in VALID_LEVELS:
        if level_name not in tree:
            continue
        courses = tree[level_name]

        level_icon = {"M1": "🟢", "M2": "🔵", "Commun": "🟡"}.get(level_name, "📁")

        with st.expander(f"{level_icon} {level_name}", expanded=True):
            loose = courses.pop("_loose_files", [])
            if loose:
                st.caption("Fichiers hors cours :")
                for f in loose:
                    st.text(f"  📄 {f}")

            for course_name, files in sorted(courses.items()):
                st.markdown(f"**📚 {course_name}** ({len(files)} fichiers)")
                for f in files:
                    file_path = LATEX_DIR / level_name / course_name / f
                    size_kb = round(file_path.stat().st_size / 1024, 1) if file_path.exists() else "?"
                    col1, col2 = st.columns([4, 1])
                    col1.text(f"  📄 {f}  ({size_kb} Ko)")
                    if col2.button(
                        "🗑️",
                        key=f"del_{level_name}_{course_name}_{f}",
                        help=f"Supprimer {f}",
                    ):
                        st.session_state["confirm_delete"] = {
                            "level": level_name,
                            "course": course_name,
                            "file": f,
                        }

    # Unknown level folders
    known = set(VALID_LEVELS) | {"_root_files"}
    for unknown_level in sorted(set(tree.keys()) - known):
        with st.expander(f"❓ {unknown_level} (niveau inconnu)", expanded=False):
            st.warning(
                f"Ce dossier n'est pas dans les niveaux attendus "
                f"({', '.join(VALID_LEVELS)}). Son contenu sera ignoré par l'ingestion."
            )

    # Handle delete confirmation
    if "confirm_delete" in st.session_state:
        info = st.session_state["confirm_delete"]
        target = LATEX_DIR / info["level"] / info["course"] / info["file"]
        st.warning(f"Confirmer la suppression de **{info['file']}** ({info['level']}/{info['course']}) ?")
        col_y, col_n = st.columns(2)
        if col_y.button("✅ Confirmer", key="confirm_del_yes"):
            try:
                target.unlink()
                st.success(f"Fichier supprimé : {info['file']}")
                # Clean empty directory
                parent = target.parent
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()
            except Exception as e:
                st.error(f"Erreur : {e}")
            del st.session_state["confirm_delete"]
            st.rerun()
        if col_n.button("❌ Annuler", key="confirm_del_no"):
            del st.session_state["confirm_delete"]
            st.rerun()


def _render_upload_section():
    """Render the .tex file upload form.

    Level and course selectors live OUTSIDE the form so that changing
    the level triggers an immediate Streamlit rerun, refreshing the
    course dropdown dynamically.
    """
    st.markdown("### ⬆️ Upload de fichiers `.tex`")

    # --- Selectors outside the form (reactive to changes) ---
    level = st.selectbox("Niveau académique", VALID_LEVELS, key="upload_level")

    existing_courses = _get_existing_courses(level)
    course_options = existing_courses + ["➕ Nouveau cours..."]
    course_choice = st.selectbox("Cours", course_options, key="upload_course")

    new_course_name = ""
    if course_choice == "➕ Nouveau cours...":
        new_course_name = st.text_input(
            "Nom du nouveau cours",
            placeholder="Ex: Series_Temporelles",
            help="Utiliser des underscores, pas d'espaces ni d'accents.",
            key="upload_new_course",
        )

    # --- Form for file upload + submit only ---
    with st.form("upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Fichiers `.tex`",
            type=["tex"],
            accept_multiple_files=True,
            help="Vous pouvez sélectionner plusieurs fichiers à la fois.",
        )

        submitted = st.form_submit_button(
            "📤 Uploader",
            use_container_width=True,
        )

    if submitted and uploaded_files:
        # Determine target course name
        course_name = new_course_name.strip() if course_choice == "➕ Nouveau cours..." else course_choice

        if not course_name:
            st.error("Veuillez saisir un nom de cours.")
            return

        # Sanitize course name (basic)
        course_name = course_name.replace(" ", "_")

        target_dir = LATEX_DIR / level / course_name
        _ensure_directory(target_dir)

        uploaded_count = 0
        for uploaded_file in uploaded_files:
            target_path = target_dir / uploaded_file.name

            # Warn if overwriting
            if target_path.exists():
                st.warning(f"⚠️ `{uploaded_file.name}` existe déjà et sera écrasé.")

            try:
                target_path.write_bytes(uploaded_file.getvalue())
                uploaded_count += 1
            except Exception as e:
                st.error(f"Erreur pour `{uploaded_file.name}` : {e}")

        if uploaded_count:
            st.success(f"✅ {uploaded_count} fichier(s) uploadé(s) dans `latex/{level}/{course_name}/`")
            st.info(
                "💡 Lancez une **mise à jour incrémentale** (onglet Mise à jour) "
                "pour que ces fichiers soient pris en compte par ELA."
            )
            st.rerun()


def _render_rebuild_section():
    """Render the ingestion controls (dry run / incremental / full)."""
    st.markdown("### 🔄 Mise à jour de la base de connaissances")

    col1, col2, col3 = st.columns(3)
    col1.metric("Fichiers .tex", _count_tex_files())
    stats = _get_store_stats()
    col2.metric("Chunks indexés", stats["count"] if stats["count"] is not None else "—")
    col3.metric("Backend", stats["backend"])
    if stats["error"]:
        st.warning(f"Store inaccessible : {stats['error']}")

    st.divider()
    st.caption(
        "**Mise à jour incrémentale** : seuls les chunks nouveaux ou modifiés sont vectorisés, "
        "les chunks disparus sont supprimés ; les sessions en cours basculent sur la base mise à jour "
        "à leur prochaine question. **Reconstruction complète** : vide le store puis réingère tout."
    )

    col_dry, col_inc, col_full = st.columns(3)
    if col_dry.button("🔍 Dry run", use_container_width=True, help="Analyse les fichiers et affiche le rapport."):
        st.session_state["rebuild_mode"] = "dry_run"
    if col_inc.button(
        "⚡ Mise à jour incrémentale",
        use_container_width=True,
        type="primary",
        help="Vectorise uniquement ce qui a changé.",
    ):
        st.session_state["rebuild_mode"] = "incremental"
    if col_full.button("🚀 Reconstruction complète", use_container_width=True, help="Vide le store et réingère tout."):
        st.session_state["rebuild_mode"] = "confirm"

    if st.session_state.get("rebuild_mode") == "confirm":
        st.warning("⚠️ Cette opération va **recalculer tous les embeddings**. Cela peut prendre plusieurs minutes.")
        col_y, col_n = st.columns(2)
        if col_y.button("✅ Confirmer", key="confirm_rebuild"):
            st.session_state["rebuild_mode"] = "full"
            st.rerun()
        if col_n.button("❌ Annuler", key="cancel_rebuild"):
            del st.session_state["rebuild_mode"]
            st.rerun()

    mode = st.session_state.get("rebuild_mode")
    if mode in ("dry_run", "incremental", "full"):
        label = {"dry_run": "Dry run", "incremental": "Mise à jour incrémentale", "full": "Reconstruction complète"}[
            mode
        ]
        with st.spinner(f"⏳ {label} en cours... Patientez."):
            success, log, duration = _run_ingestion(mode)

        duration_str = f"{duration:.1f}s"
        if success:
            st.success(f"✅ {label} terminé en {duration_str}.")
        else:
            st.error(f"❌ {label} échoué après {duration_str}.")
        with st.expander("📋 Log d'ingestion", expanded=True):
            st.code(log, language="text")
        st.session_state.setdefault("ingestion_history", []).append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": label,
                "success": success,
                "duration": duration_str,
            }
        )
        if "rebuild_mode" in st.session_state:
            del st.session_state["rebuild_mode"]


def _render_history_section():
    """Render the ingestion history from the current session."""
    history = st.session_state.get("ingestion_history", [])
    if not history:
        return

    st.markdown("### 📜 Historique de la session")
    for entry in reversed(history):
        icon = "✅" if entry["success"] else "❌"
        st.text(f"{icon} [{entry['timestamp']}] {entry['mode']} — {entry['duration']}")


# ---------------------------------------------------------------------------
# Main page entry point
# ---------------------------------------------------------------------------


def show_rag_management_page():
    """Render the full RAG management page."""
    st.markdown("## 📚 Gestion de la Base de Connaissances (RAG)")
    st.caption(f"Dossier source : `{LATEX_DIR}`")
    st.divider()

    # Tabs for cleaner UX
    tab_tree, tab_upload, tab_rebuild = st.tabs(
        [
            "📂 Arborescence",
            "⬆️ Upload",
            "🔄 Mise à jour",
        ]
    )

    with tab_tree:
        _render_tree_view()

    with tab_upload:
        _render_upload_section()

    with tab_rebuild:
        _render_rebuild_section()
        _render_history_section()
