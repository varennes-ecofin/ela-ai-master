# admin/pages/rag_management.py
"""
RAG Management page for the ELA backoffice.

Allows the admin to:
- Browse the latex/ directory tree.
- Upload .tex files with level/course assignment.
- Delete .tex files from the knowledge base.
- Trigger a full ChromaDB rebuild (ingest.py) with atomic swap.
- View ingestion logs and statistics.
"""

import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import streamlit as st

# ---------------------------------------------------------------------------
# Configuration — paths are relative to the Docker container layout.
# In Docker:  /app/latex, /app/chroma_db, /app/ingest.py
# In local:   ../latex,   ../chroma_db,   ../ingest.py
# ---------------------------------------------------------------------------

# Resolve base path: check Docker layout first, then fall back to local dev
_DOCKER_BASE = Path("/app")
_LOCAL_BASE = Path(__file__).resolve().parent.parent.parent  # admin/../ = project root

if (_DOCKER_BASE / "ingest.py").exists():
    BASE_DIR = _DOCKER_BASE
else:
    BASE_DIR = _LOCAL_BASE

LATEX_DIR = BASE_DIR / "data" / "latex"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
INGEST_SCRIPT = BASE_DIR / "ingest.py"

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
                    tex_files = sorted(
                        f.name for f in course_entry.iterdir()
                        if f.suffix == ".tex"
                    )
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


def _get_chroma_stats() -> dict:
    """Get basic stats about the current ChromaDB."""
    stats = {"exists": CHROMA_DIR.exists(), "size_mb": 0.0, "file_count": 0}
    if stats["exists"]:
        files = [f for f in CHROMA_DIR.rglob("*") if f.is_file()]
        stats["size_mb"] = round(sum(f.stat().st_size for f in files) / (1024 * 1024), 1)
        stats["file_count"] = len(files)
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
    return sorted(
        d.name for d in level_dir.iterdir() if d.is_dir()
    )


def _ensure_directory(path: Path):
    """Create directory (and parents) if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _run_ingestion(dry_run: bool = False) -> tuple[bool, str, float]:
    """Execute ingest.py as a subprocess with optional atomic swap.

    Strategy (from GUIDE Phase 5):
    1. Ingest into chroma_db_new/
    2. Atomic swap: mv chroma_db → chroma_db_old, mv chroma_db_new → chroma_db
    3. Cleanup: rm chroma_db_old after validation.

    Args:
        dry_run: If True, run --dry-run (parse only, no embeddings).

    Returns:
        Tuple of (success: bool, log_output: str, duration_seconds: float).
    """
    if not INGEST_SCRIPT.exists():
        return False, f"❌ Script introuvable : {INGEST_SCRIPT}", 0.0

    new_db = BASE_DIR / "data" / "chroma_db_new"
    old_db = BASE_DIR / "data" / "chroma_db_old"

    cmd = [
        "python", str(INGEST_SCRIPT),
        "--source-dir", str(LATEX_DIR),
        "--db-path", str(new_db) if not dry_run else str(CHROMA_DIR),
    ]
    if dry_run:
        cmd.append("--dry-run")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
            cwd=str(BASE_DIR),
        )
        duration = time.time() - start
        log = result.stdout + ("\n" + result.stderr if result.stderr else "")

        if result.returncode != 0:
            return False, f"❌ Erreur (code {result.returncode}):\n{log}", duration

        # Dry-run: no swap needed
        if dry_run:
            return True, log, duration

        # Atomic swap
        if not new_db.exists():
            return False, f"❌ chroma_db_new/ non créé par ingest.py.\n{log}", duration

        try:
            # Remove leftover old_db if exists
            if old_db.exists():
                shutil.rmtree(old_db)

            # Swap: current → old, new → current
            if CHROMA_DIR.exists():
                CHROMA_DIR.rename(old_db)
            new_db.rename(CHROMA_DIR)

            # Cleanup old
            if old_db.exists():
                shutil.rmtree(old_db)

            log += "\n\n✅ Swap atomique réussi : nouvelle base active."
        except Exception as swap_err:
            log += f"\n\n❌ Erreur lors du swap : {swap_err}"
            # Try to rollback
            if old_db.exists() and not CHROMA_DIR.exists():
                old_db.rename(CHROMA_DIR)
                log += "\n⚠️ Rollback effectué (ancienne base restaurée)."
            return False, log, duration

        return True, log, duration

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return False, "❌ Timeout : l'ingestion a dépassé 10 minutes.", duration
    except Exception as e:
        duration = time.time() - start
        return False, f"❌ Exception : {e}", duration


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
                        "🗑️", key=f"del_{level_name}_{course_name}_{f}",
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
        st.warning(
            f"Confirmer la suppression de **{info['file']}** "
            f"({info['level']}/{info['course']}) ?"
        )
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
            "📤 Uploader", use_container_width=True,
        )

    if submitted and uploaded_files:
        # Determine target course name
        course_name = (
            new_course_name.strip()
            if course_choice == "➕ Nouveau cours..."
            else course_choice
        )

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
                st.warning(
                    f"⚠️ `{uploaded_file.name}` existe déjà et sera écrasé."
                )

            try:
                target_path.write_bytes(uploaded_file.getvalue())
                uploaded_count += 1
            except Exception as e:
                st.error(f"Erreur pour `{uploaded_file.name}` : {e}")

        if uploaded_count:
            st.success(
                f"✅ {uploaded_count} fichier(s) uploadé(s) dans "
                f"`latex/{level}/{course_name}/`"
            )
            st.info(
                "💡 N'oubliez pas de **reconstruire la base vectorielle** "
                "pour que ces fichiers soient pris en compte par ELA."
            )
            st.rerun()


def _render_rebuild_section():
    """Render the ChromaDB rebuild controls."""
    st.markdown("### 🔄 Reconstruction de la base vectorielle")

    # Current stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Fichiers .tex", _count_tex_files())

    chroma_stats = _get_chroma_stats()
    col2.metric(
        "Base ChromaDB",
        f"{chroma_stats['size_mb']} Mo" if chroma_stats["exists"] else "Absente",
    )
    col3.metric(
        "Statut",
        "✅ Active" if chroma_stats["exists"] else "❌ Absente",
    )

    st.divider()

    st.caption(
        "**Stratégie** : L'ingestion crée une nouvelle base (`chroma_db_new/`), "
        "puis effectue un swap atomique. Les sessions ELA existantes ne sont pas "
        "interrompues ; seules les nouvelles sessions chargeront la base mise à jour."
    )

    col_dry, col_full = st.columns(2)

    # Dry run button
    if col_dry.button(
        "🔍 Dry Run (vérifier sans embeddings)",
        use_container_width=True,
        help="Parse les fichiers et affiche le rapport sans générer les vecteurs.",
    ):
        st.session_state["rebuild_mode"] = "dry_run"

    # Full rebuild button
    if col_full.button(
        "🚀 Reconstruire la base",
        use_container_width=True,
        type="primary",
        help="Relance l'ingestion complète avec embeddings et swap atomique.",
    ):
        st.session_state["rebuild_mode"] = "confirm"

    # Confirmation for full rebuild
    if st.session_state.get("rebuild_mode") == "confirm":
        st.warning(
            "⚠️ Cette opération va **recalculer tous les embeddings** et "
            "remplacer la base vectorielle actuelle. "
            "Cela peut prendre plusieurs minutes."
        )
        col_y, col_n = st.columns(2)
        if col_y.button("✅ Confirmer", key="confirm_rebuild"):
            st.session_state["rebuild_mode"] = "full"
            st.rerun()
        if col_n.button("❌ Annuler", key="cancel_rebuild"):
            del st.session_state["rebuild_mode"]
            st.rerun()

    # Execute rebuild
    mode = st.session_state.get("rebuild_mode")
    if mode in ("dry_run", "full"):
        is_dry = mode == "dry_run"
        label = "Dry run" if is_dry else "Ingestion complète"

        with st.spinner(f"⏳ {label} en cours... Patientez."):
            success, log, duration = _run_ingestion(dry_run=is_dry)

        # Display results
        duration_str = f"{duration:.1f}s"
        if success:
            st.success(f"✅ {label} terminé en {duration_str}.")
        else:
            st.error(f"❌ {label} échoué après {duration_str}.")

        with st.expander("📋 Log d'ingestion", expanded=True):
            st.code(log, language="text")

        # Store in session for history
        st.session_state.setdefault("ingestion_history", []).append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": label,
            "success": success,
            "duration": duration_str,
        })

        # Reset mode
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
        st.text(
            f"{icon} [{entry['timestamp']}] "
            f"{entry['mode']} — {entry['duration']}"
        )


# ---------------------------------------------------------------------------
# Main page entry point
# ---------------------------------------------------------------------------

def show_rag_management_page():
    """Render the full RAG management page."""
    st.markdown("## 📚 Gestion de la Base de Connaissances (RAG)")
    st.caption(
        f"Dossier source : `{LATEX_DIR}` · "
        f"Base vectorielle : `{CHROMA_DIR}`"
    )
    st.divider()

    # Tabs for cleaner UX
    tab_tree, tab_upload, tab_rebuild = st.tabs([
        "📂 Arborescence",
        "⬆️ Upload",
        "🔄 Reconstruction",
    ])

    with tab_tree:
        _render_tree_view()

    with tab_upload:
        _render_upload_section()

    with tab_rebuild:
        _render_rebuild_section()
        _render_history_section()