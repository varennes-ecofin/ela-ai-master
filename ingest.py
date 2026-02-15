# ingest.py
"""
Multi-level LaTeX ingestion pipeline for ELA AI.

Scans latex/ recursively with the expected structure:
    latex/{level}/{course}/file.tex

Where:
    - level:  M1, M2, or Commun
    - course: folder name (e.g. Series_Temporelles, Econometrie_Base)

Features:
    - Resolves \\include{} and \\input{} before parsing (recursive, cycle-safe).
    - Supports both \\chapter and \\section based documents.
    - Extracts level/course from directory structure.
    - Skips chapter files already pulled in by a master document.

Produces ChromaDB documents with metadata:
    source, level, course, chapter, section, slide_title, type
"""

# Pour vérifier la structure avant l'embedding : python ingest.py --dry-run

import os
import re
import shutil
import argparse
from collections import Counter
from typing import List, Dict, Set, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --- CONFIGURATION ---
SOURCE_DIR = "./data/latex"    # au lieu de "./latex"
DB_PATH = "./data/chroma_db"   # au lieu de "./chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

VALID_LEVELS = {"M1", "M2", "Commun"}


# --- INCLUDE/INPUT RESOLVER ---

# Matches \include{filename} and \input{filename}
# Skips commented-out lines (starting with %)
_INCLUDE_PATTERN = re.compile(
    r'^[ \t]*\\(?:include|input)\{([^}]+)\}',
    re.MULTILINE,
)


def resolve_includes(
    content: str,
    base_dir: str,
    visited: Set[str] | None = None,
    depth: int = 0,
    max_depth: int = 10,
) -> str:
    r"""Recursively resolve \include{} and \input{} in LaTeX content.

    Replaces each directive with the contents of the referenced file.
    Handles cycle detection and depth limits to avoid infinite loops.

    Args:
        content: Raw LaTeX string.
        base_dir: Directory of the current .tex file (for relative paths).
        visited: Set of already-resolved absolute paths (cycle detection).
        depth: Current recursion depth.
        max_depth: Maximum recursion depth before stopping.

    Returns:
        LaTeX string with all includes resolved inline.
    """
    if visited is None:
        visited = set()

    if depth > max_depth:
        print(f"      ⚠️  Profondeur max ({max_depth}) atteinte, arrêt de la résolution.")
        return content

    # First strip commented lines to avoid resolving commented-out includes
    lines = content.split('\n')
    '\n'.join(
        line for line in lines if not line.lstrip().startswith('%')
    )
    # But we keep the original content and only match on uncommented patterns
    # Actually, let's work on uncommented content for matching,
    # but replace in the original. Simpler: just process full content
    # and skip lines starting with %

    def _replace_include(match: re.Match) -> str:
        # Skip if the line is a comment
        line_start = content.rfind('\n', 0, match.start()) + 1
        line_prefix = content[line_start:match.start()].lstrip()
        if line_prefix.startswith('%'):
            return match.group(0)  # Keep as-is

        ref_name = match.group(1).strip()

        # \include{file} implicitly adds .tex; \input{file} may or may not
        candidates = [ref_name]
        if not ref_name.endswith(".tex"):
            candidates.append(ref_name + ".tex")

        for candidate in candidates:
            ref_path = os.path.normpath(os.path.join(base_dir, candidate))

            if ref_path in visited:
                print(f"      ⚠️  Référence circulaire détectée : {candidate}")
                return f"% [CIRCULAR: {candidate}]"

            if os.path.isfile(ref_path):
                visited.add(ref_path)
                print(f"      📎 Résolution include : {candidate}")

                with open(ref_path, 'r', encoding='utf-8', errors='replace') as f:
                    included_content = f.read()

                # Recurse into the included file
                child_dir = os.path.dirname(ref_path)
                return resolve_includes(
                    included_content, child_dir, visited, depth + 1, max_depth,
                )

        # File not found — leave a comment marker
        print(f"      ⚠️  Fichier inclus introuvable : {ref_name} (cherché dans {base_dir})")
        return f"% [NOT FOUND: {ref_name}]"

    return _INCLUDE_PATTERN.sub(_replace_include, content)


# --- LATEX PARSER ---

class LatexIngestor:
    """Parse .tex files into LangChain Documents with level/course metadata."""

    def __init__(self, source_dir: str):
        self.source_dir = source_dir

    # ---- Cleaning ----

    def clean_latex_content(self, text: str) -> str:
        """Strip layout-only LaTeX commands, keeping math and semantic content."""
        text = re.sub(r'(?<!\\)%.*', '', text)
        text = re.sub(r'\\includegraphics\[.*?\]\{.*?\}', '[FIGURE]', text)

        commands_to_remove = [
            r'\\centering', r'\\medskip', r'\\bigskip', r'\\small',
            r'\\footnotesize', r'\\tiny', r'\\vfill', r'\\newpage',
            r'\\tableofcontents', r'\\pause', r'\\listoffigures',
            r'\\listoftables', r'\\frontmatter', r'\\mainmatter',
            r'\\backmatter', r'\\maketitle',
        ]
        for cmd in commands_to_remove:
            text = re.sub(cmd, '', text)

        text = text.replace(r'\item', '\n* ')
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        return text

    # ---- File-level entry point ----

    def parse_file(self, file_path: str, level: str, course: str) -> List[Document]:
        """Parse a .tex file, resolving includes, then splitting into Documents.

        Args:
            file_path: Absolute path to the .tex file.
            level: Academic level (M1, M2, Commun).
            course: Course folder name.

        Returns:
            List of Documents with enriched metadata.
        """
        filename = os.path.basename(file_path)
        print(f"   📖 [{level}/{course}] {filename}")

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            raw_content = f.read()

        # Resolve \include{} and \input{} before parsing
        base_dir = os.path.dirname(os.path.abspath(file_path))
        content = resolve_includes(
            raw_content,
            base_dir,
            visited={os.path.abspath(file_path)},
        )

        file_docs: List[Document] = []

        # Try chapter-based splitting first (\chapter{Title})
        chapter_pattern = re.compile(
            r'\\chapter(?:\*)?(?:\[.*?\])?\{(.*?)\}', re.DOTALL,
        )
        chapter_parts = chapter_pattern.split(content)

        if len(chapter_parts) > 1:
            # Document uses \chapter
            self._parse_chapters(
                chapter_parts, filename, level, course, file_docs,
            )
        else:
            # No chapters — section-based splitting
            self._parse_sections(content, filename, level, course, file_docs)

        return file_docs

    # ---- Chapter-level splitting ----

    def _parse_chapters(
        self,
        parts: list,
        filename: str,
        level: str,
        course: str,
        docs_list: List[Document],
    ):
        """Split content that uses \\chapter{} into documents.

        Args:
            parts: Result of regex split on \\chapter — alternating
                   [preamble, title1, body1, title2, body2, ...].
            filename: Source .tex filename.
            level: Academic level.
            course: Course folder name.
            docs_list: Accumulator for Documents.
        """
        # Preamble (before first chapter)
        preamble = parts[0]
        if preamble.strip():
            self._parse_sections(
                preamble, filename, level, course, docs_list,
                parent_chapter="Préambule",
            )

        # Alternate: Chapter Title → Chapter Body
        for i in range(1, len(parts), 2):
            chapter_title = parts[i].strip()
            chapter_body = parts[i + 1] if i + 1 < len(parts) else ""
            self._parse_sections(
                chapter_body, filename, level, course, docs_list,
                parent_chapter=chapter_title,
            )

    # ---- Section-level splitting ----

    def _parse_sections(
        self,
        content: str,
        filename: str,
        level: str,
        course: str,
        docs_list: List[Document],
        parent_chapter: str = "",
    ):
        """Split content by \\section{} then extract frames or text blocks.

        Args:
            content: LaTeX content (possibly a single chapter body).
            filename: Source .tex filename.
            level: Academic level.
            course: Course folder name.
            docs_list: Accumulator for Documents.
            parent_chapter: Parent chapter title (empty if no chapters).
        """
        section_pattern = re.compile(
            r'\\section(?:\*)?(?:\[.*?\])?\{(.*?)\}', re.DOTALL,
        )
        parts = section_pattern.split(content)

        default_section = "Introduction / Général"
        if parent_chapter:
            default_section = f"{parent_chapter} — Introduction"

        if parts[0].strip():
            self._extract_frames(
                parts[0], filename, default_section,
                level, course, docs_list, parent_chapter,
            )

        for i in range(1, len(parts), 2):
            section_title = parts[i].strip()
            section_content = parts[i + 1] if i + 1 < len(parts) else ""
            self._extract_frames(
                section_content, filename, section_title,
                level, course, docs_list, parent_chapter,
            )

    # ---- Frame/block extraction ----

    def _extract_frames(
        self,
        content: str,
        filename: str,
        section: str,
        level: str,
        course: str,
        docs_list: List[Document],
        chapter: str = "",
    ):
        """Extract individual frame environments or text blocks.

        Args:
            content: Raw LaTeX content of the section.
            filename: Source .tex filename.
            section: Section title.
            level: Academic level.
            course: Course name.
            docs_list: Accumulator for Documents.
            chapter: Parent chapter title (empty if not applicable).
        """
        frame_pattern = re.compile(
            r'\\begin\{frame\}(?:\[.*?\])?(?:\{(.*?)\})?'
            r'(.*?)'
            r'\\end\{frame\}',
            re.DOTALL,
        )
        matches = list(frame_pattern.finditer(content))

        base_metadata = {
            "source": filename,
            "level": level,
            "course": course,
            "section": section,
        }
        if chapter:
            base_metadata["chapter"] = chapter

        # Build context header for page_content
        header_lines = [
            f"SOURCE: {filename}",
            f"NIVEAU: {level}",
            f"COURS: {course}",
        ]
        if chapter:
            header_lines.append(f"CHAPITRE: {chapter}")
        header_lines.append(f"SECTION: {section}")

        if not matches and content.strip():
            clean_body = self.clean_latex_content(content)
            if len(clean_body) > 10:
                doc = Document(
                    page_content=(
                        "\n".join(header_lines) + f"\nCONTENU:\n{clean_body}"
                    ),
                    metadata={**base_metadata, "type": "text_block"},
                )
                docs_list.append(doc)
            return

        for match in matches:
            frame_title = match.group(1) if match.group(1) else "Sans titre"
            frame_body = match.group(2)
            clean_body = self.clean_latex_content(frame_body)

            doc = Document(
                page_content=(
                    "\n".join(header_lines)
                    + f"\nTITRE SLIDE: {frame_title}"
                    + f"\nCONTENU:\n{clean_body}"
                ),
                metadata={
                    **base_metadata,
                    "slide_title": frame_title,
                    "type": "slide",
                },
            )
            docs_list.append(doc)

    # ---- Top-level scanner ----

    def load_documents(self) -> Tuple[List[Document], Dict[str, int]]:
        """Recursively scan latex/{level}/{course}/ and parse all .tex files.

        Master files using \\include{chapter} are fully resolved: place
        the master .tex alongside its chapter files in the same course
        folder. Chapter files pulled in by a master are automatically
        skipped to avoid double-ingestion.

        Returns:
            Tuple of (all_documents, stats_dict) where stats_dict maps
            'level/course' to document count.
        """
        all_docs: List[Document] = []
        stats: Counter = Counter()

        if not os.path.isdir(self.source_dir):
            print(f"❌ Dossier source introuvable : {self.source_dir}")
            return all_docs, dict(stats)

        print(f"📂 Scan récursif de {self.source_dir}/ ...")

        # Pre-scan: identify master files and their includes
        master_files = self._find_master_files()

        for level_name in sorted(os.listdir(self.source_dir)):
            level_path = os.path.join(self.source_dir, level_name)

            if not os.path.isdir(level_path):
                if level_name.endswith(".tex"):
                    print(f"   ⚠️  Fichier racine (legacy) : {level_name}")
                    docs = self.parse_file(
                        os.path.join(self.source_dir, level_name),
                        level="legacy", course="legacy",
                    )
                    all_docs.extend(docs)
                    stats["legacy/legacy"] += len(docs)
                continue

            if level_name not in VALID_LEVELS:
                print(
                    f"   ⚠️  Dossier ignoré (niveau inconnu) : {level_name}/ "
                    f"— attendu : {', '.join(sorted(VALID_LEVELS))}"
                )
                continue

            for course_name in sorted(os.listdir(level_path)):
                course_path = os.path.join(level_path, course_name)

                if not os.path.isdir(course_path):
                    if course_name.endswith(".tex"):
                        docs = self.parse_file(
                            os.path.join(level_path, course_name),
                            level=level_name, course="Général",
                        )
                        all_docs.extend(docs)
                        stats[f"{level_name}/Général"] += len(docs)
                    continue

                tex_files = sorted(
                    f for f in os.listdir(course_path) if f.endswith(".tex")
                )
                if not tex_files:
                    print(f"   ⚠️  Aucun .tex dans {level_name}/{course_name}/")
                    continue

                # Determine which files are chapter includes (to skip)
                abs_course = os.path.abspath(course_path)
                local_masters = master_files.get(abs_course, set())
                chapter_basenames = set()
                for master_path in local_masters:
                    chapter_basenames.update(
                        self._get_included_basenames(master_path)
                    )

                for tex_name in tex_files:
                    tex_path = os.path.join(course_path, tex_name)
                    abs_path = os.path.abspath(tex_path)

                    # Skip chapter files that a master will pull in
                    bare_name = os.path.splitext(tex_name)[0]
                    if bare_name in chapter_basenames and abs_path not in local_masters:
                        print(f"      ⏭️  Skip (inclus par master) : {tex_name}")
                        continue

                    docs = self.parse_file(tex_path, level_name, course_name)
                    all_docs.extend(docs)
                    stats[f"{level_name}/{course_name}"] += len(docs)

        return all_docs, dict(stats)

    # ---- Helper: find master files ----

    def _find_master_files(self) -> Dict[str, Set[str]]:
        """Identify .tex files that contain \\include or \\input directives.

        Returns:
            Dict mapping directory abspath to set of master file abspaths.
        """
        masters: Dict[str, Set[str]] = {}

        for root, _dirs, files in os.walk(self.source_dir):
            for fname in files:
                if not fname.endswith(".tex"):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                        text = f.read()
                    if _INCLUDE_PATTERN.search(text):
                        abs_dir = os.path.abspath(root)
                        abs_file = os.path.abspath(fpath)
                        masters.setdefault(abs_dir, set()).add(abs_file)
                except Exception:
                    pass

        return masters

    def _get_included_basenames(self, master_path: str) -> Set[str]:
        """Extract basenames referenced by \\include/\\input in a master file.

        Args:
            master_path: Absolute path to the master .tex file.

        Returns:
            Set of basenames (without .tex extension).
        """
        basenames = set()
        try:
            with open(master_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            for match in _INCLUDE_PATTERN.finditer(text):
                ref = match.group(1).strip()
                if ref.endswith(".tex"):
                    ref = ref[:-4]
                basenames.add(os.path.basename(ref))
        except Exception:
            pass
        return basenames


# --- REPORTING ---

def print_ingestion_report(stats: Dict[str, int], total_chunks: int):
    """Print a summary table of ingested documents by level/course.

    Args:
        stats: Dict mapping 'level/course' to raw document count.
        total_chunks: Number of chunks after text splitting.
    """
    total_docs = sum(stats.values())

    print("\n" + "=" * 55)
    print("  📊 RAPPORT D'INGESTION")
    print("=" * 55)
    print(f"  {'Niveau/Cours':<35} {'Segments':>8}")
    print("-" * 55)

    for key in sorted(stats.keys()):
        print(f"  {key:<35} {stats[key]:>8}")

    print("-" * 55)
    print(f"  {'TOTAL segments bruts':<35} {total_docs:>8}")
    print(f"  {'TOTAL chunks (après split)':<35} {total_chunks:>8}")
    print("=" * 55)

    if "legacy/legacy" in stats:
        print(
            "\n  ⚠️  Des fichiers .tex à la racine de latex/ ont été "
            "ingérés avec level='legacy'. Déplacez-les dans la structure "
            "latex/{M1|M2|Commun}/{Cours}/ pour un cloisonnement correct."
        )


# --- MAIN ---

def main():
    """Run the full ingestion pipeline: parse → split → embed → store."""
    parser = argparse.ArgumentParser(description="ELA AI — Ingestion multi-niveaux")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse et affiche le rapport sans générer les embeddings.",
    )
    parser.add_argument(
        "--source-dir", default=SOURCE_DIR,
        help=f"Dossier source LaTeX (défaut : {SOURCE_DIR}).",
    )
    parser.add_argument(
        "--db-path", default=DB_PATH,
        help=f"Dossier cible ChromaDB (défaut : {DB_PATH}).",
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    db_path = args.db_path

    # 1. Validate source
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        print(f"⚠️ Dossier '{source_dir}' créé. Organisez vos .tex et relancez !")
        print(f"   Structure attendue : {source_dir}/M1/NomCours/fichier.tex")
        return

    # 2. Parse (with include resolution)
    ingestor = LatexIngestor(source_dir)
    raw_docs, stats = ingestor.load_documents()

    if not raw_docs:
        print("❌ Aucun document trouvé ou extrait. Vérifiez vos fichiers .tex.")
        return

    print(f"\n✅ {len(raw_docs)} segments extraits.")

    # 3. Split (safety net for oversized slides/blocks)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200,
    )
    splits = text_splitter.split_documents(raw_docs)

    # 4. Report
    print_ingestion_report(stats, total_chunks=len(splits))

    if args.dry_run:
        print("\n🏁 Dry-run terminé. Aucun embedding généré.")
        return

    # 5. Clean old DB
    if os.path.exists(db_path):
        print(f"\n🗑️  Suppression de l'ancienne base : {db_path}")
        shutil.rmtree(db_path)

    # 6. Embed & store
    print("🧠 Calcul des vecteurs (Embeddings)... Patientez.")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=db_path,
    )

    print(f"\n🎉 Ingestion terminée ! Base vectorielle créée dans '{db_path}'")


if __name__ == "__main__":
    main()