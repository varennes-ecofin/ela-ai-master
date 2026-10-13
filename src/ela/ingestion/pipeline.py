"""Ingestion pipeline: parse LaTeX → split → embed → store (``ela-ingest``).

Usage::

    ela-ingest                      # full rebuild of the vector store
    ela-ingest --dry-run            # parse and report only
    ela-ingest --source-dir ... --db-path ...

Paths default to ``LATEX_SOURCE_DIR`` and ``CHROMA_DB_PATH`` (settings).
Phase 2 makes this incremental and job-based.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ela.config import get_settings
from ela.ingestion.latex import LatexIngestor

# --- REPORTING ---


def print_ingestion_report(stats: dict[str, int], total_chunks: int):
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="ELA AI — Ingestion multi-niveaux")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse et affiche le rapport sans générer les embeddings.",
    )
    parser.add_argument(
        "--source-dir",
        default=settings.latex_source_dir,
        help=f"Dossier source LaTeX (défaut : {settings.latex_source_dir}).",
    )
    parser.add_argument(
        "--db-path",
        default=settings.chroma_db_path,
        help=f"Dossier cible ChromaDB (défaut : {settings.chroma_db_path}).",
    )
    return parser


def run(source_dir: str, db_path: str, dry_run: bool = False) -> int:
    """Run the full ingestion pipeline: parse → split → embed → store.

    Args:
        source_dir: Root of the ``latex/{level}/{course}/`` tree.
        db_path: Target Chroma directory (replaced entirely).
        dry_run: Parse and report without computing embeddings.

    Returns:
        Number of chunks written (0 on dry run or when nothing was found).
    """
    settings = get_settings()

    # 1. Validate source
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        print(f"⚠️ Dossier '{source_dir}' créé. Organisez vos .tex et relancez !")
        print(f"   Structure attendue : {source_dir}/M1/NomCours/fichier.tex")
        return 0

    # 2. Parse (with include resolution)
    ingestor = LatexIngestor(source_dir)
    raw_docs, stats = ingestor.load_documents()
    if not raw_docs:
        print("❌ Aucun document trouvé ou extrait. Vérifiez vos fichiers .tex.")
        return 0
    print(f"\n✅ {len(raw_docs)} segments extraits.")

    # 3. Split (safety net for oversized slides/blocks)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    splits = splitter.split_documents(raw_docs)

    # 4. Report
    print_ingestion_report(stats, total_chunks=len(splits))
    if dry_run:
        print("\n🏁 Dry-run terminé. Aucun embedding généré.")
        return 0

    # 5. Clean old DB
    if os.path.exists(db_path):
        print(f"\n🗑️  Suppression de l'ancienne base : {db_path}")
        shutil.rmtree(db_path)

    # 6. Embed & store (imported here: heavy dependencies)
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    print("🧠 Calcul des vecteurs (Embeddings)... Patientez.")
    embedding_model = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)
    Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory=db_path)
    print(f"\n🎉 Ingestion terminée ! Base vectorielle créée dans '{db_path}'")
    return len(splits)


def main() -> None:
    """Console entry point (``ela-ingest``)."""
    # The report uses emojis: never crash on a cp1252 console/pipe (Windows).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_arg_parser().parse_args()
    run(args.source_dir, args.db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
