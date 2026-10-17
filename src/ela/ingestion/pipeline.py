"""Ingestion pipeline: parse LaTeX → chunk → embed → store (``ela-ingest``).

Usage::

    ela-ingest                      # incremental: only new/changed chunks are embedded
    ela-ingest --full               # clear the store and rebuild everything
    ela-ingest --dry-run            # parse and report only
    ela-ingest --dump chunks.jsonl  # also write every chunk (text + metadata) for review

The store is selected by ``RETRIEVAL_BACKEND`` (``pgvector`` in ``DATABASE_URL``,
or ``chroma`` in ``CHROMA_DB_PATH``). Incremental mode compares the content
hash of every chunk with what the store holds: unchanged chunks are kept,
new ones are embedded and inserted, chunks that disappeared are deleted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ela.config import Settings, get_settings
from ela.ingestion.latex import Chunk, LatexIngestor, ParseReport

Logger = Callable[[str], None]


@dataclass
class IngestResult:
    """Outcome of one ingestion run."""

    parsed: int = 0
    added: int = 0
    unchanged: int = 0
    removed: int = 0
    total: int = 0
    seconds: float = 0.0
    dry_run: bool = False
    backend: str = ""
    stats: dict[str, int] = field(default_factory=dict)
    by_type: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# --- REPORTING ---


def print_parse_report(report: ParseReport, log: Logger = print) -> None:
    """Print the per-course / per-type summary of a parse."""
    log("\n" + "=" * 60)
    log("  📊 RAPPORT D'ANALYSE DES COURS")
    log("=" * 60)
    log(f"  {'Niveau/Cours':<40} {'Chunks':>8}")
    log("-" * 60)
    for key in sorted(report.stats):
        log(f"  {key:<40} {report.stats[key]:>8}")
    log("-" * 60)
    log(f"  {'TOTAL':<40} {len(report.chunks):>8}")
    by_type = report.by_type()
    log("  Par type : " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items(), key=lambda x: -x[1])))
    kinds = Counter(c.doc_kind for c in report.chunks)
    log("  Par document : " + ", ".join(f"{k}={v}" for k, v in kinds.items()))
    log("=" * 60)
    for w in report.warnings:
        log(f"  ⚠️  {w}")
    if "legacy/legacy" in report.stats:
        log(
            "\n  ⚠️  Des fichiers .tex à la racine de latex/ ont été ingérés avec level='legacy'. "
            "Déplacez-les dans latex/{M1|M2|Commun}/{Cours}/ pour un cloisonnement correct."
        )


def dump_chunks(chunks: list[Chunk], path: str) -> None:
    """Write chunks as JSON lines (one object per chunk) for review."""
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"text": c.page_content(), **c.metadata()}, ensure_ascii=False) + "\n")


# --- PIPELINE ---


def _embed(embeddings, chunks: list[Chunk], batch_size: int, log: Logger) -> list[list[float]]:
    vectors: list[list[float]] = []
    total = len(chunks)
    start = time.time()
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        vectors.extend(embeddings.embed_documents([c.page_content() for c in batch]))
        done = min(i + batch_size, total)
        if done == total or (i // batch_size) % 10 == 0:
            rate = done / max(time.time() - start, 1e-6)
            log(f"   🧠 {done}/{total} chunks vectorisés ({rate:.1f}/s)")
    return vectors


def run(
    *,
    source_dir: str | None = None,
    dry_run: bool = False,
    full: bool = False,
    dump: str | None = None,
    settings: Settings | None = None,
    store=None,
    embeddings=None,
    log: Logger = print,
) -> IngestResult:
    """Run the ingestion pipeline.

    Args:
        source_dir: Root of the ``latex/{level}/{course}/`` tree (default: settings).
        dry_run: Parse and report without touching the store.
        full: Clear the store before inserting (otherwise incremental).
        dump: Optional path of a JSON-lines dump of every chunk.
        settings: Settings override.
        store: Chunk store override (tests); default from ``RETRIEVAL_BACKEND``.
        embeddings: Embedding model override (tests).
        log: Progress logger.

    Returns:
        An :class:`IngestResult`.
    """
    s = settings or get_settings()
    source = source_dir or s.latex_source_dir
    t0 = time.time()
    result = IngestResult(dry_run=dry_run, backend=s.retrieval_backend)

    src = Path(source)
    if not src.is_dir():
        src.mkdir(parents=True, exist_ok=True)
        log(f"⚠️ Dossier '{source}' créé. Organisez vos .tex ({source}/M1/NomCours/fichier.tex) et relancez.")
        return result

    # 1. Parse
    ingestor = LatexIngestor(
        source,
        target_chars=s.chunk_target_chars,
        max_chars=s.chunk_max_chars,
        include_notes=s.ingest_include_notes,
        log=log,
    )
    report = ingestor.scan()
    print_parse_report(report, log)
    result.parsed = len(report.chunks)
    result.stats = report.stats
    result.by_type = report.by_type()
    result.warnings = report.warnings
    if dump:
        dump_chunks(report.chunks, dump)
        log(f"\n📝 {len(report.chunks)} chunks écrits dans {dump}")
    if not report.chunks:
        log("❌ Aucun contenu extrait. Vérifiez vos fichiers .tex.")
        return result
    if dry_run:
        log("\n🏁 Dry-run terminé. Aucun embedding généré.")
        result.seconds = time.time() - t0
        return result

    # 2. Store and diff
    if store is None:
        from ela.retrieval.store import build_store

        store = build_store(s)
    result.backend = store.backend
    if hasattr(store, "ensure_schema"):
        store.ensure_schema()
    if full:
        log(f"\n🗑️  Réinitialisation du store ({store.backend})")
        store.clear()
    existing = store.existing()
    parsed_ids = {c.chunk_id: c for c in report.chunks}
    to_add = [c for cid, c in parsed_ids.items() if existing.get(cid) != c.content_hash]
    to_remove = [cid for cid in existing if cid not in parsed_ids]
    result.unchanged = len(parsed_ids) - len(to_add)
    log(f"\n🔄 {len(to_add)} chunks à vectoriser, {result.unchanged} inchangés, {len(to_remove)} à supprimer")

    # 3. Embed & upsert
    if to_add:
        if embeddings is None:
            from ela.retrieval.embeddings import build_embeddings

            embeddings = build_embeddings(s)
        vectors = _embed(embeddings, to_add, s.embedding_batch_size, log)
        if vectors and len(vectors[0]) != s.embedding_dim:
            raise RuntimeError(
                f"EMBEDDING_DIM={s.embedding_dim} mais le modèle renvoie des vecteurs de taille {len(vectors[0])}."
            )
        store.upsert(to_add, vectors)
        result.added = len(to_add)
    if to_remove:
        store.delete(to_remove)
        result.removed = len(to_remove)
    result.total = store.count()
    result.seconds = time.time() - t0
    log(f"\n🎉 Ingestion terminée ({result.seconds:.0f}s) : {result.total} chunks dans le store {store.backend}.")
    return result


# --- CLI ---


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="ELA AI — Ingestion de la base de connaissances")
    parser.add_argument("--dry-run", action="store_true", help="Analyse et rapport sans générer les embeddings.")
    parser.add_argument("--full", action="store_true", help="Vide le store puis réingère tout.")
    parser.add_argument("--dump", metavar="FICHIER", help="Écrit chaque chunk (texte + métadonnées) en JSON lines.")
    parser.add_argument(
        "--source-dir",
        default=settings.latex_source_dir,
        help=f"Dossier source LaTeX (défaut : {settings.latex_source_dir}).",
    )
    return parser


def main() -> None:
    """Console entry point (``ela-ingest``)."""
    for stream in (sys.stdout, sys.stderr):  # emojis on a cp1252 console (Windows)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_arg_parser().parse_args()
    try:
        run(source_dir=args.source_dir, dry_run=args.dry_run, full=args.full, dump=args.dump)
    except RuntimeError as exc:
        print(f"\n❌ {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
