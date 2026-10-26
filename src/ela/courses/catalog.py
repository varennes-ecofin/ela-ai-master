"""The course tree on disk: ``<root>/{M1|M2|Commun}/<Course>/*.tex``.

Two roots share this layout: the courses (``LATEX_SOURCE_DIR``, ingested) and
the past exams (``ANNALES_SOURCE_DIR``, stored for a later lot). Every
operation validates names and paths so that nothing can escape the root.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

LEVELS = ("M1", "M2", "Commun")
ALLOWED_EXTENSIONS = (".tex", ".sty", ".cls", ".bib")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_ZIP_MEMBER_BYTES = 20 * 1024 * 1024
MAX_ZIP_MEMBERS = 500
COURSE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,63}$")
_FILE_PART_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class CatalogError(Exception):
    """A refused operation (bad name, missing course, collision...)."""


@dataclass(frozen=True)
class FileInfo:
    """One file inside a course folder."""

    path: str  # relative to the course folder, POSIX separators
    size: int
    modified: datetime
    ingested: bool  # top-level .tex files are what the ingestion parses

    @property
    def name(self) -> str:
        """Base name of the file."""
        return self.path.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class CourseInfo:
    """A course folder."""

    level: str
    name: str
    files: list[FileInfo] = field(default_factory=list)

    @property
    def key(self) -> str:
        """``level/name`` (matches the ingestion statistics)."""
        return f"{self.level}/{self.name}"

    @property
    def tex_count(self) -> int:
        """Number of top-level ``.tex`` files."""
        return sum(1 for f in self.files if f.ingested)

    @property
    def size(self) -> int:
        """Total size in bytes."""
        return sum(f.size for f in self.files)


@dataclass(frozen=True)
class Tree:
    """The whole root: courses per level, plus stray files."""

    root: Path
    exists: bool
    courses: dict[str, list[CourseInfo]]  # level → courses
    stray: list[str] = field(default_factory=list)  # files or folders that the ingestion ignores

    def course(self, level: str, name: str) -> CourseInfo | None:
        """Find a course."""
        for c in self.courses.get(level, []):
            if c.name == name:
                return c
        return None

    @property
    def all_courses(self) -> list[CourseInfo]:
        """Every course, levels in order."""
        return [c for level in LEVELS for c in self.courses.get(level, [])]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def check_level(level: str) -> str:
    """Return ``level`` if it is one of :data:`LEVELS`."""
    if level not in LEVELS:
        raise CatalogError(f"niveau inconnu « {level} » (M1, M2 ou Commun)")
    return level


def check_course_name(name: str) -> str:
    """Return ``name`` if it is a valid course folder name (letters, digits, ``_``, ``-``)."""
    name = (name or "").strip()
    if not COURSE_NAME_RE.match(name):
        raise CatalogError(
            "nom de cours invalide : 2 à 64 caractères, lettres sans accent, chiffres, tiret ou souligné "
            "(par ex. Series_Temporelles)"
        )
    return name


def check_relative_path(path: str) -> str:
    """Validate a path relative to a course folder (no ``..``, no absolute path, safe characters)."""
    raw = (path or "").replace("\\", "/")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise CatalogError(f"chemin absolu refusé « {path} »")
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if not parts:
        raise CatalogError("chemin vide")
    for part in parts:
        if part == ".." or not _FILE_PART_RE.match(part):
            raise CatalogError(f"nom de fichier invalide « {part} »")
    return "/".join(parts)


def check_extension(path: str) -> None:
    """Refuse files that are not course sources."""
    if not path.lower().endswith(ALLOWED_EXTENSIONS):
        raise CatalogError(f"type de fichier refusé « {path} » (acceptés : {', '.join(ALLOWED_EXTENSIONS)})")


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def _course_info(root: Path, level: str, name: str) -> CourseInfo:
    folder = root / level / name
    files: list[FileInfo] = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        rel = path.relative_to(folder).as_posix()
        stat = path.stat()
        files.append(
            FileInfo(
                path=rel,
                size=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime),
                ingested=("/" not in rel and rel.lower().endswith(".tex")),
            )
        )
    return CourseInfo(level=level, name=name, files=files)


def scan(root: str | Path) -> Tree:
    """Describe the tree under ``root`` (missing root → empty tree, ``exists=False``)."""
    root = Path(root)
    if not root.is_dir():
        return Tree(root=root, exists=False, courses={lvl: [] for lvl in LEVELS})
    courses: dict[str, list[CourseInfo]] = {lvl: [] for lvl in LEVELS}
    stray: list[str] = []
    for entry in sorted(root.iterdir()):
        if entry.name in LEVELS and entry.is_dir():
            for sub in sorted(entry.iterdir()):
                if sub.is_dir():
                    courses[entry.name].append(_course_info(root, entry.name, sub.name))
                else:
                    stray.append(f"{entry.name}/{sub.name}")
        elif not entry.name.startswith("."):
            stray.append(entry.name + ("/" if entry.is_dir() else ""))
    return Tree(root=root, exists=True, courses=courses, stray=stray)


def course_dir(root: str | Path, level: str, name: str, *, must_exist: bool = True) -> Path:
    """Resolve and validate a course folder."""
    folder = Path(root) / check_level(level) / check_course_name(name)
    if must_exist and not folder.is_dir():
        raise CatalogError(f"cours introuvable : {level}/{name}")
    return folder


def file_path(root: str | Path, level: str, name: str, rel: str) -> Path:
    """Resolve a file inside a course folder (validated, must exist)."""
    folder = course_dir(root, level, name)
    target = folder / check_relative_path(rel)
    if not target.is_file():
        raise CatalogError(f"fichier introuvable : {rel}")
    return target


# ---------------------------------------------------------------------------
# Changes
# ---------------------------------------------------------------------------


def create_course(root: str | Path, level: str, name: str) -> Path:
    """Create an empty course folder.

    Raises:
        CatalogError: Invalid level/name or the course already exists.
    """
    folder = course_dir(root, level, name, must_exist=False)
    if folder.exists():
        raise CatalogError(f"le cours {level}/{name} existe déjà")
    folder.mkdir(parents=True)
    return folder


def save_file(root: str | Path, level: str, name: str, rel: str, data: bytes) -> tuple[str, bool]:
    """Write one source file into a course (sub-folders allowed).

    Returns:
        ``(relative path, replaced)`` — ``replaced`` is ``True`` when a file was overwritten.

    Raises:
        CatalogError: Invalid path or extension, file too large, unknown course.
    """
    folder = course_dir(root, level, name)
    rel = check_relative_path(rel)
    check_extension(rel)
    if len(data) > MAX_UPLOAD_BYTES:
        raise CatalogError(f"fichier trop volumineux « {rel} » ({len(data) // (1024 * 1024)} Mo, maximum 20 Mo)")
    target = folder / rel
    replaced = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return rel, replaced


@dataclass
class ZipReport:
    """What a zip upload did."""

    written: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # ignored members (other extensions, folders, junk)


def extract_zip(root: str | Path, level: str, name: str, data: bytes) -> ZipReport:
    """Unpack the source files of a zip archive into a course (folders kept, other files ignored).

    A single top-level folder wrapping everything (as produced by zipping a
    folder) is stripped. Members escaping the course folder are refused.

    Raises:
        CatalogError: Not a zip, too many members, a member too large, or an unsafe path.
    """
    course_dir(root, level, name)
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise CatalogError("archive zip illisible") from exc
    members = [m for m in archive.infolist() if not m.is_dir()]
    if len(members) > MAX_ZIP_MEMBERS:
        raise CatalogError(f"archive trop grande ({len(members)} fichiers, maximum {MAX_ZIP_MEMBERS})")
    names = [m.filename.replace("\\", "/") for m in members]

    def junk(n: str) -> bool:
        return "__MACOSX" in n or n.rsplit("/", 1)[-1].startswith(".")

    # strip a common wrapping folder (zip of a folder), ignoring junk members
    prefix = ""
    real = [n for n in names if not junk(n)]
    first_parts = {n.split("/", 1)[0] for n in real if "/" in n}
    if real and len(first_parts) == 1 and all("/" in n for n in real):
        prefix = first_parts.pop() + "/"
    report = ZipReport()
    for member, raw in zip(members, names, strict=True):
        if junk(raw):
            report.skipped.append(raw)
            continue
        rel = raw[len(prefix) :] if raw.startswith(prefix) else raw
        try:
            rel = check_relative_path(rel)
            check_extension(rel)
        except CatalogError:
            report.skipped.append(rel)
            continue
        if member.file_size > MAX_ZIP_MEMBER_BYTES:
            raise CatalogError(f"fichier trop volumineux dans l'archive « {rel} »")
        _, replaced = save_file(root, level, name, rel, archive.read(member))
        (report.replaced if replaced else report.written).append(rel)
    return report


def delete_file(root: str | Path, level: str, name: str, rel: str) -> None:
    """Delete one file of a course (empty sub-folders are removed too)."""
    folder = course_dir(root, level, name)
    target = file_path(root, level, name, rel)
    target.unlink()
    parent = target.parent
    while parent != folder and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent


def delete_course(root: str | Path, level: str, name: str) -> int:
    """Delete a whole course folder; returns the number of files removed."""
    folder = course_dir(root, level, name)
    count = sum(1 for p in folder.rglob("*") if p.is_file())
    shutil.rmtree(folder)
    return count


def move_course(root: str | Path, level: str, name: str, new_level: str) -> Path:
    """Move a course to another level (same name).

    Raises:
        CatalogError: Same level, unknown course, or a course of that name already exists there.
    """
    folder = course_dir(root, level, name)
    new_level = check_level(new_level)
    if new_level == level:
        raise CatalogError("le cours est déjà dans ce niveau")
    target = Path(root) / new_level / name
    if target.exists():
        raise CatalogError(f"un cours {new_level}/{name} existe déjà")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(folder, target)
    return target


def human_size(size: int) -> str:
    """``1234`` → ``1,2 Ko``."""
    for unit in ("o", "Ko", "Mo", "Go"):
        if size < 1024 or unit == "Go":
            return (f"{size:.1f}".replace(".", ",") if unit != "o" else str(size)) + f" {unit}"
        size /= 1024
    return f"{size} Go"
