"""Courses: the file tree, uploads, moves, deletions, the knowledge-base state and ingestion jobs (lot 5.3)."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response

from backoffice.deps import client_ip, csrf_protect, current_user, require_admin
from backoffice.rendering import render
from backoffice.security import Session
from ela.courses import catalog, state
from ela.db import audit
from ela.db import jobs as jobs_db
from ela.jobs.runner import MODE_LABELS, MODES, JobBusyError

logger = logging.getLogger("ela.backoffice.courses")
router = APIRouter()

ROOT_LABELS = {"latex": "Cours", "annales": "Annales (examens passés)"}
FLASH = {
    "created": "Cours créé.",
    "uploaded": "Fichiers déposés.",
    "deleted_file": "Fichier supprimé.",
    "deleted": "Cours supprimé.",
    "moved": "Cours déplacé.",
}


def _roots(request: Request) -> dict[str, str]:
    s = request.app.state.settings
    return {"latex": s.latex_source_dir, "annales": s.annales_source_dir}


def _root(request: Request, key: str) -> str:
    roots = _roots(request)
    if key not in roots:
        raise catalog.CatalogError(f"espace inconnu « {key} »")
    return roots[key]


def _error_page(request: Request, message: str, status: int = 400):
    return render(request, "error.html", {"title": "Opération refusée", "message": message}, status)


async def _job_context(request: Request) -> dict:
    """Active job + last finished job, for the ingestion panel."""
    active = await jobs_db.active_job()
    runner = request.app.state.jobs
    return {
        "active_job": active,
        "last_job": await jobs_db.last_finished(),
        "busy": active is not None or runner.busy,
        "modes": [(m, MODE_LABELS[m]) for m in MODES],
        "labels": MODE_LABELS,
    }


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


@router.get("/courses", name="courses")
async def courses_index(request: Request, session: Session = Depends(current_user)):
    """Tree of both roots, knowledge-base state, diagnostic and ingestion panel."""
    roots = _roots(request)
    tree = catalog.scan(roots["latex"])
    annales = catalog.scan(roots["annales"])
    kb = await asyncio.to_thread(state.knowledge_base, None, request.app.state.settings)
    diagnostics = state.diagnose(tree, kb)
    ctx = {
        "tree": tree,
        "annales": annales,
        "kb": kb,
        "kb_courses": {c.key: c for c in kb.courses},
        "diagnostics": diagnostics,
        "problems": [d for d in diagnostics if d.status != state.OK],
        "levels": catalog.LEVELS,
        "flash": FLASH.get(request.query_params.get("ok", "")),
        "human_size": catalog.human_size,
        **(await _job_context(request)),
    }
    return render(request, "courses/index.html", ctx)


@router.get("/courses/partials/job", name="job_panel")
async def job_panel(request: Request, session: Session = Depends(current_user)):
    """HTMX fragment: the ingestion panel (polled while a job runs)."""
    ctx = await _job_context(request)
    ctx["levels"] = catalog.LEVELS
    return render(request, "courses/_job_panel.html", ctx)


@router.post("/courses/new", name="course_create", dependencies=[Depends(csrf_protect)])
async def course_create(
    request: Request,
    root: str = Form(default="latex"),
    level: str = Form(default=""),
    name: str = Form(default=""),
    session: Session = Depends(require_admin),
):
    """Create an empty course folder."""
    try:
        catalog.create_course(_root(request, root), level, name)
    except catalog.CatalogError as exc:
        return _error_page(request, str(exc).capitalize() + ".")
    await audit.record(audit.COURSE_CREATE, session.identifier, target=f"{root}:{level}/{name}", ip=client_ip(request))
    return RedirectResponse(f"{request.app.state.prefix}/courses/{root}/{level}/{name}?ok=created", status_code=303)


# ---------------------------------------------------------------------------
# One course
# ---------------------------------------------------------------------------


@router.get("/courses/{root}/{level}/{name}", name="course_detail")
async def course_detail(request: Request, root: str, level: str, name: str, session: Session = Depends(current_user)):
    """Files of a course, upload / move / delete forms, store statistics."""
    try:
        base = _root(request, root)
        catalog.course_dir(base, level, name)
    except catalog.CatalogError as exc:
        return _error_page(request, str(exc).capitalize() + ".", 404)
    tree = catalog.scan(base)
    course = tree.course(level, name)
    stats = None
    if root == "latex":
        kb = await asyncio.to_thread(state.knowledge_base, None, request.app.state.settings)
        stats = kb.course(level, name)
    ctx = {
        "root": root,
        "root_label": ROOT_LABELS[root],
        "course": course,
        "stats": stats,
        "levels": catalog.LEVELS,
        "other_levels": [lvl for lvl in catalog.LEVELS if lvl != level],
        "allowed": ", ".join(catalog.ALLOWED_EXTENSIONS),
        "flash": FLASH.get(request.query_params.get("ok", "")),
        "report": request.app.state.oneshot.pop(request.query_params.get("report", ""), owner=session.identifier)
        if request.query_params.get("report")
        else None,
        "human_size": catalog.human_size,
        **(await _job_context(request)),
    }
    return render(request, "courses/detail.html", ctx)


@router.get("/courses/{root}/{level}/{name}/files/{path:path}", name="course_file")
async def course_file(
    request: Request, root: str, level: str, name: str, path: str, session: Session = Depends(current_user)
):
    """Download one source file."""
    try:
        target = catalog.file_path(_root(request, root), level, name, path)
    except catalog.CatalogError as exc:
        return _error_page(request, str(exc).capitalize() + ".", 404)
    return FileResponse(target, media_type="text/plain; charset=utf-8", filename=target.name)


@router.post("/courses/{root}/{level}/{name}/upload", name="course_upload", dependencies=[Depends(csrf_protect)])
async def course_upload(
    request: Request,
    root: str,
    level: str,
    name: str,
    files: list[UploadFile] = File(default=[]),
    session: Session = Depends(require_admin),
):
    """Store uploaded source files (``.zip`` archives are unpacked)."""
    try:
        base = _root(request, root)
        catalog.course_dir(base, level, name)
        written: list[str] = []
        replaced: list[str] = []
        skipped: list[str] = []
        for upload in files:
            if not upload.filename:
                continue
            data = await upload.read(catalog.MAX_UPLOAD_BYTES + 1)
            if upload.filename.lower().endswith(".zip"):
                report = catalog.extract_zip(base, level, name, data)
                written += report.written
                replaced += report.replaced
                skipped += report.skipped
            else:
                rel, was_replaced = catalog.save_file(base, level, name, upload.filename, data)
                (replaced if was_replaced else written).append(rel)
    except catalog.CatalogError as exc:
        return _error_page(request, str(exc).capitalize() + ".")
    if not written and not replaced:
        return _error_page(
            request, "Aucun fichier exploitable dans ce dépôt (extensions acceptées : .tex, .sty, .cls, .bib, .zip)."
        )
    await audit.record(
        audit.COURSE_UPLOAD,
        session.identifier,
        target=f"{root}:{level}/{name}",
        ip=client_ip(request),
        details={"written": len(written), "replaced": len(replaced), "skipped": len(skipped)},
    )
    token = request.app.state.oneshot.put(
        {"written": written, "replaced": replaced, "skipped": skipped}, owner=session.identifier
    )
    return RedirectResponse(
        f"{request.app.state.prefix}/courses/{root}/{level}/{name}?ok=uploaded&report={token}", status_code=303
    )


@router.post(
    "/courses/{root}/{level}/{name}/delete-file", name="course_delete_file", dependencies=[Depends(csrf_protect)]
)
async def course_delete_file(
    request: Request,
    root: str,
    level: str,
    name: str,
    path: str = Form(default=""),
    session: Session = Depends(require_admin),
):
    """Delete one file."""
    try:
        catalog.delete_file(_root(request, root), level, name, path)
    except catalog.CatalogError as exc:
        return _error_page(request, str(exc).capitalize() + ".")
    await audit.record(
        audit.COURSE_DELETE_FILE, session.identifier, target=f"{root}:{level}/{name}/{path}", ip=client_ip(request)
    )
    return RedirectResponse(
        f"{request.app.state.prefix}/courses/{root}/{level}/{name}?ok=deleted_file", status_code=303
    )


@router.post("/courses/{root}/{level}/{name}/move", name="course_move", dependencies=[Depends(csrf_protect)])
async def course_move(
    request: Request,
    root: str,
    level: str,
    name: str,
    to_level: str = Form(default=""),
    session: Session = Depends(require_admin),
):
    """Move the course to another level; for courses, start the incremental ingestion that re-indexes it."""
    runner = request.app.state.jobs
    if root == "latex" and (runner.busy or await jobs_db.active_job()):
        return _error_page(
            request, "Un job est en cours (ingestion ou évaluation) : attendez sa fin avant de déplacer un cours.", 409
        )
    try:
        catalog.move_course(_root(request, root), level, name, to_level)
    except catalog.CatalogError as exc:
        return _error_page(request, str(exc).capitalize() + ".")
    await audit.record(
        audit.COURSE_MOVE,
        session.identifier,
        target=f"{root}:{level}/{name}",
        ip=client_ip(request),
        details={"to": to_level},
    )
    prefix = request.app.state.prefix
    if root != "latex":
        return RedirectResponse(f"{prefix}/courses/{root}/{to_level}/{name}?ok=moved", status_code=303)
    try:
        job_id = runner.start_ingest(
            "incremental", started_by=session.identifier, target=f"déplacement {level}/{name} → {to_level}"
        )
    except JobBusyError:
        return RedirectResponse(f"{prefix}/courses/{root}/{to_level}/{name}?ok=moved", status_code=303)
    await audit.record(
        audit.INGEST_START,
        session.identifier,
        target=str(job_id),
        ip=client_ip(request),
        details={"mode": "incremental", "reason": "move"},
    )
    return RedirectResponse(f"{prefix}/jobs/{job_id}", status_code=303)


@router.post("/courses/{root}/{level}/{name}/delete", name="course_delete", dependencies=[Depends(csrf_protect)])
async def course_delete(request: Request, root: str, level: str, name: str, session: Session = Depends(require_admin)):
    """Delete the whole course folder (its chunks go away at the next incremental update)."""
    try:
        count = catalog.delete_course(_root(request, root), level, name)
    except catalog.CatalogError as exc:
        return _error_page(request, str(exc).capitalize() + ".")
    await audit.record(
        audit.COURSE_DELETE,
        session.identifier,
        target=f"{root}:{level}/{name}",
        ip=client_ip(request),
        details={"files": count},
    )
    return RedirectResponse(f"{request.app.state.prefix}/courses?ok=deleted", status_code=303)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


@router.post("/ingest", name="ingest_start", dependencies=[Depends(csrf_protect)])
async def ingest_start(request: Request, mode: str = Form(default=""), session: Session = Depends(require_admin)):
    """Start an ingestion job (dry run, incremental, full)."""
    if mode not in MODES:
        return _error_page(request, "Mode d'ingestion inconnu.")
    if await jobs_db.active_job():
        return _error_page(request, "Un job est déjà en cours (ingestion ou évaluation).", 409)
    try:
        job_id = request.app.state.jobs.start_ingest(mode, started_by=session.identifier)
    except JobBusyError:
        return _error_page(request, "Un job est déjà en cours (ingestion ou évaluation).", 409)
    await audit.record(
        audit.INGEST_START, session.identifier, target=str(job_id), ip=client_ip(request), details={"mode": mode}
    )
    return RedirectResponse(f"{request.app.state.prefix}/jobs/{job_id}", status_code=303)


def _not_found(request: Request) -> Response:
    return render(
        request, "error.html", {"title": "Job introuvable", "message": "Aucun job ne porte cet identifiant."}, 404
    )


@router.get("/jobs", name="jobs")
async def jobs_list(request: Request, session: Session = Depends(current_user)):
    """History of jobs."""
    return render(request, "jobs/list.html", {"jobs": await jobs_db.list_jobs(limit=50), "labels": MODE_LABELS})


@router.get("/jobs/{job_id}", name="job_detail")
async def job_detail(request: Request, job_id: int, session: Session = Depends(current_user)):
    """One job with its live log."""
    job = await jobs_db.get_job(job_id)
    if job is None:
        return _not_found(request)
    return render(request, "jobs/detail.html", {"job": job, "labels": MODE_LABELS})


@router.get("/jobs/{job_id}/partial", name="job_partial")
async def job_partial(request: Request, job_id: int, session: Session = Depends(current_user)):
    """HTMX fragment of a job (status, progress, log), polled every 2 s while it runs."""
    job = await jobs_db.get_job(job_id)
    if job is None:
        return _not_found(request)
    return render(request, "jobs/_status.html", {"job": job, "labels": MODE_LABELS})
