"""Evaluation of the RAG from the backoffice (lot 5.5a).

Launch ``ela-eval retrieval`` / ``ela-eval answer`` as a background job
(``JobRunner``), browse the saved reports (``eval/reports/*.json``), open
one report (metrics, per-course table, per-question table, failures
first), download or delete it, and compare two reports of the same kind.
``admin`` launches and deletes; ``supervisor`` reads everything.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, get_args

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, RedirectResponse, Response

from backoffice.csvexport import csv_response
from backoffice.deps import client_ip, csrf_protect, current_user, require_admin
from backoffice.rendering import render
from backoffice.security import Session
from ela.config import Settings
from ela.db import audit
from ela.db import jobs as jobs_db
from ela.eval.compare import KIND_ANSWER, KIND_RETRIEVAL, METRIC_SPECS, ReportKindError, compare_reports, report_kind
from ela.eval.dataset import QuestionType, load_dataset
from ela.eval.reports import count_failures, valid_label
from ela.jobs.runner import EVAL_KINDS, MODE_LABELS, EvalOptions, JobBusyError

logger = logging.getLogger("ela.backoffice.evaluation")
router = APIRouter()

KIND_LABELS = {KIND_RETRIEVAL: "Retrieval", KIND_ANSWER: "Réponses", "code": "Atelier Code"}
QUESTION_TYPES: tuple[str, ...] = get_args(QuestionType)
PROVIDERS = ("cerebras", "ilaas", "groq")
PROVIDER_LABELS = {"cerebras": "Cerebras", "ilaas": "ILaaS", "groq": "Groq"}
#: Suggested model names (datalist); the field itself is free text.
MODEL_SUGGESTIONS = ("gpt-oss-120b", "qwen-3.8-27b", "qwen3.8-27b", "gemma-4-31b", "mistral-small-4-119b")
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,79}$")
DEFAULT_LIMIT = 20
MAX_K = 50
FLASH = {"deleted": "Rapport supprimé.", "started": "Évaluation lancée."}

METRIC_LABELS = {kind: {key: label for key, label, _ in specs} for kind, specs in METRIC_SPECS.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_page(request: Request, message: str, status: int = 400) -> Response:
    return render(request, "error.html", {"title": "Opération refusée", "message": message}, status)


def available_providers(settings: Settings) -> list[tuple[str, str]]:
    """Return the providers whose API key (and base URL) are configured on the server."""
    out = []
    for name in PROVIDERS:
        try:
            _, key = settings.provider_credentials(name)
        except ValueError:
            continue
        if key:
            out.append((name, PROVIDER_LABELS[name]))
    return out


def dataset_facts(settings: Settings) -> dict[str, Any]:
    """Courses, types and question counts of the reference set (for the form and the cost estimate)."""
    try:
        ds = load_dataset(Path(settings.eval_dataset))
    except Exception as exc:  # missing or invalid file: the form still renders, with a notice
        logger.warning("Evaluation dataset unavailable: %s", exc)
        return {"error": str(exc), "courses": [], "types": [], "counts": {}, "retrieval_counts": {}, "total": 0}
    counts: Counter[tuple[str, str]] = Counter()
    retrieval: Counter[tuple[str, str]] = Counter()
    for q in ds.questions:
        counts[(q.course, q.type)] += 1
        if q.expects_retrieval:
            retrieval[(q.course, q.type)] += 1
    return {
        "error": None,
        "version": ds.version,
        "courses": sorted({q.course for q in ds.questions}),
        "types": [t for t in QUESTION_TYPES if any(q.type == t for q in ds.questions)],
        "counts": {f"{c}|{t}": n for (c, t), n in counts.items()},
        "retrieval_counts": {f"{c}|{t}": n for (c, t), n in retrieval.items()},
        "total": len(ds.questions),
    }


def judge_defaults(settings: Settings) -> tuple[str, str]:
    """Judge model and provider as the server resolves them."""
    return settings.eval_judge_model or settings.llm_model_chat, settings.eval_judge_provider or settings.llm_provider


def parse_options(form: dict[str, str], settings: Settings, facts: dict[str, Any]) -> EvalOptions:
    """Validate the launch form.

    Raises:
        ValueError: With a French message for the user.
    """
    kind = (form.get("kind") or "").strip()
    if kind not in EVAL_KINDS:
        raise ValueError("Type d'évaluation inconnu.")
    label = (form.get("label") or "").strip()
    if not valid_label(label):
        raise ValueError("Étiquette invalide : lettres, chiffres, « . », « _ » et « - », 40 caractères au plus.")
    course = (form.get("course") or "").strip() or None
    if course and course not in facts["courses"]:
        raise ValueError("Cours inconnu dans le jeu de questions.")
    qtype = (form.get("type") or "").strip() or None
    if qtype and qtype not in QUESTION_TYPES:
        raise ValueError("Type de question inconnu.")
    raw_limit = (form.get("limit") or "").strip()
    try:
        limit = int(raw_limit) if raw_limit else 0
    except ValueError:
        raise ValueError("La limite doit être un nombre entier.") from None
    if limit < 0:
        raise ValueError("La limite doit être positive (vide = toutes les questions).")
    try:
        k = int((form.get("k") or "5").strip())
    except ValueError:
        raise ValueError("k doit être un nombre entier.") from None
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k doit être compris entre 1 et {MAX_K}.")
    judge = form.get("judge") in ("1", "on", "true")
    provider = (form.get("provider") or "").strip() or None
    model = (form.get("model") or "").strip() or None
    if provider and provider not in {name for name, _ in available_providers(settings)}:
        raise ValueError("Fournisseur non configuré sur ce serveur.")
    if model and not MODEL_RE.match(model):
        raise ValueError("Nom de modèle invalide.")
    # Same model and provider as the server: no override at all (judge and fallback untouched).
    if (model or settings.llm_model_chat) == settings.llm_model_chat and (
        provider or settings.llm_provider
    ) == settings.llm_provider:
        model = provider = None
    judge_model, judge_provider = judge_defaults(settings)
    return EvalOptions(
        kind=kind,
        label=label,
        course=course,
        qtype=qtype,
        limit=limit,
        judge=judge if kind == KIND_ANSWER else False,
        k=k,
        model=model,
        provider=provider,
        judge_model=judge_model,
        judge_provider=judge_provider,
    )


async def _job_context(request: Request) -> dict[str, Any]:
    active = await jobs_db.active_job()
    runner = request.app.state.jobs
    return {
        "active_job": active,
        "busy": active is not None or runner.busy,
        "last_eval": await jobs_db.last_finished(jobs_db.KIND_EVAL),
        "labels": MODE_LABELS,
    }


def _form_context(request: Request) -> dict[str, Any]:
    s: Settings = request.app.state.settings
    judge_model, judge_provider = judge_defaults(s)
    facts = dataset_facts(s)
    return {
        "facts": facts,
        "kinds": [(k, MODE_LABELS[k]) for k in EVAL_KINDS],
        "types": facts["types"],
        "providers": available_providers(s),
        "server_model": s.llm_model_chat,
        "server_provider": s.llm_provider,
        "judge_model": judge_model,
        "judge_provider": judge_provider,
        "model_suggestions": sorted({s.llm_model_chat, *MODEL_SUGGESTIONS}),
        "default_limit": DEFAULT_LIMIT,
        "reports_writable": request.app.state.reports.writable(),
        "reports_dir": str(request.app.state.reports.directory),
        "form": {},
        "form_error": None,
    }


# ---------------------------------------------------------------------------
# Index: launch form, live job, history
# ---------------------------------------------------------------------------


@router.get("/evaluation", name="evaluation")
async def evaluation_index(request: Request, session: Session = Depends(current_user)):
    """Launch form (admin), running job, history of the saved reports."""
    kind = request.query_params.get("kind") or ""
    if kind not in KIND_LABELS:
        kind = ""
    reports = request.app.state.reports.list(kind or None)
    ctx = {
        "reports": reports,
        "kind_filter": kind,
        "kind_labels": KIND_LABELS,
        "flash": FLASH.get(request.query_params.get("ok", "")),
        **_form_context(request),
        **(await _job_context(request)),
    }
    return render(request, "evaluation/index.html", ctx)


@router.get("/evaluation/partials/launch", name="evaluation_launch_panel")
async def launch_panel(request: Request, session: Session = Depends(current_user)):
    """HTMX fragment: the running job (polled every 2 s) or, once it ends, the launch form."""
    ctx = {**_form_context(request), **(await _job_context(request))}
    return render(request, "evaluation/_launch.html", ctx)


@router.post("/evaluation/start", name="evaluation_start", dependencies=[Depends(csrf_protect)])
async def evaluation_start(request: Request, session: Session = Depends(require_admin)):
    """Validate the form and start the evaluation job."""
    form = {k: v for k, v in (await request.form()).items() if isinstance(v, str)}
    s: Settings = request.app.state.settings
    facts = dataset_facts(s)
    if facts["error"]:
        return _error_page(request, f"Jeu de questions illisible ({s.eval_dataset}) : {facts['error']}", 500)
    if not request.app.state.reports.writable():
        return _error_page(
            request,
            f"Le dossier des rapports ({request.app.state.reports.directory}) n'est pas inscriptible par le "
            "serveur : vérifiez le volume et son propriétaire (uid 1000 dans Docker).",
            500,
        )
    try:
        options = parse_options(form, s, facts)
    except ValueError as exc:
        ctx = {
            "reports": request.app.state.reports.list(),
            "kind_filter": "",
            "kind_labels": KIND_LABELS,
            "flash": None,
            **_form_context(request),
            **(await _job_context(request)),
        }
        ctx["form"] = form
        ctx["form_error"] = str(exc)
        return render(request, "evaluation/index.html", ctx, 400)
    if await jobs_db.active_job():
        return _error_page(request, "Un job est déjà en cours (ingestion ou évaluation) : attendez sa fin.", 409)
    try:
        job_id = request.app.state.jobs.start_eval(options, started_by=session.identifier)
    except JobBusyError:
        return _error_page(request, "Un job est déjà en cours (ingestion ou évaluation) : attendez sa fin.", 409)
    await audit.record(
        audit.EVAL_START, session.identifier, target=str(job_id), ip=client_ip(request), details=options.summary()
    )
    return RedirectResponse(f"{request.app.state.prefix}/jobs/{job_id}", status_code=303)


# ---------------------------------------------------------------------------
# One report
# ---------------------------------------------------------------------------


def _not_found(request: Request) -> Response:
    return render(
        request, "error.html", {"title": "Rapport introuvable", "message": "Aucun rapport ne porte ce nom."}, 404
    )


def _question_rows(kind: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-question rows, failures first, with a ``flag`` describing why a row deserves a look."""
    rows = []
    for q in report.get("questions") or []:
        flag = ""
        if kind == KIND_RETRIEVAL:
            if (q.get("found") or 0) == 0:
                flag = "non trouvée"
        elif kind == "code":
            if q.get("error"):
                flag = "erreur"
            elif not q.get("run_ok"):
                flag = "échec"
        else:
            if q.get("error"):
                flag = "erreur"
            elif q.get("phantom"):
                flag = "citation fantôme"
            elif q.get("fidelity") is not None and q["fidelity"] < 0.5:
                flag = "fidélité < 0,5"
            elif q.get("expected_behavior") == "answer" and q.get("refused"):
                flag = "refus indu"
        rows.append({**q, "flag": flag})
    rows.sort(key=lambda r: 0 if r["flag"] else 1)
    return rows


def _metric_rows(kind: str, report: dict[str, Any]) -> list[tuple[str, Any]]:
    metrics = report if kind == KIND_RETRIEVAL else (report.get("metrics") or {})
    return [(label, metrics.get(key)) for key, label, _ in METRIC_SPECS[kind] if key in metrics]


@router.get("/evaluation/reports/{name}", name="evaluation_report")
async def report_detail(request: Request, name: str, session: Session = Depends(current_user)):
    """Metrics, per-course table and per-question table of one report."""
    index = request.app.state.reports
    report = index.load(name)
    if report is None:
        return _not_found(request)
    try:
        kind = report_kind(report)
    except ReportKindError:
        return _error_page(request, "Ce fichier n'est pas un rapport d'évaluation reconnu.", 415)
    rows = _question_rows(kind, report)
    others = [s for s in index.list(kind) if s.name != name and s.valid]
    ctx = {
        "name": name,
        "report": report,
        "kind": kind,
        "kind_label": KIND_LABELS.get(kind, kind),
        "summary": index.get(name),
        "metric_rows": _metric_rows(kind, report),
        "by_course": sorted((report.get("by_course") or {}).items()),
        "rows": rows,
        "n_flagged": sum(1 for r in rows if r["flag"]),
        "n_failures": count_failures(kind, report),
        "others": others,
        "flash": FLASH.get(request.query_params.get("ok", "")),
    }
    return render(request, "evaluation/report.html", ctx)


@router.get("/evaluation/reports/{name}/download", name="evaluation_download")
async def report_download(request: Request, name: str, session: Session = Depends(current_user)):
    """Download the JSON file."""
    path = request.app.state.reports.path(name)
    if path is None:
        return _not_found(request)
    return FileResponse(path, media_type="application/json", filename=path.name)


@router.get("/evaluation/reports/{name}/questions.csv", name="evaluation_csv")
async def report_csv(request: Request, name: str, session: Session = Depends(current_user)):
    """Per-question table as CSV."""
    report = request.app.state.reports.load(name)
    if report is None:
        return _not_found(request)
    try:
        kind = report_kind(report)
    except ReportKindError:
        return _error_page(request, "Ce fichier n'est pas un rapport d'évaluation reconnu.", 415)
    rows = _question_rows(kind, report)
    if kind == KIND_RETRIEVAL:
        columns = [
            ("id", "Id"),
            ("course", "Cours"),
            ("type", "Type"),
            ("difficulty", "Difficulté"),
            ("expected", "Slides attendues"),
            ("found", "Trouvées"),
            ("first_rank", "Premier rang"),
            ("candidate_rank", "Rang dans le pool"),
            ("latency_s", "Latence (s)"),
            ("flag", "À regarder"),
        ]
    elif kind == "code":
        columns = [
            ("id", "Id"),
            ("course", "Cours"),
            ("generated", "Script généré"),
            ("syntax_ok", "Syntaxe OK"),
            ("run_ok_first", "Exécution OK (1er essai)"),
            ("run_ok", "Exécution OK"),
            ("n_fixes", "Corrections"),
            ("has_output", "Sortie chiffrée"),
            ("has_figure", "Figure"),
            ("terms_found", "Termes présents"),
            ("latency_generation_s", "Latence génération (s)"),
            ("latency_run_s", "Latence exécution (s)"),
            ("error", "Erreur"),
            ("flag", "À regarder"),
        ]
    else:
        columns = [
            ("id", "Id"),
            ("course", "Cours"),
            ("type", "Type"),
            ("difficulty", "Difficulté"),
            ("latency_s", "Latence (s)"),
            ("n_searches", "Recherches"),
            ("n_tool_calls", "Appels d'outils"),
            ("fidelity", "Fidélité"),
            ("phantom", "Citation fantôme"),
            ("refused", "Refus"),
            ("general_knowledge", "Connaissances générales"),
            ("error", "Erreur"),
            ("flag", "À regarder"),
        ]
    return csv_response(f"eval-{name[:-5]}", columns, rows)


@router.post("/evaluation/reports/{name}/delete", name="evaluation_delete", dependencies=[Depends(csrf_protect)])
async def report_delete(request: Request, name: str, session: Session = Depends(require_admin)):
    """Delete one report file."""
    if not request.app.state.reports.delete(name):
        return _not_found(request)
    await audit.record(audit.EVAL_DELETE, session.identifier, target=name, ip=client_ip(request))
    return RedirectResponse(f"{request.app.state.prefix}/evaluation?ok=deleted", status_code=303)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@router.get("/evaluation/compare", name="evaluation_compare")
async def compare_page(request: Request, session: Session = Depends(current_user)):
    """Choose two reports of the same kind and show the before/after table."""
    index = request.app.state.reports
    a = request.query_params.get("a") or ""
    b = request.query_params.get("b") or ""
    reports = [s for s in index.list() if s.valid]
    ctx: dict[str, Any] = {
        "reports": reports,
        "a": a,
        "b": b,
        "kind_labels": KIND_LABELS,
        "comparison": None,
        "error": None,
        "metric_labels": METRIC_LABELS,
    }
    if a and b:
        ra, rb = index.load(a), index.load(b)
        if ra is None or rb is None:
            ctx["error"] = "Un des deux rapports est introuvable."
        else:
            try:
                ctx["comparison"] = compare_reports(ra, rb)
            except ReportKindError as exc:
                ctx["error"] = str(exc).capitalize() + "."
    elif a or b:
        ctx["error"] = "Choisissez deux rapports."
    return render(request, "evaluation/compare.html", ctx)
