"""Accounts: list, create, edit, passwords, activation, purge, CSV import (lot 5.2).

Reading is open to ``admin`` and ``supervisor``; every change requires
``admin``, a CSRF token, and is written to the audit log.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import PlainTextResponse, RedirectResponse, Response

from backoffice.deps import client_ip, csrf_protect, current_user, require_admin
from backoffice.rendering import render
from backoffice.security import Session
from ela.accounts import importer
from ela.auth import passwords
from ela.db import audit
from ela.db import users as users_db

logger = logging.getLogger("ela.backoffice.users")
router = APIRouter()

FLASH = {
    "created": "Compte créé.",
    "updated": "Modifications enregistrées.",
    "password": "Mot de passe remplacé.",
    "activated": "Compte réactivé.",
    "deactivated": "Compte désactivé.",
    "purged": "Compte supprimé.",
}
MAX_CSV_BYTES = 2 * 1024 * 1024


def _quota_from_form(value: str) -> int | None:
    """``0`` or empty means unlimited; otherwise a positive integer."""
    value = (value or "").strip()
    if not value or value == "0":
        return None
    quota = int(value)
    if quota < 0 or quota > 10_000:
        raise ValueError("quota hors limites")
    return quota


def _prefix(request: Request) -> str:
    return request.app.state.prefix


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


@router.get("/users", name="users_list")
async def users_list(
    request: Request,
    q: str = "",
    role: str = "",
    level: str = "",
    state: str = "",
    ok: str = "",
    session: Session = Depends(current_user),
):
    """List the accounts in a filterable table (an HTMX request only gets the table fragment)."""
    active = {"active": True, "inactive": False}.get(state)
    accounts = await users_db.list_accounts(search=q or None, role=role or None, level=level or None, active=active)
    context = {
        "accounts": accounts,
        "filters": {"q": q, "role": role, "level": level, "state": state},
        "roles": users_db.ROLES,
        "levels": users_db.LEVELS,
        "flash": FLASH.get(ok),
    }
    template = "users/_table.html" if request.headers.get("hx-request") == "true" else "users/list.html"
    return render(request, template, context)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


@router.get("/users/new", name="user_new")
async def user_new(request: Request, session: Session = Depends(require_admin)):
    """Creation form."""
    return render(
        request,
        "users/form.html",
        {
            "account": None,
            "form": {"role": "student", "level": "M1", "quota": "50"},
            "roles": users_db.ROLES,
            "levels": users_db.LEVELS,
            "policy": passwords.describe_policy(),
        },
    )


@router.post("/users/new", name="user_create", dependencies=[Depends(csrf_protect)])
async def user_create(
    request: Request,
    identifier: str = Form(default=""),
    role: str = Form(default="student"),
    level: str = Form(default="M1"),
    quota: str = Form(default="50"),
    password: str = Form(default=""),
    generate: str = Form(default=""),
    session: Session = Depends(require_admin),
):
    """Create an account; the password is typed or generated (shown once)."""
    form = {"identifier": identifier, "role": role, "level": level, "quota": quota}
    errors: list[str] = []
    ident = users_db.normalize_identifier(identifier)
    if problem := users_db.identifier_problem(ident):
        errors.append(problem.capitalize() + ".")
    if role not in users_db.ROLES or level not in users_db.LEVELS:
        errors.append("Rôle ou niveau invalide.")
    try:
        daily_quota = _quota_from_form(quota)
    except ValueError:
        errors.append("Quota invalide (0 = illimité).")
        daily_quota = None
    generated = bool(generate)
    if generated:
        password = passwords.generate()
    elif not password:
        errors.append("Saisissez un mot de passe ou demandez-en un généré.")
    elif issues := passwords.problems(password):
        errors.append("Mot de passe trop faible : " + ", ".join(issues) + ".")

    if errors:
        return render(
            request,
            "users/form.html",
            {
                "account": None,
                "form": form,
                "errors": errors,
                "roles": users_db.ROLES,
                "levels": users_db.LEVELS,
                "policy": passwords.describe_policy(),
            },
            400,
        )
    try:
        await users_db.create_account(
            ident, password_hash=passwords.hash_password(password), role=role, level=level, daily_quota=daily_quota
        )
    except users_db.AccountExistsError:
        return render(
            request,
            "users/form.html",
            {
                "account": None,
                "form": form,
                "errors": [f"L'identifiant « {ident} » existe déjà."],
                "roles": users_db.ROLES,
                "levels": users_db.LEVELS,
                "policy": passwords.describe_policy(),
            },
            409,
        )
    await audit.record(
        audit.USER_CREATE,
        session.identifier,
        target=ident,
        ip=client_ip(request),
        details={"role": role, "level": level, "quota": daily_quota, "generated_password": generated},
    )
    url = f"{_prefix(request)}/users/{ident}?ok=created"
    if generated:
        token = request.app.state.oneshot.put(password, owner=session.identifier)
        url += f"&secret={token}"
    return RedirectResponse(url, status_code=303)


# ---------------------------------------------------------------------------
# CSV import
# ---------------------------------------------------------------------------


def _import_context(**extra) -> dict:
    ctx = {
        "levels": users_db.LEVELS,
        "form": {"level": "", "quota": "50", "update_existing": False},
        "policy": passwords.describe_policy(),
    }
    ctx.update(extra)
    return ctx


@router.get("/users/import", name="users_import")
async def users_import_form(request: Request, session: Session = Depends(require_admin)):
    """Upload form and format description."""
    return render(request, "users/import.html", _import_context())


@router.post("/users/import", name="users_import_preview", dependencies=[Depends(csrf_protect)])
async def users_import_preview(
    request: Request,
    file: UploadFile = File(...),
    level: str = Form(default=""),
    quota: str = Form(default="50"),
    update_existing: str = Form(default=""),
    session: Session = Depends(require_admin),
):
    """Parse the file, check every line against the database and show the plan."""
    form = {"level": level, "quota": quota, "update_existing": bool(update_existing)}
    raw = await file.read(MAX_CSV_BYTES + 1)
    if len(raw) > MAX_CSV_BYTES:
        return render(
            request,
            "users/import.html",
            _import_context(form=form, errors=["Fichier trop volumineux (2 Mo maximum)."]),
            400,
        )
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    try:
        default_quota = _quota_from_form(quota)
    except ValueError:
        return render(
            request, "users/import.html", _import_context(form=form, errors=["Quota invalide (0 = illimité)."]), 400
        )
    if level and level not in users_db.LEVELS:
        return render(
            request, "users/import.html", _import_context(form=form, errors=["Niveau par défaut invalide."]), 400
        )
    options = {"default_level": level or None, "default_quota": default_quota, "update_existing": bool(update_existing)}
    plan = await importer.plan_import(text, **options)
    token = None
    if plan.applicable:
        token = request.app.state.oneshot.put(
            {"text": text, "options": options, "filename": file.filename}, owner=session.identifier
        )
    return render(
        request,
        "users/import_preview.html",
        _import_context(form=form, plan=plan, token=token, filename=file.filename),
    )


@router.post("/users/import/{token}/apply", name="users_import_apply", dependencies=[Depends(csrf_protect)])
async def users_import_apply(request: Request, token: str, session: Session = Depends(require_admin)):
    """Create/update the accounts of a previewed import; generated passwords are shown once."""
    pending = request.app.state.oneshot.pop(token, owner=session.identifier)
    if pending is None:
        return render(
            request,
            "users/import.html",
            _import_context(errors=["Prévisualisation expirée : recommencez l'import."]),
            410,
        )
    plan = await importer.plan_import(pending["text"], **pending["options"])  # the database may have changed
    result = await importer.apply_import(plan)
    await audit.record(
        audit.USERS_IMPORT,
        session.identifier,
        ip=client_ip(request),
        details={
            "file": pending.get("filename"),
            "created": len(result.created),
            "updated": len(result.updated),
            "failed": len(result.failed),
            "generated_passwords": len(result.credentials),
        },
    )
    credentials_token = None
    if result.credentials:
        credentials_token = request.app.state.oneshot.put(result.credentials_csv(), owner=session.identifier)
    return render(
        request, "users/import_result.html", _import_context(result=result, credentials_token=credentials_token)
    )


@router.get("/users/import/{token}/credentials.csv", name="users_import_credentials")
async def users_import_credentials(request: Request, token: str, session: Session = Depends(require_admin)):
    """One-time download of the generated credentials."""
    csv_text = request.app.state.oneshot.pop(token, owner=session.identifier)
    if csv_text is None:
        return render(
            request,
            "error.html",
            {"title": "Lien expiré", "message": "Ce fichier a déjà été téléchargé ou a expiré."},
            410,
        )
    return PlainTextResponse(
        csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="ela-identifiants.csv"', "Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# Detail and edit
# ---------------------------------------------------------------------------


async def _detail_context(request: Request, session: Session, account: users_db.Account, **extra) -> dict:
    activity = await users_db.activity_summary(account.identifier)
    admins = await users_db.count_admins()
    is_self = account.identifier == session.identifier
    last_admin = account.role == "admin" and account.is_active and admins <= 1
    secret = None
    token = request.query_params.get("secret")
    if token:
        secret = request.app.state.oneshot.pop(token, owner=session.identifier)
    ctx = {
        "account": account,
        "activity": activity,
        "has_activity": any(activity.values()),
        "is_self": is_self,
        "last_admin": last_admin,
        "roles": users_db.ROLES,
        "levels": users_db.LEVELS,
        "flash": FLASH.get(request.query_params.get("ok", "")),
        "secret": secret,
        "policy": passwords.describe_policy(),
        "form": {
            "role": account.role,
            "level": account.level,
            "quota": "0" if account.daily_quota is None else str(account.daily_quota),
        },
    }
    ctx.update(extra)
    return ctx


async def _load(request: Request, identifier: str) -> users_db.Account | Response:
    account = await users_db.get_account(identifier)
    if account is None:
        return render(
            request, "error.html", {"title": "Compte introuvable", "message": f"Aucun compte « {identifier} »."}, 404
        )
    return account


@router.get("/users/{identifier}", name="user_detail")
async def user_detail(request: Request, identifier: str, session: Session = Depends(current_user)):
    """Account page (read-only for supervisors)."""
    account = await _load(request, identifier)
    if isinstance(account, Response):
        return account
    return render(request, "users/detail.html", await _detail_context(request, session, account))


@router.post("/users/{identifier}", name="user_update", dependencies=[Depends(csrf_protect)])
async def user_update(
    request: Request,
    identifier: str,
    role: str = Form(default=""),
    level: str = Form(default=""),
    quota: str = Form(default=""),
    session: Session = Depends(require_admin),
):
    """Change role, level and quota."""
    account = await _load(request, identifier)
    if isinstance(account, Response):
        return account
    errors: list[str] = []
    is_self = account.identifier == session.identifier
    if role not in users_db.ROLES or level not in users_db.LEVELS:
        errors.append("Rôle ou niveau invalide.")
    try:
        daily_quota = _quota_from_form(quota)
    except ValueError:
        errors.append("Quota invalide (0 = illimité).")
        daily_quota = account.daily_quota
    if is_self and role != account.role:
        errors.append("Vous ne pouvez pas changer votre propre rôle.")
    if account.role == "admin" and role != "admin" and account.is_active and await users_db.count_admins() <= 1:
        errors.append("Impossible de rétrograder le dernier administrateur actif.")
    if errors:
        ctx = await _detail_context(request, session, account, errors=errors)
        ctx["form"] = {"role": role, "level": level, "quota": quota}
        return render(request, "users/detail.html", ctx, 400)
    changes = {}
    if role != account.role:
        changes["role"] = role
    if level != account.level:
        changes["level"] = level
    if daily_quota != account.daily_quota:
        changes["quota"] = daily_quota
    await users_db.update_account(identifier, role=role, level=level, daily_quota=daily_quota)
    if changes:
        await audit.record(
            audit.USER_UPDATE, session.identifier, target=identifier, ip=client_ip(request), details=changes
        )
    return RedirectResponse(f"{_prefix(request)}/users/{identifier}?ok=updated", status_code=303)


@router.post("/users/{identifier}/password", name="user_password", dependencies=[Depends(csrf_protect)])
async def user_password(
    request: Request,
    identifier: str,
    password: str = Form(default=""),
    generate: str = Form(default=""),
    session: Session = Depends(require_admin),
):
    """Replace the password (typed, or generated and shown once)."""
    account = await _load(request, identifier)
    if isinstance(account, Response):
        return account
    generated = bool(generate)
    if generated:
        password = passwords.generate()
    elif not password:
        ctx = await _detail_context(
            request, session, account, errors=["Saisissez un mot de passe ou demandez-en un généré."]
        )
        return render(request, "users/detail.html", ctx, 400)
    elif issues := passwords.problems(password):
        ctx = await _detail_context(
            request, session, account, errors=["Mot de passe trop faible : " + ", ".join(issues) + "."]
        )
        return render(request, "users/detail.html", ctx, 400)
    await users_db.set_password_hash(identifier, passwords.hash_password(password))
    await audit.record(
        audit.USER_PASSWORD_RESET,
        session.identifier,
        target=identifier,
        ip=client_ip(request),
        details={"generated": generated},
    )
    url = f"{_prefix(request)}/users/{identifier}?ok=password"
    if generated:
        url += "&secret=" + request.app.state.oneshot.put(password, owner=session.identifier)
    return RedirectResponse(url, status_code=303)


@router.post("/users/{identifier}/active", name="user_active", dependencies=[Depends(csrf_protect)])
async def user_active(
    request: Request, identifier: str, active: str = Form(default=""), session: Session = Depends(require_admin)
):
    """Activate (``active=1``) or deactivate (``active=0``) an account."""
    account = await _load(request, identifier)
    if isinstance(account, Response):
        return account
    wanted = active == "1"
    errors: list[str] = []
    if account.identifier == session.identifier and not wanted:
        errors.append("Vous ne pouvez pas désactiver votre propre compte.")
    if not wanted and account.role == "admin" and account.is_active and await users_db.count_admins() <= 1:
        errors.append("Impossible de désactiver le dernier administrateur actif.")
    if errors:
        return render(
            request, "users/detail.html", await _detail_context(request, session, account, errors=errors), 400
        )
    if wanted != account.is_active:
        await users_db.update_account(identifier, is_active=wanted)
        await audit.record(
            audit.USER_ACTIVATE if wanted else audit.USER_DEACTIVATE,
            session.identifier,
            target=identifier,
            ip=client_ip(request),
        )
    return RedirectResponse(
        f"{_prefix(request)}/users/{identifier}?ok={'activated' if wanted else 'deactivated'}", status_code=303
    )


@router.post("/users/{identifier}/purge", name="user_purge", dependencies=[Depends(csrf_protect)])
async def user_purge(request: Request, identifier: str, session: Session = Depends(require_admin)):
    """Delete an account without any activity (confirmation happens in the page)."""
    account = await _load(request, identifier)
    if isinstance(account, Response):
        return account
    if account.identifier == session.identifier:
        ctx = await _detail_context(
            request, session, account, errors=["Vous ne pouvez pas supprimer votre propre compte."]
        )
        return render(request, "users/detail.html", ctx, 400)
    try:
        await users_db.purge_account(identifier)
    except users_db.AccountHasActivityError:
        ctx = await _detail_context(
            request, session, account, errors=["Ce compte a une activité : désactivez-le plutôt que de le supprimer."]
        )
        return render(request, "users/detail.html", ctx, 409)
    await audit.record(
        audit.USER_PURGE, session.identifier, target=identifier, ip=client_ip(request), details={"role": account.role}
    )
    return RedirectResponse(f"{_prefix(request)}/users?ok=purged", status_code=303)
