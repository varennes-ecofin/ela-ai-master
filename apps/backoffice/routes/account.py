"""The signed-in person's own account (password change)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse

from backoffice.deps import client_ip, csrf_protect, current_user
from backoffice.rendering import render
from backoffice.security import Session
from ela.auth import passwords
from ela.db import audit
from ela.db import users as users_db

router = APIRouter()


@router.get("/account/password", name="account_password")
async def account_password_form(request: Request, session: Session = Depends(current_user)):
    """Form: current password, new password twice."""
    return render(
        request,
        "account_password.html",
        {"policy": passwords.describe_policy(), "done": request.query_params.get("ok") == "1"},
    )


@router.post("/account/password", name="account_password_submit", dependencies=[Depends(csrf_protect)])
async def account_password_submit(
    request: Request,
    current: str = Form(default=""),
    password: str = Form(default=""),
    confirm: str = Form(default=""),
    session: Session = Depends(current_user),
):
    """Check the current password and the policy, then replace the hash."""
    errors: list[str] = []
    stored = await users_db.get_password_hash(session.identifier)
    if not stored or not passwords.check_password(current, stored):
        errors.append("Mot de passe actuel incorrect.")
    if issues := passwords.problems(password):
        errors.append("Nouveau mot de passe trop faible : " + ", ".join(issues) + ".")
    if password != confirm:
        errors.append("Les deux saisies du nouveau mot de passe diffèrent.")
    if password and password == current:
        errors.append("Le nouveau mot de passe doit différer de l'actuel.")
    if errors:
        return render(request, "account_password.html", {"policy": passwords.describe_policy(), "errors": errors}, 400)
    await users_db.set_password_hash(session.identifier, passwords.hash_password(password))
    await audit.record(audit.ACCOUNT_PASSWORD, session.identifier, target=session.identifier, ip=client_ip(request))
    return RedirectResponse(f"{request.app.state.prefix}/account/password?ok=1", status_code=303)
