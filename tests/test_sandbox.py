"""Tests for the Atelier Code 2.0: runner, HTTP service, client, execution service, agent tool."""

from __future__ import annotations

import base64
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

import ela.sandbox.service as service
from ela.config import get_settings
from ela.sandbox.client import RunResult, SandboxUnavailableError, run_code
from fakes import ScriptedChatModel
from test_agent_tools import ctx  # noqa: F401  (fixture)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "sandbox"))
from runner import RunLimits, run_python  # noqa: E402

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


# ---------------------------------------------------------------- runner (apps/sandbox)


def test_runner_captures_output_errors_and_figures():
    ok = run_python("import numpy as np\nprint(np.arange(4).sum())")
    assert ok.status == "ok" and ok.stdout.strip() == "6" and ok.exit_code == 0 and ok.images == []
    err = run_python("x = 1 / 0")
    assert err.status == "error" and err.exit_code == 1 and "ZeroDivisionError" in err.stderr
    figs = run_python("import matplotlib.pyplot as plt\nplt.plot([1, 2])\nplt.figure()\nplt.plot([2, 1])\nprint('ok')")
    assert figs.status == "ok" and len(figs.images) == 2
    assert base64.b64decode(figs.images[0])[:8] == b"\x89PNG\r\n\x1a\n"


def test_runner_enforces_cpu_and_wall_limits():
    cpu = run_python("while True: pass", RunLimits(timeout_seconds=10, cpu_seconds=1))
    assert cpu.status == "timeout" and "interrompue" in cpu.stderr
    wall = run_python("import time\ntime.sleep(5)", RunLimits(timeout_seconds=1))
    assert wall.status == "timeout" and "1 s" in wall.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="memory limits rely on resource.setrlimit (POSIX only)")
def test_runner_enforces_memory_limit():
    mem = run_python("x = bytearray(900_000_000)", RunLimits(memory_bytes=256 * 1024 * 1024))
    assert mem.status in {"error", "timeout"}


def test_runner_isolates_the_script():
    out = run_python("import sys, os\nprint(sys.flags.isolated, os.environ.get('CEREBRAS_API_KEY', 'none'))")
    assert out.stdout.strip() == "1 none"


# ---------------------------------------------------------------- HTTP service


@pytest.fixture(scope="module")
def sandbox_server():
    import server as sandbox_server_mod

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), sandbox_server_mod.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


async def test_service_runs_code_over_http(sandbox_server, monkeypatch):
    monkeypatch.setenv("SANDBOX_URL", sandbox_server)
    get_settings.cache_clear()
    try:
        async with httpx.AsyncClient() as client:
            health = await client.get(sandbox_server + "/health")
            assert health.json() == {"status": "ok"}
            bad = await client.post(sandbox_server + "/run", json={"code": ""})
            assert bad.status_code == 400
            r = await client.post(sandbox_server + "/run", json={"code": "print(2 + 2)", "language": "R"})
            assert r.status_code == 400
        result = await run_code("print(2 + 2)")
        assert result.ok and result.stdout.strip() == "4" and result.elapsed_ms >= 0
    finally:
        get_settings.cache_clear()


# ---------------------------------------------------------------- client


def _transport(handler):
    return httpx.MockTransport(handler)


async def test_client_parses_payload_and_errors(monkeypatch):
    monkeypatch.setenv("SANDBOX_URL", "http://sandbox:8080/")
    get_settings.cache_clear()
    try:
        seen = {}

        def ok_handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            payload = {
                "status": "ok",
                "stdout": "x",
                "stderr": "",
                "exit_code": 0,
                "elapsed_ms": 12,
                "images": [base64.b64encode(PNG_1PX).decode()],
            }
            return httpx.Response(200, json=payload)

        result = await run_code("print('x')", transport=_transport(ok_handler))
        assert seen["url"] == "http://sandbox:8080/run" and seen["body"]["language"] == "python"
        assert result.ok and result.images == [PNG_1PX]

        with pytest.raises(SandboxUnavailableError, match="refused"):
            await run_code("x", transport=_transport(lambda r: httpx.Response(413, json={"error": "too long"})))

        def down(request):
            raise httpx.ConnectError("down")

        with pytest.raises(SandboxUnavailableError, match="unreachable"):
            await run_code("x", transport=_transport(down))
    finally:
        get_settings.cache_clear()


async def test_client_disabled_without_url(monkeypatch):
    monkeypatch.delenv("SANDBOX_URL", raising=False)
    get_settings.cache_clear()
    with pytest.raises(SandboxUnavailableError, match="not configured"):
        await run_code("x")


# ---------------------------------------------------------------- execution service


def test_extract_code_prefers_the_language_fence():
    text = "Intro\n```r\nx <- 1\n```\nthen\n```python\nprint(1)\n```\nend"
    assert service.extract_code(text, "python") == "print(1)"
    assert service.extract_code(text, "r") == "x <- 1"
    assert service.extract_code("```\nraw\n```") == "raw"
    assert service.extract_code("no code") is None
    assert service.extract_code("```python\n\n```") is None


def test_format_result_and_summary():
    ok = RunResult("ok", stdout="a\nb", elapsed_ms=1500, images=[PNG_1PX])
    text = service.format_result(ok)
    assert (
        text.startswith("✅ **Exécution terminée** en 1.5 s") and "```text\na\nb\n```" in text and "1 graphique" in text
    )
    err = RunResult("error", stderr="\n".join(f"line {i}" for i in range(40)), exit_code=1)
    text = service.format_result(err)
    assert text.startswith("❌ **Le script a échoué** (code 1)") and "…\nline 15" in text and "line 39" in text
    assert service.format_result(RunResult("timeout")).startswith("⏱️")
    summary = service.summary_for_model(err)
    assert summary.startswith("Statut : error") and "ERREUR :" in summary and "commente-la" in summary


@pytest.fixture
def fake_runs(monkeypatch):
    """Sandbox client and persistence replaced by fakes; returns the recorded runs."""
    recorded = []
    state = {"today": 0, "result": RunResult("ok", stdout="42", elapsed_ms=5)}

    async def _run(code, *, language="python"):
        if state["result"] is None:
            raise SandboxUnavailableError("down")
        return state["result"]

    async def _record(run):
        recorded.append(run)
        return True

    async def _today(user):
        return state["today"]

    monkeypatch.setattr(service, "run_code", _run)
    monkeypatch.setattr(service, "record_run", _record)
    monkeypatch.setattr(service, "runs_today", _today)
    monkeypatch.setenv("SANDBOX_URL", "http://sandbox:8080")
    monkeypatch.setenv("SANDBOX_DAILY_QUOTA", "3")
    get_settings.cache_clear()
    yield recorded, state
    get_settings.cache_clear()


async def test_execute_records_runs_and_enforces_quota(fake_runs):
    recorded, state = fake_runs
    result = await service.execute("print(42)", user_id="alice", thread_id="t", origin="workshop", topic="AR(1)")
    assert result.stdout == "42" and recorded[-1].status == "ok" and recorded[-1].topic == "AR(1)"
    assert await service.quota_remaining("alice") == 3 and await service.quota_remaining(None) is None
    state["today"] = 3
    with pytest.raises(service.QuotaExceededError) as exc:
        await service.execute("print(1)", user_id="alice")
    assert exc.value.limit == 3
    await service.execute("print(1)", user_id=None)  # anonymous: no quota, nothing recorded
    assert len(recorded) == 1
    state["today"], state["result"] = 0, None
    with pytest.raises(SandboxUnavailableError):
        await service.execute("print(1)", user_id="alice")
    assert recorded[-1].status == "unavailable"


async def test_fix_code_returns_the_repaired_block():
    llm = ScriptedChatModel(["```python\nimport numpy as np\nprint(np.pi)\n```"])
    fixed = await service.fix_code(llm, "print(np.pi)", "NameError: name 'np' is not defined", topic="pi")
    assert fixed == "import numpy as np\nprint(np.pi)"
    prompt = llm.calls[0][0].content
    assert "sujet : « pi »" in prompt and "NameError" in llm.calls[0][1].content
    with pytest.raises(ValueError):
        await service.fix_code(ScriptedChatModel(["pas de code"]), "x", "err")


# ---------------------------------------------------------------- agent tool


async def test_run_python_tool(fake_runs):
    from ela.agent.context import TurnContext
    from ela.agent.tools import build_tools, run_python
    from test_agent_tools import _runtime

    recorded, state = fake_runs
    assert [t.name for t in build_tools()][-1] == "run_python"
    assert "run_python" not in [t.name for t in build_tools(sandbox_enabled=False)]
    turn = TurnContext(level="M1", user_id="alice", thread_id="t")
    out = await run_python.ainvoke({"code": "print(42)", "runtime": _runtime(turn)})
    assert out.startswith("Statut : ok") and "SORTIE :\n42" in out
    assert turn.code_runs[0][0] == "print(42)" and recorded[-1].origin == "agent"
    state["today"] = 3
    out = await run_python.ainvoke({"code": "print(1)", "runtime": _runtime(turn)})
    assert out.startswith("Exécution refusée : l'étudiant a atteint son quota de 3")
    state["today"], state["result"] = 0, None
    out = await run_python.ainvoke({"code": "print(1)", "runtime": _runtime(turn)})
    assert out.startswith("Le bac à sable d'exécution est indisponible")


def test_code_block_in_system_prompt(monkeypatch):
    from ela.agent.middleware import code_block

    assert code_block(enabled=False) == ""
    assert code_block(enabled=True).startswith("CALCUL NUMÉRIQUE")
    monkeypatch.delenv("SANDBOX_URL", raising=False)
    get_settings.cache_clear()
    assert code_block() == ""


async def test_engine_emits_code_run_event(fake_runs):
    from langchain_core.messages import AIMessage

    from ela.agent.agent import build_agent
    from ela.agent.events import CodeRunEvent, FinalEvent
    from ela.engine import ElaEngine
    from fakes import tool_call_message

    model = ScriptedChatModel(
        [tool_call_message(("run_python", {"code": "print(42)"})), AIMessage(content="Le résultat vaut 42.")]
    )
    agent = build_agent(model=model)
    eng = ElaEngine("M1", retriever=object(), llm=model, agent=agent)
    events = [e async for e in eng.ask("calcule", user_id="alice", session_id="t")]
    runs = [e for e in events if isinstance(e, CodeRunEvent)]
    assert len(runs) == 1 and runs[0].code == "print(42)" and runs[0].result.stdout == "42"
    assert isinstance(events[-1], FinalEvent) and events[-1].text == "Le résultat vaut 42."


# ---------------------------------------------------------------- interpretation


def test_has_numbers_and_figure_titles():
    assert service.has_numbers("alpha = 0.05") and not service.has_numbers("done\n")
    code = 'axes[0].set_title("Rendements simulés")\nplt.title(f"Volatilité {x}")'
    assert service.figure_titles(code, 0) == "aucune"
    assert service.figure_titles(code, 2) == "2 — figure 1 « Rendements simulés »; figure 2 « Volatilité {x} »"
    assert service.figure_titles("plt.plot(x)", 1) == "1 (sans titre repéré)"
    assert service.figure_titles(code, 3).startswith("3 (titres repérés : ")


async def test_interpret_run_builds_table_with_verified_citations(ctx):  # noqa: F811
    from ela.sandbox.service import interpret_run

    llm = ScriptedChatModel(["| paramètre | estimation |\n| phi | 0.78 |\n\nLa stationnarité [S1] et un fantôme [S7]."])
    result = RunResult("ok", stdout="phi_hat = 0.78\nse = 0.02", images=[PNG_1PX])
    interp = await interpret_run(
        llm, ctx.get_retriever(), 'plt.title("AR(1)")\nprint(phi)', result, topic="stationnarité", level="M1"
    )
    assert not interp.needs_output and "[S1]" in interp.text and "[S7]" not in interp.text
    assert [r.sid for r in interp.cited] == ["S1"]
    prompt = llm.calls[0][0].content
    assert "phi_hat = 0.78" in prompt and "FIGURES : 1 — figure 1 « AR(1) »" in prompt and "[S1]" in prompt
    assert "niveau M1" in prompt and "sujet : « stationnarité »" in prompt


async def test_interpret_run_without_numbers_asks_for_output(ctx):  # noqa: F811
    from ela.sandbox.service import interpret_run

    llm = ScriptedChatModel([])
    interp = await interpret_run(llm, ctx.get_retriever(), "plt.plot(x)", RunResult("ok", stdout="done"), topic="x")
    assert interp.needs_output and "Compléter le script" in interp.text and llm.calls == []


def test_runner_prints_unicode_whatever_the_console_encoding(monkeypatch):
    monkeypatch.setenv("PYTHONIOENCODING", "cp1252")  # ignored by -I anyway; the script must still write UTF-8
    out = run_python("print('modèle ‑ α+β ≈ 0,9')")
    assert out.status == "ok" and out.stdout.strip() == "modèle ‑ α+β ≈ 0,9"


def test_syntax_error_and_history_block():
    assert service.syntax_error("x = 1\nprint(x)") is None
    err = service.syntax_error("print('hi'")
    assert err is not None and err.startswith("SyntaxError (ligne 1)")
    assert service.syntax_error("x <- 1", "R") is None
    assert service.history_block([]) == ""
    block = service.history_block(["Traceback\n  File x\nNameError: np", "ValueError: boom"])
    assert "ERREURS DÉJÀ RENCONTRÉES" in block and "NameError: np" in block and "ValueError: boom" in block


async def test_fix_code_passes_history_to_the_prompt():
    llm = ScriptedChatModel(["```python\nprint(1)\n```"])
    await service.fix_code(llm, "print(1", "SyntaxError", topic="t", history=["NameError: np"])
    assert "ERREURS DÉJÀ RENCONTRÉES" in llm.calls[0][0].content and "NameError: np" in llm.calls[0][0].content


def test_with_overrides_changes_effort_only_when_present():
    from langchain_openai import ChatOpenAI

    from ela.structured import unwrap_fallbacks, with_overrides

    with_effort = ChatOpenAI(model="m", api_key="k", max_tokens=100, reasoning_effort="low")
    without = ChatOpenAI(model="f", api_key="k", max_tokens=100)
    out = with_overrides(with_effort.with_fallbacks([without]), max_tokens=4096, reasoning_effort="medium")
    p, fbs = unwrap_fallbacks(out)
    assert p.max_tokens == 4096 and p.reasoning_effort == "medium"
    assert fbs[0].max_tokens == 4096 and fbs[0].reasoning_effort is None  # provider without the parameter
