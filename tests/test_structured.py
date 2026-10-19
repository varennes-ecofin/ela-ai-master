"""Tests for ``ela.structured`` (structured output with JSON fallback)."""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from ela.structured import ainvoke_structured, extract_json


class Item(BaseModel):
    name: str
    n: int


def test_extract_json_variants():
    assert extract_json('Voici :\n```json\n{"a": 1}\n```\nfin') == '{"a": 1}'
    assert extract_json('texte {"a": {"b": 2}} suite') == '{"a": {"b": 2}}'
    assert extract_json("[1, 2] et {}") == "[1, 2]"
    with pytest.raises(ValueError):
        extract_json("rien")


async def test_fallback_to_json_text_when_structured_output_unavailable():
    # GenericFakeChatModel has no bind_tools → with_structured_output raises → JSON fallback
    llm = GenericFakeChatModel(messages=iter(['```json\n{"name": "AR(1)", "n": 3}\n```']))
    item = await ainvoke_structured(llm, Item, [HumanMessage(content="go")])
    assert item == Item(name="AR(1)", n=3)


async def test_invalid_json_raises():
    llm = GenericFakeChatModel(messages=iter(['{"name": "x"}']))
    with pytest.raises(ValueError):
        await ainvoke_structured(llm, Item, [HumanMessage(content="go")])


def test_repair_escapes_doubles_lone_backslashes():
    from ela.structured import repair_escapes

    assert repair_escapes(r'{"q": "$\hat{\rho}$ et \"ok\" \\"}') == r'{"q": "$\\hat{\\rho}$ et \"ok\" \\"}'


async def test_json_text_with_unescaped_latex_is_repaired():
    class Q(BaseModel):
        question: str
        options: list[str]

    raw = r'{"question": "Forme de $\hat{\rho}_m$ et $\beta$ ?", "options": ["$\nu$", "$\theta$"]}'
    llm = GenericFakeChatModel(messages=iter([raw]))
    q = await ainvoke_structured(llm, Q, [HumanMessage(content="go")])
    assert q.question == r"Forme de $\hat{\rho}_m$ et $\beta$ ?" and q.options == [r"$\nu$", r"$\theta$"]


async def test_fallback_wrapped_model_uses_the_text_path():
    from langchain_core.runnables.fallbacks import RunnableWithFallbacks

    primary = GenericFakeChatModel(messages=iter(['{"name": "primary", "n": 1}']))
    fallback = GenericFakeChatModel(messages=iter(['{"name": "fallback", "n": 2}']))
    llm = primary.with_fallbacks([fallback])
    assert isinstance(llm, RunnableWithFallbacks)
    item = await ainvoke_structured(llm, Item, [HumanMessage(content="go")])
    assert item.name == "primary"


def test_with_max_tokens_copies_real_models_and_keeps_fallbacks():
    from langchain_openai import ChatOpenAI

    from ela.structured import unwrap_fallbacks, with_max_tokens

    primary = ChatOpenAI(model="m", api_key="k", max_tokens=100)
    fallback = ChatOpenAI(model="f", api_key="k", max_tokens=100)
    bigger = with_max_tokens(primary.with_fallbacks([fallback]), 4096)
    p, fbs = unwrap_fallbacks(bigger)
    assert p.max_tokens == 4096 and [f.max_tokens for f in fbs] == [4096]
    assert primary.max_tokens == 100  # originals untouched
    fake = GenericFakeChatModel(messages=iter([]))
    assert with_max_tokens(fake, 4096) is fake


async def test_provider_400_disables_the_structured_path_for_that_model(monkeypatch):
    import ela.structured as mod
    from fakes import ScriptedChatModel

    class Rejecting(ScriptedChatModel):
        def with_structured_output(self, schema, **kwargs):
            raise type("BadRequestError", (Exception,), {"status_code": 400})("response_format unsupported")

    monkeypatch.setattr(mod, "_structured_rejected", set())
    llm = Rejecting(['{"name": "a", "n": 1}', '{"name": "b", "n": 2}'])
    assert (await ainvoke_structured(llm, Item, [HumanMessage(content="go")])).name == "a"
    assert mod._structured_rejected == {"Rejecting:"}
    assert (await ainvoke_structured(llm, Item, [HumanMessage(content="go")])).name == "b"
