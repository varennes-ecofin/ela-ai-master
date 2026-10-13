"""Offline tests for ``ela.config`` and ``ela.llm`` (no network, no API key needed).

Run with::

    python -m pytest tests/ -q
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from ela.config import Settings
from ela.llm import build_chat_model, encode_image_block, get_chat_model

# 1x1 transparent PNG
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _settings(**overrides) -> Settings:
    """Build a Settings object that ignores any local .env file."""
    base = {
        "cerebras_api_key": "test-key",
        "llm_provider": "cerebras",
        "llm_model_chat": "gpt-oss-120b",
        "llm_model_vision": "qwen-3.8-27b",
        "llm_model_fallback": "qwen-3.8-27b",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_cerebras_credentials():
    s = _settings()
    assert s.provider_credentials("cerebras") == ("https://api.cerebras.ai/v1", "test-key")


def test_ilaas_requires_chat_base_url():
    s = _settings(ilaas_api_key="k")
    with pytest.raises(ValueError, match="ILAAS_CHAT_BASE_URL"):
        s.provider_credentials("ilaas")
    s2 = _settings(ilaas_api_key="k", ilaas_chat_base_url="https://api.ilaas.fr/v1")
    assert s2.provider_credentials("ilaas") == ("https://api.ilaas.fr/v1", "k")


def test_build_chat_model_uses_provider_base_url():
    m = build_chat_model("gpt-oss-120b", settings=_settings())
    assert m.openai_api_base == "https://api.cerebras.ai/v1"
    assert m.model_name == "gpt-oss-120b"
    assert m.reasoning_effort == "low"
    assert m.streaming is True


def test_missing_api_key_is_explicit():
    with pytest.raises(ValueError, match="CEREBRAS_API_KEY"):
        build_chat_model("gpt-oss-120b", settings=_settings(cerebras_api_key=None))


def test_groq_does_not_send_reasoning_effort():
    m = build_chat_model(
        "llama-3.3-70b-versatile", "groq", settings=_settings(groq_api_key="g")
    )
    assert m.reasoning_effort is None
    assert "groq" in m.openai_api_base


def test_chat_role_has_fallback_when_configured():
    chat = get_chat_model("chat", settings=_settings())
    assert type(chat).__name__ == "RunnableWithFallbacks"
    assert chat.runnable.model_name == "gpt-oss-120b"
    assert chat.fallbacks[0].model_name == "qwen-3.8-27b"


def test_chat_role_without_fallback():
    chat = get_chat_model("chat", settings=_settings(llm_model_fallback="none"))
    assert type(chat).__name__ == "ChatOpenAI"
    same = get_chat_model("chat", settings=_settings(llm_model_fallback="gpt-oss-120b"))
    assert type(same).__name__ == "ChatOpenAI"


def test_vision_role_uses_vision_model_without_fallback():
    vision = get_chat_model("vision", settings=_settings())
    assert type(vision).__name__ == "ChatOpenAI"
    assert vision.model_name == "qwen-3.8-27b"


def test_encode_image_block(tmp_path: Path):
    img = tmp_path / "plot.png"
    img.write_bytes(_PNG)
    block = encode_image_block(img, None)  # MIME guessed from extension
    assert block["type"] == "image_url"
    assert block["image_url"]["url"].startswith("data:image/png;base64,")
    block2 = encode_image_block(img, "image/webp")  # reported MIME wins
    assert block2["image_url"]["url"].startswith("data:image/webp;base64,")


def test_encode_image_block_rejects_non_images(tmp_path: Path):
    doc = tmp_path / "notes.pdf"
    doc.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="Unsupported image type"):
        encode_image_block(doc, "application/pdf")


def test_vision_payload_is_openai_compatible(tmp_path: Path):
    img = tmp_path / "plot.png"
    img.write_bytes(_PNG)
    vision = get_chat_model("vision", settings=_settings())
    messages = [
        SystemMessage(content="sys"),
        HumanMessage(content=[{"type": "text", "text": "Explique ce graphique"}, encode_image_block(img)]),
    ]
    payload = vision._get_request_payload(messages)
    assert payload["model"] == "qwen-3.8-27b"
    assert payload["reasoning_effort"] == "low"
    assert payload["messages"][1]["content"][1]["type"] == "image_url"


def test_blank_env_values_fall_back_to_defaults(tmp_path: Path):
    """``VAR=`` lines (even followed by an inline comment) must not break startup."""
    env = tmp_path / ".env"
    env.write_text(
        "LLM_PROVIDER=cerebras   # commentaire inline\n"
        "LLM_VISION_PROVIDER=                 # vide = même fournisseur\n"
        "LLM_FALLBACK_PROVIDER=\n"
        "LLM_MODEL_FALLBACK=none\n"
        "LLM_REASONING_EFFORT=omit\n"
        "LLM_MAX_TOKENS=\n"
        "ILAAS_CHAT_BASE_URL=\n"
        "CEREBRAS_API_KEY=abc\n",
        encoding="utf-8",
    )
    s = Settings(_env_file=env)
    assert s.llm_provider == "cerebras"
    assert s.llm_vision_provider is None and s.llm_fallback_provider is None
    assert s.fallback_model is None
    assert s.reasoning_effort is None
    assert s.llm_max_tokens == 2048
    assert s.ilaas_chat_base_url is None
    assert s.cerebras_api_key.get_secret_value() == "abc"
    m = build_chat_model("gpt-oss-120b", settings=s)
    assert m.reasoning_effort is None
    assert type(get_chat_model("chat", settings=s)).__name__ == "ChatOpenAI"
