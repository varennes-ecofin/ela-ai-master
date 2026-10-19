"""Versioned prompt templates.

Prompts live as Markdown files next to this module (``system_v1.md``,
``quiz_v1.md``...). The active version is ``PROMPT_VERSION`` (``v1`` by
default) and is recorded in traces and evaluation reports, so that two
prompt versions can be compared with ``ela-eval compare``.

Templates use ``string.Template`` placeholders (``$level_block``) rather
than ``str.format`` braces, which would clash with LaTeX.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from string import Template

from ela.config import get_settings

PROMPT_NAMES = (
    "system",
    "quiz",
    "exercise",
    "check_step",
    "transcribe",
    "summary",
    "review_sheet",
    "fix_code",
    "interpret_run",
)


def prompt_version() -> str:
    """Active prompt version (``PROMPT_VERSION``)."""
    return get_settings().prompt_version


@lru_cache(maxsize=32)
def load_template(name: str, version: str) -> Template:
    """Load ``<name>_<version>.md`` as a template.

    Args:
        name: Prompt name (``system``, ``quiz``...).
        version: Version suffix (``v1``).

    Returns:
        The template.

    Raises:
        FileNotFoundError: When no such prompt file is packaged.
    """
    path = resources.files(__name__).joinpath(f"{name}_{version}.md")
    if not path.is_file():
        raise FileNotFoundError(f"prompt '{name}' version '{version}' not found ({path})")
    return Template(path.read_text(encoding="utf-8"))


def render(name: str, *, version: str | None = None, **variables: str) -> str:
    """Render a prompt with its variables (unknown placeholders are left as is).

    Args:
        name: Prompt name.
        version: Version override (defaults to ``PROMPT_VERSION``).
        **variables: Placeholder values.

    Returns:
        The rendered prompt text.
    """
    return load_template(name, version or prompt_version()).safe_substitute(**variables).strip()
