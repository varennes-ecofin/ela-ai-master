r"""Normalise LaTeX delimiters in model output for the Chainlit renderer.

Chainlit renders mathematics with KaTeX through ``remark-math``, which only
recognises ``$...$`` (inline) and ``$$...$$`` (display). Models frequently
emit ``\\(...\\)``, ``\\[...\\]`` or bare ``equation``/``align`` environments,
which are then shown as raw LaTeX. :func:`normalize_math` rewrites those
forms, leaving fenced code blocks and inline code untouched.
"""

from __future__ import annotations

import re

_CODE_SPLIT = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_DISPLAY_ENVS = r"(?:equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?|alignat\*?)"
_BARE_ENV = re.compile(r"(?<!\$)\\begin\{(" + _DISPLAY_ENVS + r")\}(.*?)\\end\{\1\}(?!\$)", re.DOTALL)


def _wrap_env(m: re.Match) -> str:
    name = m.group(1).rstrip("*")
    inner = m.group(2).strip("\n")
    if name == "equation":
        return f"\n$$\n{inner.strip()}\n$$\n"
    kind = "aligned" if name in {"align", "alignat", "eqnarray"} else "gathered"
    return f"\n$$\n\\begin{{{kind}}}\n{inner}\n\\end{{{kind}}}\n$$\n"


def _normalize_segment(text: str) -> str:
    # \[ ... \]  →  $$ ... $$   (on its own lines)
    text = re.sub(r"\\\[\s*(.*?)\s*\\\]", lambda m: f"\n$$\n{m.group(1)}\n$$\n", text, flags=re.DOTALL)
    # \( ... \)  →  $ ... $
    text = re.sub(r"\\\(\s*(.*?)\s*\\\)", lambda m: f"${m.group(1)}$", text, flags=re.DOTALL)
    # bare display environments outside $$
    text = _BARE_ENV.sub(_wrap_env, text)
    # collapse the blank lines we may have introduced around display math
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def normalize_math(text: str) -> str:
    r"""Rewrite LaTeX delimiters into the ``$`` forms understood by KaTeX.

    Args:
        text: Markdown produced by the model.

    Returns:
        The same Markdown with ``\\(..\\)`` → ``$..$``, ``\\[..\\]`` → ``$$..$$``
        and bare ``equation``/``align`` environments wrapped in ``$$``. Code
        blocks and inline code are left untouched.
    """
    if not text or ("\\(" not in text and "\\[" not in text and "\\begin{" not in text):
        return text
    parts = _CODE_SPLIT.split(text)
    return "".join(p if p.startswith("`") else _normalize_segment(p) for p in parts)
