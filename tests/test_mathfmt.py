"""Tests for the LaTeX delimiter normalisation."""

from __future__ import annotations

from ela.mathfmt import normalize_math


def test_inline_and_display_delimiters():
    assert normalize_math(r"a \(x^2\) b") == "a $x^2$ b"
    assert normalize_math(r"a \[ y = 1 \] b") == "a \n$$\ny = 1\n$$\n b"


def test_bare_align_environment_is_wrapped():
    out = normalize_math("x\n\\begin{align}a &= b\\\\ c &= d\\end{align}\ny")
    assert "$$\n\\begin{aligned}" in out and "\\end{aligned}\n$$" in out


def test_multiline_display_block_gets_its_own_lines():
    raw = "on a\n\n$$\\begin{aligned}\nE(Y) & = a\\\\\nE(Z) & = b\n\\end{aligned}$$\n\nce qui implique $C$"
    out = normalize_math(raw)
    assert "$$\n\\begin{aligned}" in out and "\\end{aligned}\n$$" in out and "$$\\begin" not in out
    assert out.endswith("ce qui implique $C$")


def test_single_line_display_and_code_untouched():
    assert normalize_math("a $$x$$ b") == "a $$x$$ b"
    assert normalize_math("`\\(x\\)` et ```\n\\[y\\]\n```") == "`\\(x\\)` et ```\n\\[y\\]\n```"
