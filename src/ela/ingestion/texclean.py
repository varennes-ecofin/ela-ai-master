r"""LaTeX → readable text conversion for retrieval chunks.

The goal is not to typeset but to produce text that embeds and indexes
well while keeping the mathematics intact:

* comments are removed (``\\%`` is preserved);
* zero-argument ``\\newcommand`` / ``\\def`` macros of the course are expanded,
  so ``\\bX`` becomes ``\\boldsymbol{X}`` in the chunk text;
* math (``$...$``, ``\\(...\\)``, ``\\[...\\]``, ``equation``, ``align``...) is
  preserved verbatim, wrapped in ``$...$`` / ``$$...$$`` (what KaTeX and the
  LLM both understand);
* theorem-like environments become labelled paragraphs
  (``**Définition (titre)** : ...``) and are also returned as typed elements;
* lists become Markdown bullets, tables become pipe-separated rows,
  figures become ``[Figure : légende]``;
* beamer layout (columns, overlays, blocks, spacing, font sizes) is unwrapped.

Everything here is pure functions on strings; no LaTeX engine is needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Environment taxonomy
# ---------------------------------------------------------------------------

#: Theorem-like environments → (canonical type, French label).
THEOREM_ENVS: dict[str, tuple[str, str]] = {
    "definition": ("definition", "Définition"),
    "defn": ("definition", "Définition"),
    "theorem": ("theorem", "Théorème"),
    "thm": ("theorem", "Théorème"),
    "proposition": ("proposition", "Proposition"),
    "prop": ("proposition", "Proposition"),
    "lemma": ("lemma", "Lemme"),
    "lem": ("lemma", "Lemme"),
    "corollary": ("corollary", "Corollaire"),
    "cor": ("corollary", "Corollaire"),
    "example": ("example", "Exemple"),
    "ex": ("example", "Exemple"),
    "remark": ("remark", "Remarque"),
    "model": ("model", "Modèle"),
    "assumption": ("assumption", "Hypothèse"),
    "assum": ("assumption", "Hypothèse"),
    "proof": ("proof", "Preuve"),
    "exampleblock": ("example", "Exemple"),
}

#: Display-math environments (kept verbatim inside ``$$ ... $$``).
MATH_ENVS = {
    "equation",
    "equation*",
    "align",
    "align*",
    "alignat",
    "alignat*",
    "eqnarray",
    "eqnarray*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "displaymath",
    "flalign",
    "flalign*",
}

#: Environments whose content is kept but whose wrapper carries no meaning.
UNWRAP_ENVS = {
    "columns",
    "column",
    "center",
    "flushleft",
    "flushright",
    "minipage",
    "multicols",
    "overprint",
    "onlyenv",
    "beamercolorbox",
    "small",
    "footnotesize",
    "scriptsize",
    "tiny",
    "large",
    "Large",
    "normalsize",
    "quote",
    "quotation",
    "abstract",
    "frame",
    "document",
    "adjustbox",
    "wrapfigure",
    "spacing",
    "singlespace",
    "onehalfspace",
}

#: Environments dropped entirely (no retrievable content).
DROP_ENVS = {
    "tikzpicture",
    "comment",
    "thebibliography",
    "pgfpicture",
    "axis",
    "picture",
}

_CODE_ENVS = {"lstlisting", "verbatim", "minted", "Verbatim"}

# Layout / spacing commands removed outright (with their optional argument).
_LAYOUT_COMMANDS = [
    "centering",
    "raggedright",
    "raggedleft",
    "medskip",
    "bigskip",
    "smallskip",
    "vfill",
    "hfill",
    "pause",
    "newpage",
    "clearpage",
    "framebreak",
    "noindent",
    "indent",
    "linebreak",
    "newline",
    "tableofcontents",
    "listoffigures",
    "listoftables",
    "frontmatter",
    "mainmatter",
    "backmatter",
    "maketitle",
    "justifying",
    "small",
    "footnotesize",
    "scriptsize",
    "tiny",
    "normalsize",
    "large",
    "Large",
    "LARGE",
    "huge",
    "Huge",
    "bfseries",
    "itshape",
    "ttfamily",
    "displaystyle",
    "allowframebreaks",
    "hline",
    "toprule",
    "midrule",
    "bottomrule",
    "appendix",
    "newblock",
    "protect",
    "relax",
    "selectfont",
    "onslide",
    "only",
    "uncover",
    "visible",
    "invisible",
    "alt",
]

# Commands removed together with their braced argument(s).
_DROP_WITH_ARG = [
    "label",
    "vspace",
    "vspace*",
    "hspace",
    "hspace*",
    "setlength",
    "setbeamercovered",
    "setbeamertemplate",
    "setbeamercolor",
    "setbeamerfont",
    "setcounter",
    "addtocounter",
    "pagestyle",
    "thispagestyle",
    "usebackgroundtemplate",
    "includegraphics",  # handled before (figures); leftover ones are dropped
    "rule",
    "phantom",
    "hphantom",
    "vphantom",
    "bibliographystyle",
    "bibliography",
    "index",
    "addcontentsline",
    "pgfdeclareimage",
    "pgfuseimage",
    "logo",
    "titlegraphic",
    "lstset",
    "captionsetup",
    "renewcommand",
    "newcommand",
    "def",
    "fontsize",
    "color",
    "columnwidth",
    "linewidth",
    "textwidth",
    "textheight",
    "resizebox",
    "scalebox",
]

_MATH_PLACEHOLDER = "\x00MATH{}\x00"
_PLACEHOLDER_RE = re.compile(r"\x00MATH(\d+)\x00")

_NONLETTER = r"(?![A-Za-z])"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def strip_comments(text: str) -> str:
    r"""Remove ``%`` comments (keeping ``\\%``) and ``comment`` environments.

    Args:
        text: Raw LaTeX.

    Returns:
        LaTeX without comments. A comment that occupies the whole line is
        removed with its line break so that paragraphs are not split.
    """
    text = re.sub(r"\\begin\{comment\}.*?\\end\{comment\}", "", text, flags=re.DOTALL)
    out: list[str] = []
    for line in text.split("\n"):
        i = 0
        cut = None
        while i < len(line):
            ch = line[i]
            if ch == "\\":
                i += 2
                continue
            if ch == "%":
                cut = i
                break
            i += 1
        if cut is None:
            out.append(line)
        elif cut == 0 or not line[:cut].strip():
            continue  # whole-line comment
        else:
            out.append(line[:cut].rstrip())
    return "\n".join(out)


def read_group(text: str, pos: int, open_ch: str = "{", close_ch: str = "}") -> tuple[str, int] | None:
    """Read a balanced group starting at ``pos`` (which must be ``open_ch``).

    Args:
        text: Source string.
        pos: Index of the opening delimiter.
        open_ch: Opening delimiter.
        close_ch: Closing delimiter.

    Returns:
        ``(content, end_index)`` where ``end_index`` is the index after the
        closing delimiter, or ``None`` when ``pos`` is not an opening
        delimiter or the group is unbalanced.
    """
    if pos >= len(text) or text[pos] != open_ch:
        return None
    depth = 0
    i = pos
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[pos + 1 : i], i + 1
        i += 1
    return None


def skip_spaces(text: str, pos: int) -> int:
    """Return the first index ≥ ``pos`` that is not a space/tab/newline."""
    n = len(text)
    while pos < n and text[pos] in " \t\r\n":
        pos += 1
    return pos


def read_optional(text: str, pos: int) -> tuple[str | None, int]:
    """Read an optional ``[...]`` argument at ``pos`` (spaces allowed before).

    Returns:
        ``(content or None, new_position)``.
    """
    p = skip_spaces(text, pos)
    if p < len(text) and text[p] == "[":
        grp = read_group(text, p, "[", "]")
        if grp is not None:
            return grp[0], grp[1]
    return None, pos


def find_env_end(text: str, name: str, start: int) -> tuple[int, int] | None:
    r"""Find the matching ``\\end{name}`` for an environment opened before ``start``.

    Nested environments of the same name are handled.

    Args:
        text: Source string.
        name: Environment name (may contain ``*``).
        start: Index just after the opening ``\\begin{name}``.

    Returns:
        ``(end_start, end_stop)``: indexes of ``\\end{name}`` and of the
        character after it, or ``None`` if unbalanced.
    """
    pat = re.compile(r"\\(begin|end)\{" + re.escape(name) + r"\}")
    depth = 1
    for m in pat.finditer(text, start):
        depth += 1 if m.group(1) == "begin" else -1
        if depth == 0:
            return m.start(), m.end()
    return None


# ---------------------------------------------------------------------------
# Macros
# ---------------------------------------------------------------------------

_NEWCOMMAND_RE = re.compile(r"\\(?:re)?newcommand\*?\s*\{?\\([A-Za-z]+)\}?\s*(\[(\d)\])?\s*(?=\{)")
_DEF_RE = re.compile(r"\\def\s*\\([A-Za-z]+)\s*(?=\{)")


def collect_macros(text: str) -> dict[str, str]:
    r"""Collect zero-argument ``\\newcommand`` and ``\\def`` macros.

    Args:
        text: LaTeX (preamble included, comments removed).

    Returns:
        Mapping ``name → replacement`` (no leading backslash in the name).
    """
    macros: dict[str, str] = {}
    for regex in (_NEWCOMMAND_RE, _DEF_RE):
        for m in regex.finditer(text):
            if regex is _NEWCOMMAND_RE and m.group(2):
                continue  # macro with parameters: not expanded
            grp = read_group(text, m.end())
            if grp is None:
                continue
            body = grp[0].strip()
            if "#" in body or not body:
                continue
            macros[m.group(1)] = body
    return macros


def expand_macros(text: str, macros: dict[str, str], passes: int = 3) -> str:
    """Expand zero-argument macros (a few passes for nested definitions).

    Args:
        text: LaTeX body.
        macros: Mapping from :func:`collect_macros`.
        passes: Maximum number of expansion rounds.

    Returns:
        Expanded text.
    """
    if not macros:
        return text
    names = sorted(macros, key=len, reverse=True)
    pattern = re.compile(r"\\(" + "|".join(re.escape(n) for n in names) + r")" + _NONLETTER)
    for _ in range(passes):
        new = pattern.sub(lambda m: macros[m.group(1)], text)
        if new == text:
            break
        text = new
    return text


# ---------------------------------------------------------------------------
# Math protection
# ---------------------------------------------------------------------------


def _clean_math(inner: str) -> str:
    inner = re.sub(r"\\label\{[^}]*\}", "", inner)
    inner = re.sub(r"\\(nonumber|notag)\b", "", inner)
    inner = re.sub(r"[ \t]+\n", "\n", inner)
    return inner.strip()


def protect_math(text: str) -> tuple[str, list[str]]:
    r"""Replace every math segment by a placeholder.

    Display math (environments, ``\\[...\\]``, ``$$...$$``) is normalised to
    ``$$ ... $$`` and inline math (``$...$``, ``\\(...\\)``) to ``$...$``.

    Args:
        text: LaTeX body (comments removed).

    Returns:
        ``(text_with_placeholders, segments)``.
    """
    segments: list[str] = []

    def stash(rendered: str) -> str:
        segments.append(rendered)
        return _MATH_PLACEHOLDER.format(len(segments) - 1)

    # 1. Display-math environments
    env_names = "|".join(re.escape(e) for e in sorted(MATH_ENVS, key=len, reverse=True))
    env_re = re.compile(r"\\begin\{(" + env_names + r")\}")
    out: list[str] = []
    pos = 0
    while True:
        m = env_re.search(text, pos)
        if not m:
            out.append(text[pos:])
            break
        name = m.group(1)
        end = find_env_end(text, name, m.end())
        if end is None:
            out.append(text[pos : m.end()])
            pos = m.end()
            continue
        inner = text[m.end() : end[0]]
        out.append(text[pos : m.start()])
        base = name.rstrip("*")
        if base in {"align", "alignat", "gather", "multline", "flalign", "eqnarray"}:
            kind = "aligned" if base in {"align", "alignat", "flalign", "eqnarray"} else "gathered"
            rendered = f"$$\\begin{{{kind}}}\n{_clean_math(inner)}\n\\end{{{kind}}}$$"
        else:
            rendered = f"$$ {_clean_math(inner)} $$"
        out.append("\n" + stash(rendered) + "\n")
        pos = end[1]
    text = "".join(out)

    # 2. \[ ... \] and $$ ... $$
    def _display(m: re.Match) -> str:
        return "\n" + stash(f"$$ {_clean_math(m.group(1))} $$") + "\n"

    text = re.sub(r"\\\[(.*?)\\\]", _display, text, flags=re.DOTALL)
    text = re.sub(r"\$\$(.*?)\$\$", _display, text, flags=re.DOTALL)

    # 3. \( ... \) and $ ... $
    text = re.sub(r"\\\((.*?)\\\)", lambda m: stash(f"${_clean_math(m.group(1))}$"), text, flags=re.DOTALL)
    text = re.sub(r"(?<!\\)\$([^$]+?)\$", lambda m: stash(f"${_clean_math(m.group(1))}$"), text, flags=re.DOTALL)
    return text, segments


def restore_math(text: str, segments: list[str]) -> str:
    """Put math segments back in place of their placeholders."""
    return _PLACEHOLDER_RE.sub(lambda m: segments[int(m.group(1))], text)


# ---------------------------------------------------------------------------
# Typed elements
# ---------------------------------------------------------------------------


@dataclass
class Element:
    """A theorem-like block extracted from a slide or a section.

    Attributes:
        kind: Canonical type (``definition``, ``theorem``...).
        label: Human label (``Définition``).
        title: Optional title given in ``[...]``.
        text: Readable text of the block (math preserved).
    """

    kind: str
    label: str
    title: str
    text: str


@dataclass
class Converted:
    """Result of :func:`convert`."""

    text: str
    elements: list[Element] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Environment conversion
# ---------------------------------------------------------------------------

_BEGIN_RE = re.compile(r"\\begin\{([A-Za-z*]+)\}")


def _tabular_to_rows(inner: str) -> str:
    """Convert a ``tabular`` body into pipe-separated rows."""
    inner = re.sub(r"\\(hline|toprule|midrule|bottomrule|cline\{[^}]*\})", "", inner)
    inner = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", "", inner)
    rows = []
    for raw in re.split(r"\\\\", inner):
        cells = [c.strip() for c in raw.split("&")]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _convert_items(inner: str, depth: int, ctx: Converted, ordered: bool) -> str:
    """Convert an ``itemize``/``enumerate`` body into Markdown bullets."""
    inner = _convert_envs(inner, ctx, depth + 1)
    parts = re.split(r"\\item" + _NONLETTER, inner)
    lines = []
    prefix = "  " * depth
    for idx, part in enumerate(parts):
        part = part.strip()
        if idx == 0:
            if part:
                lines.append(prefix + part)
            continue
        label = None
        if part.startswith("["):
            grp = read_group(part, 0, "[", "]")
            if grp is not None:
                label, end = grp[0].strip(), grp[1]
                part = part[end:].strip()
        if label:
            bullet = f"{prefix}- {label} "
        elif ordered:
            bullet = f"{prefix}{idx}. "
        else:
            bullet = f"{prefix}- "
        lines.append(bullet + part)
    return "\n" + "\n".join(lines) + "\n"


def _convert_envs(text: str, ctx: Converted, depth: int = 0) -> str:
    """Recursively convert environments (math already protected)."""
    out: list[str] = []
    pos = 0
    while True:
        m = _BEGIN_RE.search(text, pos)
        if not m:
            out.append(text[pos:])
            break
        name = m.group(1)
        end = find_env_end(text, name, m.end())
        if end is None:
            out.append(text[pos : m.end()])
            pos = m.end()
            continue
        out.append(text[pos : m.start()])
        inner = text[m.end() : end[0]]
        pos = end[1]

        # Optional / mandatory arguments right after \begin{...}
        opt, p = read_optional(inner, 0)
        inner = inner[p:]
        if name in {"column", "minipage", "beamercolorbox", "adjustbox", "wrapfigure"}:
            grp = read_group(inner, skip_spaces(inner, 0))
            if grp is not None:
                inner = inner[grp[1] :]
            if name == "wrapfigure":  # second argument
                grp = read_group(inner, skip_spaces(inner, 0))
                if grp is not None:
                    inner = inner[grp[1] :]
        if name == "columns":
            opt, _ = read_optional(inner, 0)

        if name in DROP_ENVS:
            continue
        if name in _CODE_ENVS:
            out.append("\n```\n" + inner.strip("\n") + "\n```\n")
            continue
        if name in {"itemize", "enumerate", "description"}:
            out.append(_convert_items(inner, depth, ctx, ordered=(name == "enumerate")))
            continue
        if name in {"tabular", "tabular*", "tabularx", "array"}:
            grp = read_group(inner, skip_spaces(inner, 0))
            if grp is not None:
                inner = inner[grp[1] :]
            out.append("\n" + _tabular_to_rows(inner) + "\n")
            continue
        if name in {"figure", "figure*", "table", "table*"}:
            caption = ""
            cm = re.search(r"\\caption(?:\[[^\]]*\])?", inner)
            if cm:
                grp = read_group(inner, skip_spaces(inner, cm.end()))
                if grp is not None:
                    caption = _convert_envs(grp[0], ctx, depth).strip()
                    inner = inner[: cm.start()] + inner[grp[1] :]
            inner = re.sub(r"\\includegraphics\s*(\[[^\]]*\])?\s*\{[^}]*\}", "", inner)
            body = _convert_envs(inner, ctx, depth).strip()
            kind = "Figure" if name.startswith("figure") else "Tableau"
            head = f"[{kind} : {caption}]" if caption else f"[{kind}]"
            out.append("\n" + head + ("\n" + body if body else "") + "\n")
            continue
        if name in {"block", "alertblock", "exampleblock"} or name in THEOREM_ENVS:
            if name in THEOREM_ENVS:
                kind, label = THEOREM_ENVS[name]
                title = (opt or "").strip()
            else:
                grp = read_group(inner, skip_spaces(inner, 0))
                title = grp[0].strip() if grp is not None else ""
                inner = inner[grp[1] :] if grp is not None else inner
                kind, label = ("example", "Exemple") if name == "exampleblock" else ("block", "")
            title = _finalize(_convert_commands(title)).strip()
            body = _convert_envs(inner, ctx, depth).strip()
            if label:
                head = f"**{label}" + (f" ({title})" if title else "") + "**"
            else:
                head = f"**{title}**" if title else ""
            if name in THEOREM_ENVS:
                ctx.elements.append(Element(kind=kind, label=label, title=title, text=body))
            out.append("\n" + (head + " : " if head else "") + body + "\n")
            continue
        # Default: unwrap (also for unknown environments)
        out.append(_convert_envs(inner, ctx, depth))
    return "".join(out)


# ---------------------------------------------------------------------------
# Command conversion
# ---------------------------------------------------------------------------

_INLINE_STYLES = {
    "textbf": "**{}**",
    "bf": "**{}**",
    "alert": "**{}**",
    "bt": "**{}**",
    "structure": "**{}**",
    "emph": "*{}*",
    "textit": "*{}*",
    "underline": "*{}*",
    "textsc": "{}",
    "texttt": "`{}`",
    "textrm": "{}",
    "textsf": "{}",
    "textnormal": "{}",
    "mbox": "{}",
    "text": "{}",
    "hbox": "{}",
    "fbox": "{}",
    "textsuperscript": "^{}",
    "textsubscript": "_{}",
    "footnote": " ({})",
    "caption": "[Légende : {}]",
    "frametitle": "{}",
    "framesubtitle": "{}",
    "title": "{}",
    "subtitle": "{}",
    "author": "{}",
    "institute": "{}",
    "date": "",
    "translate": "{}",
    "uppercase": "{}",
    "MakeUppercase": "{}",
}

_TWO_ARG_KEEP_SECOND = {"textcolor", "colorbox", "href", "hrefcol", "parbox", "makebox", "raisebox"}


def _replace_command_with_arg(text: str, name: str, template: str) -> str:
    r"""Replace ``\\name{arg}`` (balanced) by ``template.format(arg)``."""
    pat = re.compile(r"\\" + re.escape(name) + r"\s*(\[[^\]]*\])?\s*(?=\{)")
    out: list[str] = []
    pos = 0
    while True:
        m = pat.search(text, pos)
        if not m:
            out.append(text[pos:])
            break
        grp = read_group(text, m.end())
        if grp is None:
            out.append(text[pos : m.end()])
            pos = m.end()
            continue
        out.append(text[pos : m.start()])
        out.append(template.format(grp[0]))
        pos = grp[1]
    return "".join(out)


def _drop_two_arg_keep_second(text: str, name: str) -> str:
    pat = re.compile(r"\\" + re.escape(name) + r"\s*(\[[^\]]*\])?\s*(?=\{)")
    out: list[str] = []
    pos = 0
    while True:
        m = pat.search(text, pos)
        if not m:
            out.append(text[pos:])
            break
        first = read_group(text, m.end())
        if first is None:
            out.append(text[pos : m.end()])
            pos = m.end()
            continue
        second = read_group(text, skip_spaces(text, first[1]))
        out.append(text[pos : m.start()])
        if second is None:
            out.append(first[0])
            pos = first[1]
        else:
            out.append(second[0])
            pos = second[1]
    return "".join(out)


def _drop_with_args(text: str, name: str) -> str:
    r"""Remove ``\\name[opt]{arg}{arg}...`` entirely."""
    pat = re.compile(r"\\" + re.escape(name) + _NONLETTER)
    out: list[str] = []
    pos = 0
    while True:
        m = pat.search(text, pos)
        if not m:
            out.append(text[pos:])
            break
        out.append(text[pos : m.start()])
        p = m.end()
        while True:
            q = skip_spaces(text, p)
            if q < len(text) and text[q] == "[":
                grp = read_group(text, q, "[", "]")
                if grp is None:
                    break
                p = grp[1]
            elif q < len(text) and text[q] == "{":
                grp = read_group(text, q)
                if grp is None:
                    break
                p = grp[1]
            else:
                break
        pos = p
    return "".join(out)


def _convert_commands(text: str) -> str:
    """Convert or remove LaTeX commands outside math."""
    # Citations and references
    text = re.sub(r"\\cite[tp]?\*?\s*(\[[^\]]*\])?\s*\{([^}]*)\}", lambda m: "[" + m.group(2) + "]", text)
    text = re.sub(r"\\(?:eqref|ref|pageref|autoref|cref|Cref)\s*\{[^}]*\}", "", text)
    text = re.sub(r"\\url\s*\{([^}]*)\}", r"\1", text)
    for name in _TWO_ARG_KEEP_SECOND:
        text = _drop_two_arg_keep_second(text, name)
    for name, template in _INLINE_STYLES.items():
        text = _replace_command_with_arg(text, name, template)
    # Overlay specifications: \onslide<2->, \only<1>{...}, \item<2->
    text = re.sub(r"<[\d,+\-| ]*>", "", text)
    for name in _DROP_WITH_ARG:
        text = _drop_with_args(text, name)
    for name in _LAYOUT_COMMANDS:
        text = re.sub(r"\\" + re.escape(name) + r"\*?" + _NONLETTER + r"(\[[^\]]*\])?", "", text)
    # Special characters
    text = re.sub(r"\\(%|&|_|#|\$|\{|\})", r"\1", text)
    text = text.replace("~", " ").replace("\\,", " ").replace("\\;", " ").replace("\\ ", " ")
    text = re.sub(r"\\\\(\[[^\]]*\])?", "\n", text)
    text = re.sub(r"\\(?:ldots|dots)" + _NONLETTER, "…", text)
    text = re.sub(r"\\(?:textquoteleft|textquoteright|lq|rq)" + _NONLETTER, "'", text)
    text = re.sub(r"(?<!`)``(?!`)", "« ", text).replace("''", " »")
    # Unknown commands: keep the argument (\textsomething{x} → x), drop bare ones
    for _ in range(2):
        text = re.sub(r"\\[A-Za-z]+\*?\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z]+\*?" + _NONLETTER, "", text)
    # Stray grouping braces ({\large ...}, {\bf ...}) carry no meaning outside math
    text = text.replace("{", "").replace("}", "")
    return text


def _finalize(text: str) -> str:
    """Normalise whitespace of a converted block."""
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\{\s*\}", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_notes(text: str) -> tuple[str, list[str]]:
    r"""Pull ``\\note{...}`` speaker notes out of a frame body.

    Returns:
        ``(body_without_notes, notes)``.
    """
    notes: list[str] = []
    pat = re.compile(r"\\note\s*(\[[^\]]*\])?\s*(?=\{)")
    out: list[str] = []
    pos = 0
    while True:
        m = pat.search(text, pos)
        if not m:
            out.append(text[pos:])
            break
        grp = read_group(text, m.end())
        if grp is None:
            out.append(text[pos : m.end()])
            pos = m.end()
            continue
        out.append(text[pos : m.start()])
        notes.append(grp[0])
        pos = grp[1]
    return "".join(out), notes


def convert(body: str, macros: dict[str, str] | None = None, *, with_notes: bool = True) -> Converted:
    r"""Convert a LaTeX fragment (frame body, section text) into readable text.

    Args:
        body: LaTeX fragment (comments may still be present).
        macros: Zero-argument macros to expand (see :func:`collect_macros`).
        with_notes: Extract ``\\note{...}`` into :attr:`Converted.notes`.

    Returns:
        A :class:`Converted` with the text, the typed elements and the notes.
    """
    ctx = Converted(text="")
    body = strip_comments(body)
    if macros:
        body = expand_macros(body, macros)
    if with_notes:
        body, raw_notes = extract_notes(body)
        for raw in raw_notes:
            protected, segs = protect_math(raw)
            converted = _convert_envs(protected, Converted(text=""))
            ctx.notes.append(_finalize(restore_math(_convert_commands(converted), segs)))
    protected, segments = protect_math(body)
    converted = _convert_envs(protected, ctx)
    converted = _convert_commands(converted)
    ctx.text = _finalize(restore_math(converted, segments))
    for el in ctx.elements:  # element bodies were converted before command pass
        el.text = _finalize(restore_math(_convert_commands(el.text), segments))
        el.title = _finalize(restore_math(el.title, segments))
    return ctx


def plain_title(text: str, macros: dict[str, str] | None = None) -> str:
    """Convert a title fragment (frame/section title) to a one-line string."""
    conv = convert(text, macros, with_notes=False)
    return re.sub(r"\s+", " ", conv.text).strip()
