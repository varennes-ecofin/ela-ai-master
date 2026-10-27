"""Small server-side SVG charts for the backoffice (no JavaScript, no dependency).

Four shapes are enough for the activity pages: a line (daily volume),
stacked areas (daily volume by kind), a donut (share by kind) and
horizontal bars (ranking). Every function returns ``Markup`` ready to be
inlined in a Jinja template; the colours come from CSS classes
(``.s-<key>``) so that the charts follow the light/dark theme, and every
mark carries a ``<title>`` as a native tooltip.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from markupsafe import Markup, escape

WIDTH = 720
HEIGHT = 240
MARGIN = {"left": 44, "right": 14, "top": 12, "bottom": 26}
MAX_X_LABELS = 10


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}".replace(".", ",")


def _nice_ticks(maximum: float, count: int = 4) -> list[float]:
    """Return round tick values from 0 to a 'nice' bound above ``maximum`` (1-2-5 steps)."""
    if maximum <= 0:
        return [0.0, 1.0]
    raw = maximum / count
    power = 10 ** math.floor(math.log10(raw))
    step = next(s * power for s in (1, 2, 5, 10) if s * power >= raw)
    top = math.ceil(maximum / step) * step
    ticks: list[float] = []
    value = 0.0
    while value <= top + 1e-9:
        ticks.append(round(value, 6))
        value += step
    return ticks


def _x_positions(n: int, left: float, right: float) -> list[float]:
    if n <= 1:
        return [(left + right) / 2]
    step = (right - left) / (n - 1)
    return [left + i * step for i in range(n)]


def _axes(ticks: list[float], labels: Sequence[str], xs: list[float], scale_y, width: int = WIDTH) -> list[str]:
    """Return the grid lines, y tick labels and thinned x labels."""
    parts = []
    for t in ticks:
        y = scale_y(t)
        parts.append(
            f'<line class="grid" x1="{MARGIN["left"]}" x2="{width - MARGIN["right"]}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        parts.append(f'<text class="tick" x="{MARGIN["left"] - 6}" y="{y + 4:.1f}" text-anchor="end">{_fmt(t)}</text>')
    every = max(1, math.ceil(len(labels) / max(3, round(MAX_X_LABELS * width / WIDTH))))
    shown = {i for i in range(len(labels)) if i % every == 0}
    last = len(labels) - 1
    if last not in shown and last - max(shown) >= every / 2:
        shown.add(last)
    for i in sorted(shown):
        parts.append(
            f'<text class="tick" x="{xs[i]:.1f}" y="{HEIGHT - 8}" text-anchor="middle">{escape(labels[i])}</text>'
        )
    return parts


def _svg(body: list[str], height: int = HEIGHT, cls: str = "", width: int = WIDTH) -> Markup:
    return Markup(
        f'<svg class="chart {cls}" viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'preserveAspectRatio="xMidYMid meet">{"".join(body)}</svg>'
    )


def line_chart(
    labels: Sequence[str], values: Sequence[float], *, unit: str = "", key: str = "0", width: int = WIDTH
) -> Markup:
    """Draw a line with one point per label (``unit`` names the values in the tooltips; ``width`` is the viewBox width)."""
    if not labels:
        return Markup("")
    top, bottom = MARGIN["top"], HEIGHT - MARGIN["bottom"]
    ticks = _nice_ticks(max(values, default=0))
    span = ticks[-1] or 1

    def sy(v: float) -> float:
        return bottom - (bottom - top) * v / span

    xs = _x_positions(len(labels), MARGIN["left"], width - MARGIN["right"])
    body = _axes(ticks, labels, xs, sy, width)
    points = " ".join(f"{x:.1f},{sy(v):.1f}" for x, v in zip(xs, values, strict=True))
    body.append(f'<polyline class="line s-{key}" fill="none" points="{points}"/>')
    for label, x, v in zip(labels, xs, values, strict=True):
        body.append(
            f'<circle class="dot s-{key}" cx="{x:.1f}" cy="{sy(v):.1f}" r="3.2">'
            f"<title>{escape(label)} : {_fmt(v)} {escape(unit)}</title></circle>"
        )
    return _svg(body, cls="chart-line", width=width)


def stacked_area_chart(
    labels: Sequence[str], series: Sequence[tuple[str, str, Sequence[float]]], *, unit: str = "", width: int = WIDTH
) -> Markup:
    """Draw stacked areas, one per ``(key, name, values)`` series, bottom series first."""
    if not labels or not series:
        return Markup("")
    top, bottom = MARGIN["top"], HEIGHT - MARGIN["bottom"]
    n = len(labels)
    totals = [sum(values[i] for _, _, values in series) for i in range(n)]
    ticks = _nice_ticks(max(totals, default=0))
    span = ticks[-1] or 1

    def sy(v: float) -> float:
        return bottom - (bottom - top) * v / span

    xs = _x_positions(n, MARGIN["left"], width - MARGIN["right"])
    body = _axes(ticks, labels, xs, sy, width)
    base = [0.0] * n
    for key, name, values in series:
        upper = [b + v for b, v in zip(base, values, strict=True)]
        up = " ".join(f"{x:.1f},{sy(v):.1f}" for x, v in zip(xs, upper, strict=True))
        down = " ".join(f"{x:.1f},{sy(v):.1f}" for x, v in zip(reversed(xs), reversed(base), strict=True))
        tooltip = ", ".join(f"{label} : {_fmt(v)}" for label, v in zip(labels, values, strict=True) if v)
        body.append(
            f'<polygon class="area s-{escape(key)}" points="{up} {down}">'
            f"<title>{escape(name)}{' — ' + escape(tooltip) + ' ' + escape(unit) if tooltip else ''}</title></polygon>"
        )
        base = upper
    return _svg(body, cls="chart-area", width=width)


def donut_chart(items: Sequence[tuple[str, str, float]], *, unit: str = "") -> Markup:
    """Draw a donut of ``(key, name, value)`` items with the total in the middle."""
    total = sum(v for _, _, v in items)
    if total <= 0:
        return Markup("")
    size, cx, cy, r, width = 200, 100, 100, 80, 26
    body = []
    angle = -math.pi / 2
    for key, name, value in items:
        if value <= 0:
            continue
        share = value / total
        if share >= 0.9999:  # a full ring: two outer arcs and two inner arcs, even-odd filled
            ri = r - width
            ring = (
                f"M {cx + r} {cy} A {r} {r} 0 1 1 {cx - r} {cy} A {r} {r} 0 1 1 {cx + r} {cy} Z "
                f"M {cx + ri} {cy} A {ri} {ri} 0 1 0 {cx - ri} {cy} A {ri} {ri} 0 1 0 {cx + ri} {cy} Z"
            )
            body.append(
                f'<path class="slice s-{escape(key)}" fill-rule="evenodd" d="{ring}">'
                f"<title>{escape(name)} : {_fmt(value)} {escape(unit)} (100 %)</title></path>"
            )
            continue
        end = angle + 2 * math.pi * share
        x1, y1 = cx + r * math.cos(angle), cy + r * math.sin(angle)
        x2, y2 = cx + r * math.cos(end), cy + r * math.sin(end)
        ri = r - width
        x3, y3 = cx + ri * math.cos(end), cy + ri * math.sin(end)
        x4, y4 = cx + ri * math.cos(angle), cy + ri * math.sin(angle)
        large = 1 if share > 0.5 else 0
        path = (
            f"M {x1:.2f} {y1:.2f} A {r} {r} 0 {large} 1 {x2:.2f} {y2:.2f} L {x3:.2f} {y3:.2f} "
            f"A {ri} {ri} 0 {large} 0 {x4:.2f} {y4:.2f} Z"
        )
        body.append(
            f'<path class="slice s-{escape(key)}" d="{path}"><title>{escape(name)} : {_fmt(value)} {escape(unit)} '
            f"({share * 100:.0f} %)</title></path>"
        )
        angle = end
    body.append(f'<text class="donut-total" x="{cx}" y="{cy + 2}" text-anchor="middle">{_fmt(total)}</text>')
    body.append(f'<text class="tick" x="{cx}" y="{cy + 20}" text-anchor="middle">{escape(unit)}</text>')
    return Markup(
        f'<svg class="chart chart-donut" viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img">'
        f"{''.join(body)}</svg>"
    )


def hbar_chart(rows: Sequence[tuple[str, float]], *, unit: str = "", key: str = "0", max_rows: int = 25) -> Markup:
    """Draw horizontal bars for ``(label, value)`` rows, largest first as given; the height follows the row count."""
    rows = list(rows)[:max_rows]
    if not rows:
        return Markup("")
    row_h, label_w, pad = 24, 190, 8
    height = len(rows) * row_h + 2 * pad
    maximum = max(v for _, v in rows) or 1
    bar_x = label_w + pad
    bar_max = WIDTH - bar_x - 56
    body = []
    for i, (label, value) in enumerate(rows):
        y = pad + i * row_h
        w = max(1.0, bar_max * value / maximum)
        text = str(label)
        shown = text if len(text) <= 26 else text[:25] + "…"
        body.append(
            f'<text class="bar-label" x="{label_w}" y="{y + 16}" text-anchor="end"><title>{escape(text)}</title>{escape(shown)}</text>'
        )
        body.append(
            f'<rect class="bar s-{key}" x="{bar_x}" y="{y + 4}" width="{w:.1f}" height="{row_h - 8}" rx="3">'
            f"<title>{escape(text)} : {_fmt(value)} {escape(unit)}</title></rect>"
        )
        body.append(f'<text class="bar-value" x="{bar_x + w + 6:.1f}" y="{y + 16}">{_fmt(value)}</text>')
    return _svg(body, height=height, cls="chart-bars")


def day_labels(days: Sequence[Any]) -> list[str]:
    """Return ``dd/mm`` labels for a list of dates."""
    return [f"{d:%d/%m}" for d in days]
