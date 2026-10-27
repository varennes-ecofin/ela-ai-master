"""Period filter shared by the activity, pedagogy, feedback and conversation pages.

The period is a closed range of calendar days carried in the query string
(``?from=2026-09-27&to=2026-10-27``), or a preset (``?days=30``). The default
is the last 30 days, i.e. from ``today - 30`` to ``today`` inclusive.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
from urllib.parse import urlencode

DEFAULT_DAYS = 30
PRESETS = (7, 30, 90)
MAX_SPAN_DAYS = 3 * 366


@dataclass(frozen=True)
class Period:
    """A closed range of days, with the preset it comes from (if any) and a validation message."""

    start: date
    end: date
    days: int | None = None
    error: str | None = None

    @property
    def span(self) -> int:
        """Number of days in the range."""
        return (self.end - self.start).days + 1

    @property
    def query(self) -> str:
        """Query-string fragment reproducing the period (``from=…&to=…``)."""
        return urlencode({"from": self.start.isoformat(), "to": self.end.isoformat()})

    @property
    def label(self) -> str:
        """Short French description used in page captions and CSV file names."""
        if self.days:
            return f"{self.days} derniers jours"
        return f"du {self.start:%d/%m/%Y} au {self.end:%d/%m/%Y}"


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def default_period(today: date | None = None, days: int = DEFAULT_DAYS) -> Period:
    """Return the preset period ending today."""
    today = today or date.today()
    return Period(start=today - timedelta(days=days), end=today, days=days)


def parse_period(
    from_: str | None = None, to: str | None = None, days: str | int | None = None, today: date | None = None
) -> Period:
    """Build the period from the query parameters.

    Args:
        from_: ``from`` parameter (ISO date).
        to: ``to`` parameter (ISO date).
        days: ``days`` preset; wins over the dates when present and valid.
        today: Reference day (tests).

    Returns:
        A valid period; when the parameters are unusable (bad date, start after end, range too long) the default
        period is returned with ``error`` set, so the page can say why it ignored the request.
    """
    today = today or date.today()
    if days not in (None, ""):
        try:
            n = int(days)
        except (TypeError, ValueError):
            n = 0
        if n in PRESETS:
            return default_period(today, n)
        return replace(default_period(today), error="Raccourci inconnu : période par défaut affichée.")
    if not from_ and not to:
        return default_period(today)
    start, end = _parse_date(from_), _parse_date(to)
    if start is None or end is None:
        return replace(default_period(today), error="Dates invalides : période par défaut affichée.")
    if start > end:
        return replace(
            default_period(today), error="La date de début doit précéder la date de fin : période par défaut affichée."
        )
    if (end - start).days > MAX_SPAN_DAYS:
        return replace(
            default_period(today), error="Période trop longue (3 ans au plus) : période par défaut affichée."
        )
    days_preset = next((n for n in PRESETS if start == today - timedelta(days=n) and end == today), None)
    return Period(start=start, end=end, days=days_preset)
