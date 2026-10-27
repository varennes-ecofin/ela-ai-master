"""CSV download of a table of the backoffice (French Excel flavour: ``;`` separator, UTF-8 with BOM)."""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any

from fastapi.responses import Response


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "oui" if value else "non"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        return str(value).replace(".", ",")
    return value


def csv_text(columns: Sequence[tuple[str, str]], rows: Iterable[dict[str, Any]]) -> str:
    """Render rows as CSV.

    Args:
        columns: ``(key, header)`` pairs; the key is looked up in every row.
        rows: Row mappings (missing keys give empty cells).

    Returns:
        The CSV text (no BOM), ``;`` separated, CRLF line ends.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=";", quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
    writer.writerow([header for _, header in columns])
    for row in rows:
        writer.writerow([_cell(row.get(key)) for key, _ in columns])
    return buffer.getvalue()


def csv_response(filename: str, columns: Sequence[tuple[str, str]], rows: Iterable[dict[str, Any]]) -> Response:
    """Build the download response of :func:`csv_text` (``ela-<filename>.csv``, never cached)."""
    safe = re.sub(r"[^a-z0-9._-]+", "-", filename.lower()).strip("-") or "export"
    return Response(
        "﻿" + csv_text(columns, rows),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="ela-{safe}.csv"', "Cache-Control": "no-store"},
    )
