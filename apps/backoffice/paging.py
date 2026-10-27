"""Page numbers for the long lists (feedback, conversations)."""

from __future__ import annotations

from dataclasses import dataclass

PAGE_SIZE = 50


@dataclass(frozen=True)
class Paging:
    """Current page of a list of ``total`` rows."""

    page: int
    size: int
    total: int

    def with_total(self, total: int) -> Paging:
        """Return the same page with the row count known after the query."""
        return Paging(page=self.page, size=self.size, total=total)

    @property
    def offset(self) -> int:
        """Rows to skip."""
        return (self.page - 1) * self.size

    @property
    def pages(self) -> int:
        """Number of pages (at least 1)."""
        return max(1, -(-self.total // self.size))

    @property
    def has_prev(self) -> bool:
        """Whether a previous page exists."""
        return self.page > 1

    @property
    def has_next(self) -> bool:
        """Whether a next page exists."""
        return self.page < self.pages

    @property
    def first(self) -> int:
        """1-based index of the first row of the page (0 when empty)."""
        return 0 if self.total == 0 else self.offset + 1

    @property
    def last(self) -> int:
        """1-based index of the last row of the page."""
        return min(self.total, self.offset + self.size)


def paging(page: str | int | None, total: int = 0, size: int = PAGE_SIZE) -> Paging:
    """Parse the ``page`` query parameter (defaults to 1, never below 1)."""
    try:
        number = max(1, int(page or 1))
    except (TypeError, ValueError):
        number = 1
    return Paging(page=number, size=size, total=total)
