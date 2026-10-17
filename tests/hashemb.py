"""Deterministic bag-of-words hashing embedding (no model) for tests."""

from __future__ import annotations

import hashlib
import math
import re


class HashEmb:
    """Projects word counts onto ``dim`` buckets with random signs; L2-normalised.

    Args:
        dim: Vector dimension.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-zà-ÿ0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1 if (h >> 8) % 2 else -1
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
