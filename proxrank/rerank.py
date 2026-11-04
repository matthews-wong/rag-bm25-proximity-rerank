"""Rerank BM25 candidates with a bounded term-proximity bonus."""

from .bm25 import BM25Index
from .proximity import min_window
from .tokenize import tokenize

DEFAULT_WEIGHT = 1.0
DEFAULT_CANDIDATES = 20


def proximity_bonus(tokens: list[str], terms: list[str]) -> float:
    """coverage / span, in (0, 1]; 1.0 means the matched terms form an exact phrase."""
    unique = set(terms)
    span = min_window(tokens, unique)
    if span is None:
        return 0.0
    matched = len(unique & set(tokens))
    return (matched / len(unique)) * (matched / span)


def search(
    index: BM25Index,
    query: str,
    limit: int = 10,
    weight: float = DEFAULT_WEIGHT,
    candidates: int = DEFAULT_CANDIDATES,
) -> list[tuple[str, float]]:
    if weight < 0:
        raise ValueError(f"weight must be >= 0, got {weight}")
    if candidates < limit:
        raise ValueError(f"candidates ({candidates}) must be >= limit ({limit})")
    terms = list(dict.fromkeys(tokenize(query)))
    pool = index.search(query, limit=candidates)
    # The bonus scales with the top BM25 score so it stays comparable across queries.
    scale = pool[0][1] if pool else 0.0
    rescored = [
        (doc_id, score + weight * scale * proximity_bonus(index.tokens[doc_id], terms))
        for doc_id, score in pool
    ]
    return sorted(rescored, key=lambda s: (-s[1], s[0]))[:limit]
