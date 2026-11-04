"""Smallest token window that holds every query term present in a document."""

from collections import Counter


def min_window(tokens: list[str], terms: set[str]) -> int | None:
    """Length of the shortest window containing each of `terms` that occurs in `tokens`.

    Terms absent from the document are ignored, so the result measures how
    tightly the matched terms cluster. Returns None when no term matches.
    """
    present = terms & set(tokens)
    if not present:
        return None
    need = len(present)
    inside: Counter[str] = Counter()
    covered = 0
    best: int | None = None
    left = 0
    for right, token in enumerate(tokens):
        if token not in present:
            continue
        inside[token] += 1
        if inside[token] == 1:
            covered += 1
        while covered == need:
            span = right - left + 1
            if best is None or span < best:
                best = span
            dropped = tokens[left]
            left += 1
            if dropped in present:
                inside[dropped] -= 1
                if inside[dropped] == 0:
                    covered -= 1
    return best
