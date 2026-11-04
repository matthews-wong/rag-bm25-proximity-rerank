"""Compare plain BM25 with the proximity rerank on labelled fixtures."""

from pathlib import Path

from .bm25 import BM25Index
from .corpus import load_docs, load_queries
from .rerank import search

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
K = 3
WEIGHTS = (0.0, 0.5, 1.0, 2.0)


def evaluate(
    index: BM25Index, queries: list[dict[str, str]], weight: float, k: int = K
) -> tuple[float, float]:
    """Return (recall@k, MRR) for single-relevant-document queries."""
    hits = 0
    reciprocal = 0.0
    pool = len(index.tokens)
    for case in queries:
        ranked = [d for d, _ in search(index, case["query"], limit=pool, weight=weight, candidates=pool)]
        if case["relevant"] in ranked[:k]:
            hits += 1
        if case["relevant"] in ranked:
            reciprocal += 1 / (ranked.index(case["relevant"]) + 1)
    return hits / len(queries), reciprocal / len(queries)


def main() -> None:
    docs = load_docs(FIXTURES / "docs.json")
    queries = load_queries(FIXTURES / "queries.json")
    index = BM25Index(docs)
    print(f"fixtures: {len(docs)} docs, {len(queries)} queries")
    print(f"{'run':<24}{f'recall@{K}':>10}{'MRR':>8}")
    for weight in WEIGHTS:
        recall, mrr = evaluate(index, queries, weight)
        name = "plain BM25" if weight == 0 else f"proximity weight {weight:g}"
        print(f"{name:<24}{recall:>10.3f}{mrr:>8.3f}")


if __name__ == "__main__":
    main()
