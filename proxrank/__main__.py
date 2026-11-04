"""Command line search: python3 -m proxrank "query" --corpus fixtures/docs.json."""

import argparse

from .bm25 import BM25Index
from .corpus import load_docs
from .rerank import DEFAULT_CANDIDATES, DEFAULT_WEIGHT, search


def main() -> None:
    parser = argparse.ArgumentParser(prog="proxrank", description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--corpus", default="fixtures/docs.json")
    parser.add_argument("--weight", type=float, default=DEFAULT_WEIGHT)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    docs = load_docs(args.corpus)
    results = search(
        BM25Index(docs), args.query, limit=args.top, weight=args.weight, candidates=args.candidates
    )
    if not results:
        print(f"no documents match {args.query!r}")
        return
    for doc_id, score in results:
        print(f"{score:6.3f}  {doc_id}")


if __name__ == "__main__":
    main()
