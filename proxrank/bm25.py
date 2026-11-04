"""BM25 over tokenized documents, keeping token positions for reranking."""

import math
from collections import Counter

from .tokenize import tokenize

K1 = 1.2
B = 0.75


class BM25Index:
    def __init__(self, docs: dict[str, str]) -> None:
        if not docs:
            raise ValueError("cannot index an empty corpus")
        self.tokens = {doc_id: tokenize(text) for doc_id, text in docs.items()}
        self.tf = {doc_id: Counter(toks) for doc_id, toks in self.tokens.items()}
        self.avg_length = sum(len(t) for t in self.tokens.values()) / len(docs) or 1.0
        df: Counter[str] = Counter()
        for counter in self.tf.values():
            df.update(counter.keys())
        total = len(docs)
        self.idf = {t: math.log(1 + (total - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, doc_id: str, terms: list[str]) -> float:
        counter = self.tf[doc_id]
        norm = 1 - B + B * len(self.tokens[doc_id]) / self.avg_length
        total = 0.0
        for term in terms:
            freq = counter.get(term, 0)
            if freq:
                total += self.idf[term] * freq * (K1 + 1) / (freq + K1 * norm)
        return total

    def search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        terms = list(dict.fromkeys(tokenize(query)))
        scored = [(doc_id, self.score(doc_id, terms)) for doc_id in self.tokens]
        # Ties break on id so rankings are reproducible.
        ranked = sorted((s for s in scored if s[1] > 0), key=lambda s: (-s[1], s[0]))
        return ranked[:limit]
