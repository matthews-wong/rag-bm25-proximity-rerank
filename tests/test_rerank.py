import unittest

from proxrank.bm25 import BM25Index
from proxrank.rerank import proximity_bonus, search

SCATTERED = "pool " + "filler " * 30 + "connection " + "filler " * 30
PHRASE = "the connection pool " + "filler " * 40
DOCS = {"scattered": SCATTERED, "phrase": PHRASE, "other": "connection only here"}


class ProximityBonusTest(unittest.TestCase):
    def test_exact_phrase_scores_one(self):
        self.assertEqual(proximity_bonus("connection pool".split(), ["connection", "pool"]), 1.0)

    def test_partial_coverage_is_discounted(self):
        self.assertEqual(proximity_bonus("connection".split(), ["connection", "pool"]), 0.5)

    def test_no_match_scores_zero(self):
        self.assertEqual(proximity_bonus("x y".split(), ["pool"]), 0.0)


class SearchTest(unittest.TestCase):
    def test_phrase_document_outranks_scattered_one(self):
        ranked = search(BM25Index(DOCS), "connection pool")
        self.assertEqual(ranked[0][0], "phrase")

    def test_zero_weight_matches_plain_bm25(self):
        index = BM25Index(DOCS)
        self.assertEqual(search(index, "connection pool", weight=0.0), index.search("connection pool"))

    def test_rejects_negative_weight(self):
        with self.assertRaises(ValueError):
            search(BM25Index(DOCS), "pool", weight=-1)

    def test_rejects_candidates_below_limit(self):
        with self.assertRaises(ValueError):
            search(BM25Index(DOCS), "pool", limit=5, candidates=2)

    def test_unmatched_query_returns_nothing(self):
        self.assertEqual(search(BM25Index(DOCS), "kafka"), [])


if __name__ == "__main__":
    unittest.main()
