import unittest

from proxrank.bm25 import BM25Index
from proxrank.corpus import load_docs, load_queries
from proxrank.evaluate import FIXTURES, evaluate


class EvaluateTest(unittest.TestCase):
    def setUp(self):
        self.index = BM25Index(load_docs(FIXTURES / "docs.json"))
        self.queries = load_queries(FIXTURES / "queries.json")

    def test_rerank_improves_mrr_over_plain_bm25(self):
        _, plain = evaluate(self.index, self.queries, weight=0.0)
        _, boosted = evaluate(self.index, self.queries, weight=1.0)
        self.assertGreater(boosted, plain)

    def test_recall_is_a_fraction(self):
        recall, mrr = evaluate(self.index, self.queries, weight=1.0)
        self.assertTrue(0 <= recall <= 1 and 0 <= mrr <= 1)


if __name__ == "__main__":
    unittest.main()
