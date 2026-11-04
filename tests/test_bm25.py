import unittest

from proxrank.bm25 import BM25Index

DOCS = {
    "a": "connection pool exhausted under load",
    "b": "the pool is a place to swim",
    "c": "unrelated release notes",
}


class BM25IndexTest(unittest.TestCase):
    def test_ranks_the_document_with_more_query_terms_first(self):
        ranked = BM25Index(DOCS).search("connection pool")
        self.assertEqual([doc_id for doc_id, _ in ranked], ["a", "b"])

    def test_documents_without_a_hit_are_dropped(self):
        self.assertEqual(BM25Index(DOCS).search("kubernetes"), [])

    def test_limit_truncates_results(self):
        self.assertEqual(len(BM25Index(DOCS).search("pool", limit=1)), 1)

    def test_empty_corpus_is_rejected(self):
        with self.assertRaises(ValueError):
            BM25Index({})


if __name__ == "__main__":
    unittest.main()
