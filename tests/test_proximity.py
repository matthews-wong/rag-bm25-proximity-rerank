import unittest

from proxrank.proximity import min_window


class MinWindowTest(unittest.TestCase):
    def test_adjacent_phrase_spans_its_own_length(self):
        self.assertEqual(min_window("a connection pool exhausted b".split(), {"connection", "pool", "exhausted"}), 3)

    def test_scattered_terms_span_the_gap(self):
        tokens = "pool x x x x connection".split()
        self.assertEqual(min_window(tokens, {"pool", "connection"}), 6)

    def test_picks_the_tightest_of_several_occurrences(self):
        tokens = "pool x x x connection pool connection".split()
        self.assertEqual(min_window(tokens, {"pool", "connection"}), 2)

    def test_missing_terms_are_ignored(self):
        self.assertEqual(min_window("pool connection".split(), {"pool", "connection", "kafka"}), 2)

    def test_no_match_returns_none(self):
        self.assertIsNone(min_window("nothing here".split(), {"pool"}))

    def test_single_term_spans_one_token(self):
        self.assertEqual(min_window("x pool x".split(), {"pool"}), 1)


if __name__ == "__main__":
    unittest.main()
