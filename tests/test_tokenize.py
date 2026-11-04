import unittest

from proxrank.tokenize import tokenize


class TokenizeTest(unittest.TestCase):
    def test_lowercases_and_splits_on_punctuation(self):
        self.assertEqual(tokenize("Pool-Exhausted: 5xx!"), ["pool", "exhausted", "5xx"])

    def test_empty_text_has_no_tokens(self):
        self.assertEqual(tokenize("  ...  "), [])


if __name__ == "__main__":
    unittest.main()
