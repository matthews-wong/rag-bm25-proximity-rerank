import json
import tempfile
import unittest
from pathlib import Path

from proxrank.corpus import load_docs


def write(items) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(items, handle)
    handle.close()
    return Path(handle.name)


class LoadDocsTest(unittest.TestCase):
    def test_loads_id_to_text(self):
        path = write([{"id": "a", "text": "hello"}])
        self.addCleanup(path.unlink)
        self.assertEqual(load_docs(path), {"a": "hello"})

    def test_duplicate_id_names_the_id(self):
        path = write([{"id": "a", "text": "x"}, {"id": "a", "text": "y"}])
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "duplicate document id 'a'"):
            load_docs(path)

    def test_missing_text_names_the_position(self):
        path = write([{"id": "a"}])
        self.addCleanup(path.unlink)
        with self.assertRaisesRegex(ValueError, "position 0"):
            load_docs(path)


if __name__ == "__main__":
    unittest.main()
