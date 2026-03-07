"""CharTokenizer 자기점검 테스트."""

import json
import tempfile
import unittest
from pathlib import Path

import conftest  # noqa: F401
from tokenizer_char import CharTokenizer


class TestCharTokenizer(unittest.TestCase):
    def setUp(self):
        self.text = "안녕하세요"
        self.tok = CharTokenizer.from_text(self.text)

    def test_vocab_size(self):
        self.assertEqual(self.tok.vocab_size, len(set(self.text)))

    def test_vocab_sorted(self):
        self.assertEqual(list(self.tok.vocab), sorted(set(self.text)))

    def test_encode_decode_roundtrip(self):
        ids = self.tok.encode(self.text)
        self.assertEqual(self.tok.decode(ids), self.text)

    def test_encode_unknown_char_raises(self):
        with self.assertRaises(KeyError):
            self.tok.encode("Z")

    def test_decode_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            self.tok.decode([self.tok.vocab_size])

    def test_empty_text_raises(self):
        with self.assertRaises(ValueError):
            CharTokenizer.from_text("")

    def test_duplicate_vocab_raises(self):
        with self.assertRaises(ValueError):
            CharTokenizer(("a", "a"))

    def test_save_load_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tok.json"
            self.tok.save_json(path)
            loaded = CharTokenizer.load_json(path)
            self.assertEqual(loaded.vocab, self.tok.vocab)
            ids = loaded.encode(self.text)
            self.assertEqual(loaded.decode(ids), self.text)


if __name__ == "__main__":
    unittest.main()
