"""BPE 토크나이저 자기점검 테스트."""

import tempfile
import unittest
from pathlib import Path

import conftest  # noqa: F401
from bpe_tokenizer import BPETokenizer


class TestBPETokenizer(unittest.TestCase):
    def setUp(self):
        self.text = "low lower newest widest low low"
        self.tok = BPETokenizer.train(self.text, num_merges=10)

    def test_vocab_size_positive(self):
        self.assertGreater(self.tok.vocab_size, 0)

    def test_encode_returns_ids(self):
        ids = self.tok.encode(self.text)
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(i, int) for i in ids))
        self.assertTrue(all(0 <= i < self.tok.vocab_size for i in ids))

    def test_decode_roundtrip(self):
        ids = self.tok.encode(self.text)
        decoded = self.tok.decode(ids)
        # BPE는 공백 처리가 단순해서 완벽히 동일하진 않을 수 있지만,
        # 단어들은 보존되어야 함
        for word in self.text.split():
            self.assertIn(word, decoded)

    def test_encode_tokens(self):
        tokens = self.tok.encode_tokens(self.text)
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(t, str) for t in tokens))

    def test_merges_recorded(self):
        self.assertGreater(len(self.tok.merges), 0)
        self.assertLessEqual(len(self.tok.merges), 10)

    def test_save_load_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bpe.json"
            self.tok.save_json(path)
            loaded = BPETokenizer.load_json(path)
            self.assertEqual(loaded.merges, self.tok.merges)
            self.assertEqual(loaded.id_to_token, self.tok.id_to_token)
            ids_orig = self.tok.encode(self.text)
            ids_loaded = loaded.encode(self.text)
            self.assertEqual(ids_orig, ids_loaded)

    def test_empty_text_raises(self):
        with self.assertRaises(ValueError):
            BPETokenizer.train("", num_merges=5)

    def test_more_merges_smaller_tokens(self):
        tok_few = BPETokenizer.train(self.text, num_merges=2)
        tok_many = BPETokenizer.train(self.text, num_merges=50)
        ids_few = tok_few.encode(self.text)
        ids_many = tok_many.encode(self.text)
        # merge가 많을수록 토큰 수가 줄어듦(또는 같음)
        self.assertGreaterEqual(len(ids_few), len(ids_many))


if __name__ == "__main__":
    unittest.main()
