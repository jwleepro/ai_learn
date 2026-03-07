"""학습자용 Week 1 핵심 스모크 테스트."""

import sys
import unittest
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from bigram_counts import build_bigram_counts, counts_to_probs  # noqa: E402
from tokenizer_char import CharTokenizer  # noqa: E402


class TestWeek1Core(unittest.TestCase):
    def test_tokenizer_roundtrip(self) -> None:
        text = "ababa"
        tok = CharTokenizer.from_text(text)
        ids = tok.encode(text)
        self.assertEqual(tok.decode(ids), text)
        self.assertEqual(tok.vocab_size, 2)

    def test_bigram_counts_simple(self) -> None:
        # text: "abab"
        # a->b occurs 2, b->a occurs 1
        tok = CharTokenizer.from_text("abab")
        ids = np.array(tok.encode("abab"), dtype=np.int64)
        counts = build_bigram_counts(ids, tok.vocab_size)

        a_id = tok.encode("a")[0]
        b_id = tok.encode("b")[0]

        self.assertEqual(int(counts[a_id, b_id]), 2)
        self.assertEqual(int(counts[b_id, a_id]), 1)

    def test_probs_rows_sum_to_one(self) -> None:
        tok = CharTokenizer.from_text("abab")
        ids = np.array(tok.encode("abab"), dtype=np.int64)
        counts = build_bigram_counts(ids, tok.vocab_size)
        probs = counts_to_probs(counts, smoothing=0.0)
        row_sums = probs.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0))


if __name__ == "__main__":
    unittest.main()

