"""학습자용 Week 2 핵심 스모크 테스트."""

import sys
import unittest
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from bigram_nn import BigramNNConfig, bigram_probs, train_bigram_nn  # noqa: E402
from tokenizer_char import CharTokenizer  # noqa: E402


class TestWeek2NeuralBigram(unittest.TestCase):
    def test_neural_bigram_learns_deterministic_pairs(self) -> None:
        text = "abababab"
        tok = CharTokenizer.from_text(text)
        ids = np.array(tok.encode(text), dtype=np.int64)
        prev_ids = ids[:-1]
        next_ids = ids[1:]

        cfg = BigramNNConfig(lr=1.0, epochs=200, batch_size=64, seed=0, init_scale=0.01)
        W, hist = train_bigram_nn(prev_ids, next_ids, tok.vocab_size, config=cfg)

        a_id = tok.encode("a")[0]
        b_id = tok.encode("b")[0]
        p_b_given_a = float(bigram_probs(W, a_id)[b_id])
        p_a_given_b = float(bigram_probs(W, b_id)[a_id])

        self.assertGreater(p_b_given_a, 0.85)
        self.assertGreater(p_a_given_b, 0.85)

        self.assertLess(hist[-1]["train_loss"], hist[0]["train_loss"])


if __name__ == "__main__":
    unittest.main()

