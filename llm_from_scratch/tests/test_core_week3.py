import sys
import unittest
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from dataset_lm import make_context_dataset  # noqa: E402
from mlp_lm import MLPLMConfig, train_mlp_lm  # noqa: E402
from tokenizer_char import CharTokenizer  # noqa: E402


class TestWeek3MLPLM(unittest.TestCase):
    def test_context_dataset_shapes(self) -> None:
        tok = CharTokenizer.from_text("abcd")
        ids = np.array(tok.encode("abcd"), dtype=np.int64)
        X, y = make_context_dataset(ids, context_len=2)
        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(y.shape, (2,))

    def test_mlp_lm_trains_loss_down(self) -> None:
        # Small synthetic dataset; we only check that optimization moves.
        text = "012012012012012012"
        tok = CharTokenizer.from_text(text)
        ids = np.array(tok.encode(text), dtype=np.int64)
        X, y = make_context_dataset(ids, context_len=4)

        cfg = MLPLMConfig(context_len=4, embed_dim=8, hidden_dim=32, lr=0.3, epochs=30, batch_size=64, seed=0)
        params, hist = train_mlp_lm(X, y, tok.vocab_size, config=cfg)
        _ = params
        self.assertLess(hist[-1]["train_loss"], hist[0]["train_loss"])


if __name__ == "__main__":
    unittest.main()

