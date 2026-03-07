"""Transformer forward 자기점검 테스트."""

import unittest

import numpy as np

import conftest  # noqa: F401
from transformer_numpy import TransformerConfig, forward, init_params, layer_norm


class TestLayerNorm(unittest.TestCase):
    def test_normalized(self):
        x = np.array([[1.0, 2.0, 3.0, 4.0]])
        g = np.ones(4)
        b = np.zeros(4)
        out = layer_norm(x, g, b)
        # 정규화 후 mean~0, std~1
        self.assertAlmostEqual(float(out.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(out.std()), 1.0, delta=0.15)


class TestTransformerForward(unittest.TestCase):
    def setUp(self):
        self.cfg = TransformerConfig(
            vocab_size=10,
            max_seq_len=16,
            d_model=8,
            n_heads=2,
            d_ff=32,
            n_layers=2,
            seed=0,
        )
        self.params = init_params(self.cfg)

    def test_logits_shape(self):
        ids = np.array([0, 1, 2, 3], dtype=np.int64)
        logits, attn_weights = forward(self.params, ids, n_heads=self.cfg.n_heads, causal=True)
        self.assertEqual(logits.shape, (4, 10))  # (T=4, V=10)

    def test_attn_weights_count(self):
        ids = np.array([0, 1, 2], dtype=np.int64)
        _, attn_weights = forward(self.params, ids, n_heads=self.cfg.n_heads, causal=True)
        self.assertEqual(len(attn_weights), self.cfg.n_layers)

    def test_attn_weights_shape(self):
        T = 5
        ids = np.arange(T, dtype=np.int64)
        _, attn_weights = forward(self.params, ids, n_heads=self.cfg.n_heads, causal=True)
        for w in attn_weights:
            self.assertEqual(w.shape, (self.cfg.n_heads, T, T))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            forward(self.params, np.array([], dtype=np.int64), n_heads=self.cfg.n_heads)

    def test_too_long_raises(self):
        ids = np.arange(self.cfg.max_seq_len + 1, dtype=np.int64) % self.cfg.vocab_size
        with self.assertRaises(ValueError):
            forward(self.params, ids, n_heads=self.cfg.n_heads)

    def test_d_model_not_divisible_raises(self):
        with self.assertRaises(ValueError):
            TransformerConfig(vocab_size=10, d_model=7, n_heads=3)
            init_params(TransformerConfig(vocab_size=10, d_model=7, n_heads=3))


if __name__ == "__main__":
    unittest.main()
