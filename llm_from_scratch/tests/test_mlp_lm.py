"""MLP 언어모델 자기점검 테스트."""

import unittest

import numpy as np

import conftest  # noqa: F401
from dataset_lm import make_context_dataset
from mlp_lm import (
    MLPLMConfig,
    forward,
    init_params,
    loss_and_grads,
    next_token_probs,
    train_mlp_lm,
)


class TestForward(unittest.TestCase):
    def test_output_shape(self):
        cfg = MLPLMConfig(context_len=3, embed_dim=8, hidden_dim=16)
        rng = np.random.default_rng(0)
        params = init_params(5, config=cfg, rng=rng)
        X = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        logits, cache = forward(params, X)
        self.assertEqual(logits.shape, (2, 5))  # (B=2, V=5)


class TestLossAndGrads(unittest.TestCase):
    def test_loss_positive(self):
        cfg = MLPLMConfig(context_len=2, embed_dim=4, hidden_dim=8)
        rng = np.random.default_rng(0)
        params = init_params(3, config=cfg, rng=rng)
        X = np.array([[0, 1], [1, 2]], dtype=np.int64)
        y = np.array([2, 0], dtype=np.int64)
        loss, grads = loss_and_grads(params, X, y)
        self.assertGreater(loss, 0.0)
        self.assertEqual(grads.E.shape, params.E.shape)
        self.assertEqual(grads.W1.shape, params.W1.shape)


class TestTrainMLPLM(unittest.TestCase):
    def test_loss_decreases(self):
        ids = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
        X, y = make_context_dataset(ids, context_len=2)
        cfg = MLPLMConfig(context_len=2, embed_dim=8, hidden_dim=16, lr=0.5, epochs=30, batch_size=64, seed=0)
        params, history = train_mlp_lm(X, y, vocab_size=3, config=cfg)
        self.assertGreater(history[0]["train_loss"], history[-1]["train_loss"])


class TestNextTokenProbs(unittest.TestCase):
    def test_sums_to_one(self):
        cfg = MLPLMConfig(context_len=3, embed_dim=8, hidden_dim=16)
        rng = np.random.default_rng(0)
        params = init_params(5, config=cfg, rng=rng)
        ctx = np.array([0, 1, 2], dtype=np.int64)
        probs = next_token_probs(params, ctx)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=7)
        self.assertEqual(probs.shape, (5,))


if __name__ == "__main__":
    unittest.main()
