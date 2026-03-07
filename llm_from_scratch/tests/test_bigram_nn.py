"""신경망 빅램 모델 자기점검 테스트."""

import unittest

import numpy as np

import conftest  # noqa: F401
from bigram_nn import (
    BigramNNConfig,
    bigram_probs,
    init_W,
    loss_and_grad_W,
    train_bigram_nn,
)


class TestInitW(unittest.TestCase):
    def test_shape(self):
        rng = np.random.default_rng(0)
        W = init_W(10, rng)
        self.assertEqual(W.shape, (10, 10))

    def test_zero_vocab_raises(self):
        with self.assertRaises(ValueError):
            init_W(0, np.random.default_rng(0))


class TestLossAndGradW(unittest.TestCase):
    def test_loss_positive(self):
        rng = np.random.default_rng(0)
        W = init_W(5, rng)
        prev = np.array([0, 1, 2], dtype=np.int64)
        nxt = np.array([1, 2, 3], dtype=np.int64)
        loss, grad = loss_and_grad_W(W, prev, nxt)
        self.assertGreater(loss, 0.0)
        self.assertEqual(grad.shape, W.shape)

    def test_empty_batch_raises(self):
        W = np.zeros((3, 3))
        with self.assertRaises(ValueError):
            loss_and_grad_W(W, np.array([], dtype=np.int64), np.array([], dtype=np.int64))


class TestTrainBigramNN(unittest.TestCase):
    def test_loss_decreases(self):
        ids = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64)
        prev, nxt = ids[:-1], ids[1:]
        cfg = BigramNNConfig(lr=1.0, epochs=20, batch_size=64, seed=0)
        W, history = train_bigram_nn(prev, nxt, vocab_size=3, config=cfg)
        self.assertGreater(history[0]["train_loss"], history[-1]["train_loss"])

    def test_output_shape(self):
        ids = np.array([0, 1, 0, 1, 0], dtype=np.int64)
        cfg = BigramNNConfig(epochs=3)
        W, history = train_bigram_nn(ids[:-1], ids[1:], vocab_size=2, config=cfg)
        self.assertEqual(W.shape, (2, 2))
        self.assertEqual(len(history), 3)


class TestBigramProbs(unittest.TestCase):
    def test_sums_to_one(self):
        W = np.random.default_rng(0).normal(size=(5, 5))
        probs = bigram_probs(W, 2)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=7)

    def test_zero_temperature_raises(self):
        W = np.zeros((3, 3))
        with self.assertRaises(ValueError):
            bigram_probs(W, 0, temperature=0.0)


if __name__ == "__main__":
    unittest.main()
