"""Self-Attention 자기점검 테스트."""

import unittest

import numpy as np

import conftest  # noqa: F401
from attention_numpy import causal_mask, self_attention


class TestCausalMask(unittest.TestCase):
    def test_lower_triangle(self):
        scores = np.ones((4, 4))
        masked = causal_mask(scores)
        # 대각선 아래/위 확인: 상삼각(k=1)은 -1e9
        for i in range(4):
            for j in range(4):
                if j > i:
                    self.assertAlmostEqual(masked[i, j], -1e9)
                else:
                    self.assertAlmostEqual(masked[i, j], 1.0)

    def test_non_square_raises(self):
        with self.assertRaises(ValueError):
            causal_mask(np.ones((3, 4)))


class TestSelfAttention(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.T, self.D, self.Dh = 6, 8, 4
        self.X = self.rng.normal(size=(self.T, self.D))
        self.Wq = self.rng.normal(size=(self.D, self.Dh))
        self.Wk = self.rng.normal(size=(self.D, self.Dh))
        self.Wv = self.rng.normal(size=(self.D, self.Dh))

    def test_output_shapes(self):
        weights, out = self_attention(self.X, self.Wq, self.Wk, self.Wv, causal=True)
        self.assertEqual(weights.shape, (self.T, self.T))
        self.assertEqual(out.shape, (self.T, self.Dh))

    def test_weights_sum_to_one(self):
        weights, _ = self_attention(self.X, self.Wq, self.Wk, self.Wv, causal=True)
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(self.T), atol=1e-7)

    def test_causal_weights_zero_future(self):
        weights, _ = self_attention(self.X, self.Wq, self.Wk, self.Wv, causal=True)
        for i in range(self.T):
            for j in range(i + 1, self.T):
                self.assertAlmostEqual(float(weights[i, j]), 0.0, places=5)

    def test_non_causal_can_see_future(self):
        weights, _ = self_attention(self.X, self.Wq, self.Wk, self.Wv, causal=False)
        # 첫 번째 위치가 마지막 위치에도 weight > 0
        self.assertGreater(float(weights[0, -1]), 0.0)


if __name__ == "__main__":
    unittest.main()
