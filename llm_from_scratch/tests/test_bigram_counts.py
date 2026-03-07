"""빅램 카운트 모델 자기점검 테스트."""

import unittest

import numpy as np

import conftest  # noqa: F401
from bigram_counts import build_bigram_counts, counts_to_probs


class TestBuildBigramCounts(unittest.TestCase):
    def test_simple(self):
        ids = np.array([0, 1, 0, 1, 0], dtype=np.int64)
        counts = build_bigram_counts(ids, vocab_size=2)
        # 0->1 x2, 1->0 x2
        self.assertEqual(counts[0, 1], 2)
        self.assertEqual(counts[1, 0], 2)
        self.assertEqual(counts[0, 0], 0)
        self.assertEqual(counts[1, 1], 0)

    def test_shape(self):
        ids = np.array([0, 1, 2, 3], dtype=np.int64)
        counts = build_bigram_counts(ids, vocab_size=5)
        self.assertEqual(counts.shape, (5, 5))

    def test_too_short_raises(self):
        with self.assertRaises(ValueError):
            build_bigram_counts(np.array([0], dtype=np.int64), vocab_size=1)


class TestCountsToProbs(unittest.TestCase):
    def test_rows_sum_to_one(self):
        counts = np.array([[3, 1], [2, 2]], dtype=np.int64)
        probs = counts_to_probs(counts)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-10)

    def test_zero_row_becomes_uniform(self):
        counts = np.array([[0, 0], [1, 1]], dtype=np.int64)
        probs = counts_to_probs(counts)
        np.testing.assert_allclose(probs[0], [0.5, 0.5])

    def test_smoothing(self):
        counts = np.array([[1, 0], [0, 1]], dtype=np.int64)
        probs = counts_to_probs(counts, smoothing=1.0)
        # Row 0: (1+1, 0+1) = (2,1) -> (2/3, 1/3)
        self.assertAlmostEqual(float(probs[0, 0]), 2.0 / 3.0, places=7)
        self.assertAlmostEqual(float(probs[0, 1]), 1.0 / 3.0, places=7)

    def test_negative_smoothing_raises(self):
        with self.assertRaises(ValueError):
            counts_to_probs(np.array([[1, 1]]), smoothing=-1.0)


if __name__ == "__main__":
    unittest.main()
