"""softmax / log_softmax 자기점검 테스트."""

import unittest

import numpy as np

import conftest  # noqa: F401
from softmax import log_softmax, softmax


class TestSoftmax(unittest.TestCase):
    def test_sums_to_one(self):
        logits = np.array([1.0, 2.0, 3.0])
        probs = softmax(logits, axis=0)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=7)

    def test_all_positive(self):
        probs = softmax(np.array([-10.0, 0.0, 10.0]), axis=0)
        self.assertTrue(np.all(probs > 0))

    def test_2d(self):
        logits = np.array([[1.0, 2.0], [3.0, 4.0]])
        probs = softmax(logits, axis=1)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-7)

    def test_numerical_stability(self):
        logits = np.array([1000.0, 1001.0, 1002.0])
        probs = softmax(logits, axis=0)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=7)
        self.assertFalse(np.any(np.isnan(probs)))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            softmax(np.array([]), axis=0)


class TestLogSoftmax(unittest.TestCase):
    def test_consistent_with_softmax(self):
        logits = np.array([1.0, 2.0, 3.0])
        log_probs = log_softmax(logits, axis=0)
        probs = softmax(logits, axis=0)
        np.testing.assert_allclose(np.exp(log_probs), probs, atol=1e-10)

    def test_all_non_positive(self):
        log_probs = log_softmax(np.array([5.0, 3.0, 1.0]), axis=0)
        self.assertTrue(np.all(log_probs <= 0))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            log_softmax(np.array([]), axis=0)


if __name__ == "__main__":
    unittest.main()
