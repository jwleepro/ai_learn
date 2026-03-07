"""컨텍스트 데이터셋 생성 자기점검 테스트."""

import unittest

import numpy as np

import conftest  # noqa: F401
from dataset_lm import make_context_dataset


class TestMakeContextDataset(unittest.TestCase):
    def test_shape(self):
        ids = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        X, y = make_context_dataset(ids, context_len=3)
        # n = 5 - 3 = 2 samples
        self.assertEqual(X.shape, (2, 3))
        self.assertEqual(y.shape, (2,))

    def test_values(self):
        ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        X, y = make_context_dataset(ids, context_len=2)
        # sample 0: [10,20] -> 30
        # sample 1: [20,30] -> 40
        # sample 2: [30,40] -> 50
        np.testing.assert_array_equal(X[0], [10, 20])
        self.assertEqual(y[0], 30)
        np.testing.assert_array_equal(X[2], [30, 40])
        self.assertEqual(y[2], 50)

    def test_too_short_raises(self):
        with self.assertRaises(ValueError):
            make_context_dataset(np.array([0, 1], dtype=np.int64), context_len=3)

    def test_context_len_zero_raises(self):
        with self.assertRaises(ValueError):
            make_context_dataset(np.array([0, 1, 2], dtype=np.int64), context_len=0)


if __name__ == "__main__":
    unittest.main()
