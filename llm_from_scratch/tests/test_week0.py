"""Week 0: 내부 단위 테스트.

학습자용 빠른 확인은 `test_core_week0.py`를 먼저 보는 편이 낫습니다.
"""

import unittest

import numpy as np

import conftest  # noqa: F401 — sys.path 설정
from week0_dl_basics import burger_finance, fit_line_gd, relu, simple_neuron


class TestSimpleNeuron(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(simple_neuron(2.0, 3.0, 1.0), 7.0)

    def test_zero_weight(self):
        self.assertAlmostEqual(simple_neuron(5.0, 0.0, 2.0), 2.0)


class TestBurgerFinance(unittest.TestCase):
    def test_shape(self):
        result = burger_finance(np.array([10, 20, 30]))
        self.assertEqual(result.shape, (2,))

    def test_known_value(self):
        sales = np.array([10.0, 20.0, 30.0])
        result = burger_finance(sales)
        # revenue = 5000*10 + 2000*20 + 1500*30 + 10000 = 50000+40000+45000+10000 = 145000
        # profit  = 2000*10 + 1000*20 + 500*30  - 50000 = 20000+20000+15000-50000 = 5000
        np.testing.assert_allclose(result, [145000.0, 5000.0])

    def test_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            burger_finance(np.array([1, 2]))


class TestRelu(unittest.TestCase):
    def test_positive(self):
        np.testing.assert_array_equal(relu(np.array([1.0, 2.0])), [1.0, 2.0])

    def test_negative(self):
        np.testing.assert_array_equal(relu(np.array([-1.0, -0.5])), [0.0, 0.0])

    def test_mixed(self):
        np.testing.assert_array_equal(relu(np.array([-3.0, 0.0, 5.0])), [0.0, 0.0, 5.0])


class TestFitLineGD(unittest.TestCase):
    def test_loss_decreases(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = 2.0 * x + 1.0
        result = fit_line_gd(x, y, lr=0.01, steps=100)
        self.assertGreater(result.losses[0], result.losses[-1])

    def test_converges_to_known(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 3.0 * x + 2.0
        result = fit_line_gd(x, y, lr=0.01, steps=2000)
        self.assertAlmostEqual(result.w, 3.0, places=1)
        self.assertAlmostEqual(result.b, 2.0, places=1)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            fit_line_gd(np.array([]), np.array([]))


if __name__ == "__main__":
    unittest.main()
