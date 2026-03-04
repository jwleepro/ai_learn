import sys
import unittest
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from week0_dl_basics import burger_finance, fit_line_gd, simple_neuron  # noqa: E402


class TestWeek0Basics(unittest.TestCase):
    def test_simple_neuron(self) -> None:
        self.assertEqual(simple_neuron(3, 10, 20), 50)

    def test_burger_finance(self) -> None:
        revenue, profit = burger_finance(np.array([100, 80, 120], dtype=np.float64))
        self.assertEqual(int(revenue), 850000)
        self.assertEqual(int(profit), 290000)

    def test_fit_line_gd_loss_down(self) -> None:
        x = np.array([0, 1, 2, 3, 4], dtype=np.float64)
        y = np.array([1, 3, 5, 7, 9], dtype=np.float64)
        res = fit_line_gd(x, y, lr=0.1, steps=200, w0=0.0, b0=0.0)
        self.assertLess(res.losses[-1], res.losses[0])
        self.assertAlmostEqual(res.w, 2.0, delta=0.05)
        self.assertAlmostEqual(res.b, 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()

