"""샘플링 유틸리티 자기점검 테스트."""

import unittest

import numpy as np

import conftest  # noqa: F401
from sampling import SamplingConfig, sample_from_logits, sample_from_probs


class TestSampleFromProbs(unittest.TestCase):
    def test_deterministic_peak(self):
        probs = np.array([0.0, 0.0, 1.0, 0.0])
        rng = np.random.default_rng(42)
        cfg = SamplingConfig()
        result = sample_from_probs(probs, rng, cfg=cfg)
        self.assertEqual(result, 2)

    def test_returns_valid_id(self):
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        rng = np.random.default_rng(0)
        cfg = SamplingConfig()
        for _ in range(50):
            result = sample_from_probs(probs, rng, cfg=cfg)
            self.assertIn(result, range(4))

    def test_zero_temperature_raises(self):
        with self.assertRaises(ValueError):
            sample_from_probs(
                np.array([0.5, 0.5]),
                np.random.default_rng(0),
                cfg=SamplingConfig(temperature=0.0),
            )

    def test_top_k(self):
        probs = np.array([0.1, 0.2, 0.3, 0.4])
        rng = np.random.default_rng(0)
        cfg = SamplingConfig(top_k=2)
        results = {sample_from_probs(probs, rng, cfg=cfg) for _ in range(200)}
        # top-2 should only ever pick id 2 or 3
        self.assertTrue(results.issubset({2, 3}))


class TestSampleFromLogits(unittest.TestCase):
    def test_deterministic_peak(self):
        logits = np.array([-100.0, -100.0, 100.0, -100.0])
        rng = np.random.default_rng(0)
        cfg = SamplingConfig()
        result = sample_from_logits(logits, rng, cfg=cfg)
        self.assertEqual(result, 2)

    def test_returns_valid_id(self):
        logits = np.array([0.0, 0.0, 0.0])
        rng = np.random.default_rng(0)
        cfg = SamplingConfig()
        for _ in range(50):
            result = sample_from_logits(logits, rng, cfg=cfg)
            self.assertIn(result, range(3))


if __name__ == "__main__":
    unittest.main()
