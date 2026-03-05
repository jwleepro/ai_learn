"""샘플링 유틸리티(temperature / top-k / top-p).

- `sample_from_logits`: logits(점수)에서 샘플링
- `sample_from_probs`: 확률분포에서 샘플링

주의: 확률(probs)에서 temperature를 적용할 때는
`p ** (1/temperature)` 형태로 “뾰족/평평”하게 만듭니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from softmax import softmax


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None


def _validate_sampling_cfg(cfg: SamplingConfig) -> None:
    if cfg.temperature <= 0:
        raise ValueError("temperature must be > 0")
    if cfg.top_k is not None and cfg.top_k <= 0:
        raise ValueError("top_k must be > 0")
    if cfg.top_p is not None and not (0.0 < cfg.top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")


def _apply_top_k(probs: np.ndarray, k: int) -> np.ndarray:
    if k >= len(probs):
        return probs
    idx = np.argpartition(probs, -k)[-k:]
    mask = np.zeros_like(probs, dtype=bool)
    mask[idx] = True
    out = np.where(mask, probs, 0.0)
    s = out.sum()
    return out / s if s > 0 else probs


def _apply_top_p(probs: np.ndarray, p: float) -> np.ndarray:
    if p >= 1.0:
        return probs
    order = np.argsort(probs)[::-1]
    sorted_probs = probs[order]
    cumsum = np.cumsum(sorted_probs)
    # Keep the smallest prefix whose cumulative probability >= p.
    cutoff = int(np.searchsorted(cumsum, p, side="left"))
    keep_ids = order[: cutoff + 1]
    out = np.zeros_like(probs)
    out[keep_ids] = probs[keep_ids]
    s = out.sum()
    return out / s if s > 0 else probs


def sample_from_logits(logits: np.ndarray, rng: np.random.Generator, *, cfg: SamplingConfig) -> int:
    _validate_sampling_cfg(cfg)
    if logits.ndim != 1:
        raise ValueError("logits must be 1D")

    scaled = logits / float(cfg.temperature)
    probs = softmax(scaled, axis=0)
    if cfg.top_k is not None:
        probs = _apply_top_k(probs, int(cfg.top_k))
    if cfg.top_p is not None:
        probs = _apply_top_p(probs, float(cfg.top_p))
    return int(rng.choice(len(probs), p=probs))


def sample_from_probs(probs: np.ndarray, rng: np.random.Generator, *, cfg: SamplingConfig) -> int:
    _validate_sampling_cfg(cfg)
    if probs.ndim != 1:
        raise ValueError("probs must be 1D")
    if np.any(probs < 0):
        raise ValueError("probs must be non-negative")
    s = float(probs.sum())
    if s <= 0:
        raise ValueError("probs must have positive sum")

    p = probs / s
    if cfg.temperature != 1.0:
        p = p ** (1.0 / float(cfg.temperature))
        p = p / float(p.sum())
    if cfg.top_k is not None:
        p = _apply_top_k(p, int(cfg.top_k))
    if cfg.top_p is not None:
        p = _apply_top_p(p, float(cfg.top_p))
    return int(rng.choice(len(p), p=p))
