"""카운트 기반 빅램(2-gram) 유틸리티.

빅램은 “직전 토큰(prev)만 보고 다음 토큰(next)을 예측”합니다.

- counts: (V, V)  where counts[prev, next] = 등장 횟수
- probs:  (V, V)  각 prev 행(row)이 확률분포(합=1)
"""

from __future__ import annotations

import numpy as np


def build_bigram_counts(token_ids: np.ndarray, vocab_size: int) -> np.ndarray:
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be 1D")
    if len(token_ids) < 2:
        raise ValueError("token_ids must contain at least 2 tokens")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")

    prev_ids = token_ids[:-1]
    next_ids = token_ids[1:]

    counts = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    np.add.at(counts, (prev_ids, next_ids), 1)
    return counts


def counts_to_probs(counts: np.ndarray, *, smoothing: float = 0.0) -> np.ndarray:
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("counts must be a square 2D matrix")
    if smoothing < 0:
        raise ValueError("smoothing must be >= 0")

    counts_f = counts.astype(np.float64, copy=False)
    if smoothing != 0.0:
        counts_f = counts_f + smoothing

    row_sums = counts_f.sum(axis=1, keepdims=True)
    # If a row is all zeros (possible with tiny data), fall back to uniform.
    zero_rows = row_sums.squeeze(axis=1) == 0
    if np.any(zero_rows):
        counts_f = counts_f.copy()
        counts_f[zero_rows, :] = 1.0
        row_sums = counts_f.sum(axis=1, keepdims=True)

    return counts_f / row_sums


def sample_next_id(
    prev_id: int,
    probs: np.ndarray,
    rng: np.random.Generator,
    *,
    temperature: float = 1.0,
) -> int:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    row = probs[prev_id]
    if temperature != 1.0:
        row = row ** (1.0 / temperature)
        row = row / row.sum()
    return int(rng.choice(len(row), p=row))
