"""(단일 헤드) self-attention 계산 블록(numpy).

학습용 단순 구현입니다.

Shapes:
- X: (T, D)
- Wq/Wk/Wv: (D, Dh)
- weights: (T, T)
- out: (T, Dh)
"""

from __future__ import annotations

import numpy as np

from softmax import softmax


def causal_mask(scores: np.ndarray) -> np.ndarray:
    # scores: (T, T)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be (T, T)")
    T = scores.shape[0]
    masked = scores.copy()
    upper = np.triu(np.ones((T, T), dtype=bool), k=1)
    masked[upper] = -1e9
    return masked


def self_attention(
    X: np.ndarray,
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    *,
    causal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    X: (T, D)
    Wq/Wk/Wv: (D, Dh)

    Returns:
      weights: (T, T)
      out: (T, Dh)
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (T, D)")
    if Wq.shape[0] != X.shape[1] or Wk.shape[0] != X.shape[1] or Wv.shape[0] != X.shape[1]:
        raise ValueError("Wq/Wk/Wv first dim must match X feature dim")

    Q = X @ Wq  # (T, Dh)
    K = X @ Wk  # (T, Dh)
    V = X @ Wv  # (T, Dh)

    Dh = Q.shape[1]
    scores = (Q @ K.T) / np.sqrt(float(Dh))  # (T, T)
    if causal:
        scores = causal_mask(scores)

    weights = softmax(scores, axis=1)
    out = weights @ V
    return weights, out
