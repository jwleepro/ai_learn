"""softmax / log_softmax (numpy).

점수(logits)를 확률로 바꾸는 함수입니다.
큰 수를 먼저 빼서(max-shift) 계산이 터지지 않게 합니다.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, *, axis: int = -1) -> np.ndarray:
    if logits.size == 0:
        raise ValueError("logits must not be empty")
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def log_softmax(logits: np.ndarray, *, axis: int = -1) -> np.ndarray:
    if logits.size == 0:
        raise ValueError("logits must not be empty")
    shifted = logits - logits.max(axis=axis, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    return shifted - logsumexp
