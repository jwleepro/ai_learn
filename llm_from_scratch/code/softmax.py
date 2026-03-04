"""softmax / log_softmax (numpy).

수치적으로 안정(stable)하게 계산하기 위해 max-shift를 사용합니다.
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
