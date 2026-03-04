"""언어모델용 컨텍스트 데이터셋 생성 유틸리티.

예: 토큰 id가 [x0, x1, x2, x3, x4]이고 context_len=3이면
- 입력: [x0, x1, x2] -> 정답: x3
- 입력: [x1, x2, x3] -> 정답: x4
"""

from __future__ import annotations

import numpy as np


def make_context_dataset(token_ids: np.ndarray, context_len: int) -> tuple[np.ndarray, np.ndarray]:
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be 1D")
    if context_len <= 0:
        raise ValueError("context_len must be > 0")
    if len(token_ids) <= context_len:
        raise ValueError("token_ids too short for context_len")

    n = len(token_ids) - context_len
    X = np.empty((n, context_len), dtype=np.int64)
    y = np.empty((n,), dtype=np.int64)
    for i in range(n):
        X[i] = token_ids[i : i + context_len]
        y[i] = token_ids[i + context_len]
    return X, y
