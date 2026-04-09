"""언어모델용 컨텍스트 데이터셋 생성 유틸리티.

이 모듈은 토큰 수열을 (입력 컨텍스트, 정답) 쌍으로 변환합니다.

예제:
    토큰 id가 [x0, x1, x2, x3, x4]이고 context_len=3이면:
    - 입력[0]: [x0, x1, x2] -> 정답: x3
    - 입력[1]: [x1, x2, x3] -> 정답: x4

    이를 통해 모델은 "최근 k개 토큰을 보고 다음 토큰을 예측"하는 방식으로 훈련됩니다.

데이터 형태:
    X: shape (N, context_len) - N개의 컨텍스트 (각각 context_len개 토큰)
    y: shape (N,) - N개의 정답 토큰 (각각 1개)
    여기서 N = len(token_ids) - context_len
"""

from __future__ import annotations

import numpy as np


def make_context_dataset(token_ids: np.ndarray, context_len: int) -> tuple[np.ndarray, np.ndarray]:
    """토큰 ID 수열을 (컨텍스트, 정답) 쌍으로 변환합니다.

    슬라이딩 윈도우 방식으로 모든 가능한 (입력, 정답) 쌍을 생성합니다.

    Args:
        token_ids: shape (T,)인 정수 배열 - 토큰 ID 수열
        context_len: 입력 컨텍스트 길이 (>0)

    Returns:
        (X, y) 튜플:
        - X: shape (T - context_len, context_len) - 입력 컨텍스트들
        - y: shape (T - context_len,) - 정답 토큰들 (다음 토큰 ID)

    에러 조건:
        - token_ids가 1D가 아닌 경우
        - context_len <= 0인 경우
        - token_ids가 context_len보다 짧은 경우 (최소 context_len+1개 필요)

    예제:
        >>> token_ids = np.array([1, 2, 3, 4, 5])
        >>> X, y = make_context_dataset(token_ids, 2)
        >>> X
        array([[1, 2],
               [2, 3],
               [3, 4]])
        >>> y
        array([3, 4, 5])
    """
    # 입력 유효성 검사
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be 1D")
    if context_len <= 0:
        raise ValueError("context_len must be > 0")
    if len(token_ids) <= context_len:
        raise ValueError("token_ids too short for context_len")

    # 생성될 샘플 개수: 전체 토큰 수 - 컨텍스트 길이
    # (맨 끝 context_len개 토큰은 정답으로 사용되므로)
    n = len(token_ids) - context_len

    # 출력 배열 할당
    X = np.empty((n, context_len), dtype=np.int64)  # 입력 컨텍스트
    y = np.empty((n,), dtype=np.int64)              # 정답 토큰

    # 슬라이딩 윈도우로 모든 쌍 생성
    for i in range(n):
        # i번째 샘플: 토큰[i:i+context_len]을 입력, 토큰[i+context_len]을 정답으로
        X[i] = token_ids[i : i + context_len]
        y[i] = token_ids[i + context_len]

    return X, y
