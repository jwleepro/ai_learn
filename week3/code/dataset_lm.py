"""언어모델용 컨텍스트 데이터셋 생성 유틸리티.

이 모듈은 토큰 수열을 (입력 컨텍스트, 정답) 쌍으로 변환하는 슬라이딩 윈도우 방식을 구현합니다.
모델이 "최근 k개 토큰을 보고 다음 토큰을 예측"하도록 훈련하기 위해 사용됩니다.

슬라이딩 윈도우 개념:
- 토큰 id가 [x0, x1, x2, x3, x4]이고 context_len=3이면:
  * 샘플 0: 입력 [x0, x1, x2] -> 정답 x3
  * 샘플 1: 입력 [x1, x2, x3] -> 정답 x4

이 방식으로 길이 T인 수열에서 T-context_len개의 (입력, 정답) 쌍을 생성합니다.
각 원본 토큰(첫 context_len개 제외)이 정확히 한 번씩 정답으로 사용되고,
각 토큰이 컨텍스트에 여러 번 나타나므로 데이터 효율성이 높습니다.

Data shapes:
- Input X: shape (N, context_len) - N개의 컨텍스트
  * N = len(token_ids) - context_len
- Output y: shape (N,) - N개의 정답 토큰
"""

from __future__ import annotations

import numpy as np


def make_context_dataset(token_ids: np.ndarray, context_len: int) -> tuple[np.ndarray, np.ndarray]:
    """토큰 ID 수열을 (컨텍스트, 정답) 쌍으로 변환합니다.

    슬라이딩 윈도우(Sliding Window) 방식으로 모든 가능한 (입력, 정답) 쌍을 생성합니다.
    이는 언어모델 훈련의 표준 방식으로, 모델이 순차적 패턴을 학습하게 합니다.

    이 방식의 장점:
    1. 데이터 효율성: 원본 수열의 거의 모든 부분을 활용 (T-context_len개 샘플)
    2. 의미론적 정확성: 실제 생성 시나리오와 일치
       (생성할 때도 컨텍스트를 보고 다음 토큰 생성)
    3. 문맥 범위 제한: context_len으로 의존성 범위를 제어

    Args:
        token_ids: shape (T,)인 정수 배열 - 원본 토큰 ID 수열
        context_len: 컨텍스트 길이(보고 싶은 과거 토큰 수), >0이어야 함

    Returns:
        (X, y) 튜플:
        - X: shape (T - context_len, context_len) - 입력 컨텍스트들
        - y: shape (T - context_len,) - 정답 토큰들

    Raises:
        ValueError: token_ids가 1D가 아니거나 context_len 유효성 실패 시

    예제:
        >>> token_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        >>> X, y = make_context_dataset(token_ids, context_len=2)
        >>> print(X)
        [[1 2]
         [2 3]
         [3 4]]
        >>> print(y)
        [3 4 5]
    """
    # ===== 입력 유효성 검사 =====
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be 1D")
    if context_len <= 0:
        raise ValueError("context_len must be > 0")
    if len(token_ids) <= context_len:
        raise ValueError("token_ids too short for context_len")

    # ===== 샘플 개수 계산 =====
    # 생성될 샘플 개수: 전체 토큰 수 - 컨텍스트 길이
    # 예: 토큰 5개, context_len=2 -> 5-2=3개 샘플
    n = len(token_ids) - context_len

    # ===== 출력 배열 할당 =====
    X = np.empty((n, context_len), dtype=np.int64)  # 입력 컨텍스트
    y = np.empty((n,), dtype=np.int64)              # 정답 토큰

    # ===== 슬라이딩 윈도우로 샘플 생성 =====
    for i in range(n):
        # i번째 샘플 구성
        X[i] = token_ids[i : i + context_len]        # 입력: i부터 i+context_len-1까지
        y[i] = token_ids[i + context_len]            # 정답: i+context_len번째 토큰
    return X, y
