"""softmax / log_softmax (numpy).

점수(logits)를 확률로 바꾸는 함수입니다.
큰 수를 먼저 빼서(max-shift) 계산이 터지지 않게 합니다.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """
    Softmax 함수: 점수(logits)를 확률분포로 변환

    수학:
    - softmax(logits)[i] = exp(logits[i]) / sum(exp(logits))
    - 각 값은 (0, 1) 범위, 합 = 1.0

    수치 안정성 트릭 (max-shift):
    - 문제: logits이 크면 exp(logits)가 무한대(overflow) 또는 0(underflow)
    - 해결: max를 먼저 빼기: exp(logits - max) 사용
    - 효과: exp() 입력이 0 근처로 줄어들어 안정적 계산
    - 수학적으로: softmax(logits) = softmax(logits - c) (any constant c)

    예:
    - logits = [1000, 1001, 999]
    - 직접: exp([1000, 1001, 999])는 overflow
    - shift: max=1001, logits - 1001 = [-1, 0, -2]
    - exp([-1, 0, -2]) = [0.368, 1.0, 0.135] (안전)
    - 합 = 1.503 -> 정규화 -> [0.245, 0.665, 0.090]

    Args:
        logits: 입력 배열 (어떤 크기든 상관없음)
        axis: 소프트맥스를 계산할 축 (기본값: 마지막 축)

    Returns:
        확률분포 (합 = 1.0)
    """
    if logits.size == 0:
        raise ValueError("logits must not be empty")
    # max-shift: 수치 안정성 확보
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    # 정규화: 각 샘플의 확률 합 = 1
    return exp / exp.sum(axis=axis, keepdims=True)


def log_softmax(logits: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """
    Log-Softmax 함수: 확률의 로그값을 직접 계산

    정의: log_softmax(logits)[i] = logits[i] - logsumexp(logits)

    왜 직접 계산하는가?
    1. 손실함수 계산이 안정적:
       - cross-entropy = -log_softmax(logits)[target]
       - softmax() 후 log()를 하면 수치 오차 누적
       - log_softmax()를 직접 계산하면 더 정확

    2. 메모리 절약:
       - softmax는 모든 확률을 저장 (메모리 O(exp))
       - log_softmax는 로그값만 저장 (메모리 줄어듦)

    수학:
    - log(softmax(x)) = log(exp(x) / sum(exp(x)))
                      = log(exp(x)) - log(sum(exp(x)))
                      = x - logsumexp(x)

    max-shift로 수치 안정성 확보:
    - logsumexp(x) = log(sum(exp(x - max(x)))) + max(x)

    Args:
        logits: 입력 배열
        axis: 로그-소프트맥스를 계산할 축

    Returns:
        로그-확률 (합 = log(1) = 0이 되도록 정규화)
    """
    if logits.size == 0:
        raise ValueError("logits must not be empty")
    # max-shift로 수치 안정성 확보
    shifted = logits - logits.max(axis=axis, keepdims=True)
    # logsumexp 계산: log(sum(exp(shifted)))
    logsumexp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    # 로그 확률분포
    return shifted - logsumexp
