"""카운트 기반 빅램(2-gram) 유틸리티.

빅램은 “직전 토큰(prev)만 보고 다음 토큰(next)을 예측”합니다.

- counts: (V, V)  where counts[prev, next] = 등장 횟수
- probs:  (V, V)  각 prev 행(row)이 확률분포(합=1)
"""

from __future__ import annotations

import numpy as np


def build_bigram_counts(token_ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """
    토큰 시퀀스에서 빅램(2-gram) 등장 횟수를 세어 (V, V) 행렬 생성

    빅램의 의미:
    - 직전 토큰(prev)에서 다음 토큰(next)로의 전이를 기록
    - counts[i, j] = "토큰 i 다음에 토큰 j가 나온 횟수"

    예:
    - 텍스트: "안녕 안녕 하세요"
    - 토큰: [안, 녕, 안, 녕, 하, 세, 요]
    - 빅램: (안->녕), (녕->안), (안->녕), (녕->하), (하->세), (세->요)
    - counts[안_id, 녕_id] += 2 (두 번 나타남)

    Args:
        token_ids: 토큰 시퀀스 (1D 배열, 길이 >= 2)
        vocab_size: 어휘 크기 (V), 행렬은 (V, V) 크기

    Returns:
        counts: shape (V, V) - counts[i, j] = 토큰i->토큰j 전이 횟수
    """
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be 1D")
    if len(token_ids) < 2:
        raise ValueError("token_ids must contain at least 2 tokens")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")

    # 전체 토큰에서 마지막을 제외: [t0, t1, t2, ..., t(n-1)]
    prev_ids = token_ids[:-1]
    # 전체 토큰에서 처음을 제외: [t1, t2, t3, ..., tn]
    next_ids = token_ids[1:]
    # 결과: (prev_ids[i], next_ids[i])는 연속된 토큰 쌍

    counts = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    # np.add.at: (prev_ids[i], next_ids[i]) 위치에 1씩 누적
    np.add.at(counts, (prev_ids, next_ids), 1)
    return counts


def counts_to_probs(counts: np.ndarray, *, smoothing: float = 0.0) -> np.ndarray:
    """
    빅램 등장 횟수를 조건부 확률분포로 변환

    확률의 의미:
    - probs[i, :] = "토큰 i가 주어졌을 때 다음 토큰의 확률분포"
    - probs[i, j] = P(next=j | prev=i)
    - 각 행의 합 = 1.0 (확률의 기본 성질)

    스무딩(Laplace Smoothing) 개념:
    - 문제: 등장하지 않은 빅램은 확률 0 -> 생성 중 항상 불가능
    - 해결: 모든 빅램에 작은 상수를 더해주기
    - 효과: 희귀한 전이도 가능해짐 (하지만 확률은 낮음)

    예:
    - counts[i] = [3, 0, 0, 1] (토큰 i 다음: 토큰0 3회, 토큰3 1회)
    - smoothing=0: probs[i] = [0.75, 0, 0, 0.25]
    - smoothing=1: counts+1 = [4, 1, 1, 2], probs[i] = [0.5, 0.125, 0.125, 0.25]
      -> 본 적 없는 전이(토큰1, 2)도 작은 확률로 가능해짐

    Args:
        counts: shape (V, V) - 빅램 등장 횟수 행렬
        smoothing: 스무딩 상수 (0 = 스무딩 없음, 1 = Laplace smoothing)

    Returns:
        probs: shape (V, V) - 각 행이 확률분포 (행의 합 = 1.0)
    """
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("counts must be a square 2D matrix")
    if smoothing < 0:
        raise ValueError("smoothing must be >= 0")

    counts_f = counts.astype(np.float64, copy=False)
    # 스무딩: 모든 카운트에 smoothing 값 더하기
    if smoothing != 0.0:
        counts_f = counts_f + smoothing

    # 각 행(이전 토큰)의 합 계산
    # axis=1: 행별(각 이전 토큰별) 합계
    # keepdims=True: shape 유지하여 broadcasting 용이 -> (V, 1)
    row_sums = counts_f.sum(axis=1, keepdims=True)

    # 보정: 극히 드문 경우(tiny data), 어떤 토큰 이후 아무것도 없을 수 있음
    # 그 경우 균등분포(uniform)로 설정
    zero_rows = row_sums.squeeze(axis=1) == 0
    if np.any(zero_rows):
        counts_f = counts_f.copy()
        counts_f[zero_rows, :] = 1.0  # 모두 1로 설정 -> 균등분포
        row_sums = counts_f.sum(axis=1, keepdims=True)

    # 확률 계산: 각 행을 그 행의 합으로 나누면 합=1인 확률분포
    return counts_f / row_sums


def sample_next_id(
    prev_id: int,
    probs: np.ndarray,
    rng: np.random.Generator,
    *,
    temperature: float = 1.0,
) -> int:
    """
    조건부 확률분포에서 다음 토큰을 표본추출(sampling)

    온도(Temperature) 개념:
    - 확률분포의 "날카로움" vs "부드러움"을 조절
    - 수학: softmax 계열 함수에서 logit을 온도로 나누어 조절

    온도 효과:
    - T < 1 (e.g., 0.7): 확률 "집중" -> 높은 확률이 더 높아짐
      예: [0.7, 0.2, 0.1] -> [0.82, 0.13, 0.05] (더 확정적)
    - T = 1 (기본값): 원래 확률분포 유지 (변화 없음)
    - T > 1 (e.g., 1.3): 확률 "완화" -> 모든 확률이 균등해짐
      예: [0.7, 0.2, 0.1] -> [0.60, 0.23, 0.17] (더 무작위)

    수학적 구현 (softmax temperature):
    - 원래: p = softmax(logits)
    - 온도 적용: p = softmax(logits / T)
    - 여기서는 역으로: p_new = p_old^(1/T)
      (log 공간에서 logit/T = log(p)/T = (1/T)*log(p))

    Args:
        prev_id: 이전 토큰 인덱스
        probs: shape (V, V) - 빅램 확률 행렬
        rng: numpy 난수 생성기
        temperature: 온도값 (>0, 1 = 기본값)

    Returns:
        다음 토큰의 인덱스 (확률에 따라 표본추출됨)
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    row = probs[prev_id]  # 현재 토큰 다음의 확률분포

    # 온도 조절: 1.0이 아니면 확률분포 변환
    if temperature != 1.0:
        # p_new = p_old^(1/T): 각 확률을 (1/T) 거듭제곱
        row = row ** (1.0 / temperature)
        # 다시 정규화: 변환된 확률의 합을 1로 만들기
        row = row / row.sum()

    # 조정된 확률에 따라 표본추출: p가 높을수록 선택될 확률이 높음
    return int(rng.choice(len(row), p=row))
