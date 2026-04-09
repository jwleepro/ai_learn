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

import _path_setup  # noqa: F401  (code/ 하위 폴더 sys.path 등록)
from softmax import softmax


def causal_mask(scores: np.ndarray) -> np.ndarray:
    """
    Causal Masking: 미래 토큰 정보를 보이지 않게 마스킹

    배경:
    - 언어모델은 좌에서 우로(left-to-right) 생성: 이전 토큰들만 봐서 다음 토큰 예측
    - Attention은 모든 토큰 쌍을 고려하므로, "미래 토큰을 보는" 문제 발생
    - 해결: 미래 위치의 점수를 매우 낮은 값(-1e9)으로 설정

    구체적으로:
    - scores[i, j]는 위치 i에서 위치 j를 보는 가중치 점수
    - j > i인 경우 (미래): -1e9로 설정
    - softmax 후 exp(-1e9) ≈ 0, 즉 확률 ≈ 0 → 미래 토큰 무시

    Args:
        scores: shape (T, T) - 어텐션 점수 (T = 시퀀스 길이)

    Returns:
        masked: shape (T, T) - 미래 부분이 -1e9로 마스킹된 scores

    예시:
        T=3인 경우
        scores = [[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]

        masked = [[1, -1e9, -1e9],    (위치 0은 자신만 볼 수 있음)
                  [4, 5, -1e9],       (위치 1은 0, 1만 볼 수 있음)
                  [7, 8, 9]]          (위치 2는 모두 볼 수 있음)
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be (T, T)")
    T = scores.shape[0]
    masked = scores.copy()
    # np.triu: 상삼각 행렬 생성, k=1이므로 대각선 위쪽만 True
    # 즉, 미래 위치(j > i)에 해당하는 부분
    upper = np.triu(np.ones((T, T), dtype=bool), k=1)
    # 미래 위치의 점수를 매우 낮은 값으로 설정
    # softmax([-1e9]) ≈ 0이므로 사실상 무시
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
    Self-Attention 계산: 각 토큰이 다른 토큰들을 얼마나 "주목"할지 결정합니다

    Self-Attention의 3가지 구성요소:
    1. Query (Q): "이 위치에서 뭘 찾고 있는가?"
    2. Key (K): "이 위치가 제공할 수 있는 것은?"
    3. Value (V): "실제 정보"

    계산 단계:
    1. X를 Q, K, V로 투영 (각각 다른 측면을 학습)
    2. Q와 K의 내적으로 유사도 점수 계산 (scaled dot-product)
    3. 점수를 softmax로 확률로 변환 → 어텐션 가중치
    4. 가중치로 V를 조합 → 최종 출력

    수학:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_h)) @ V

    각 위치는 자신을 포함한 모든 위치의 V를 가중합으로 봅니다.
    - 높은 가중치 → 그 위치에 주목
    - 낮은 가중치 → 그 위치 무시

    Args:
        X: shape (T, D) - 입력 시퀀스 (T = 길이, D = 특성 차원)
        Wq, Wk, Wv: shape (D, Dh) - 학습 가능한 투영 행렬 (Dh = 어텐션 차원)
        causal: True면 인과적 마스킹 적용 (미래 토큰 무시)

    Returns:
        weights: shape (T, T) - 어텐션 가중치 (각 행이 해당 위치의 가중치 분포)
        out: shape (T, Dh) - 최종 어텐션 출력 (각 위치별로 주목한 정보의 조합)

    정보 흐름:
        X (T, D)
        -> Q, K, V: (T, Dh)  [투영]
        -> scores: (T, T)    [내적으로 유사도]
        -> weights: (T, T)   [softmax로 정규화]
        -> out: (T, Dh)      [가중합으로 조합]
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (T, D)")
    if Wq.shape[0] != X.shape[1] or Wk.shape[0] != X.shape[1] or Wv.shape[0] != X.shape[1]:
        raise ValueError("Wq/Wk/Wv first dim must match X feature dim")

    # 1단계: 투영
    # X의 각 행(토큰)을 Q, K, V 공간으로 변환
    Q = X @ Wq  # (T, Dh) - Query: "뭘 찾을 것인가"
    K = X @ Wk  # (T, Dh) - Key: "뭘 제공할 수 있는가"
    V = X @ Wv  # (T, Dh) - Value: "실제 정보"

    # 2단계: 어텐션 점수 계산 (scaled dot-product)
    Dh = Q.shape[1]  # 어텐션 차원
    # Q @ K^T: 모든 쌍의 유사도 계산
    # shape (T, Dh) @ (Dh, T) = (T, T)
    # [i, j] = Q[i] · K[j]: 위치 i의 질문과 위치 j의 핵심이 얼마나 매칭되는가
    scores = (Q @ K.T) / np.sqrt(float(Dh))  # (T, T)
    # sqrt(Dh)로 정규화: 차원이 커질수록 내적이 커지는 경향을 보정
    # 이를 통해 softmax가 극단적으로 뾰족해지는 것을 방지

    # 3단계: 인과적 마스킹 (선택사항)
    # 언어모델에서는 미래 정보를 보면 안 되므로, 미래 위치를 마스킹
    if causal:
        scores = causal_mask(scores)

    # 4단계: 확률 정규화
    # softmax: 각 위치의 점수를 확률분포로 변환
    # weights[i, :] = softmax(scores[i, :])
    # 의미: 위치 i에서 각 위치를 얼마나 "주목"할지의 확률
    weights = softmax(scores, axis=1)  # (T, T), 각 행의 합 = 1

    # 5단계: 가중합으로 최종 출력 계산
    # weights @ V: 각 위치마다 모든 V를 가중합
    # out[i, :] = sum(weights[i, j] * V[j, :])
    # 의미: 위치 i의 최종 출력은 모든 V의 가중합 (가중치는 어텐션 확률)
    out = weights @ V  # (T, T) @ (T, Dh) = (T, Dh)

    return weights, out
