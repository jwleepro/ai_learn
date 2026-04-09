"""Week 0: 딥러닝에서 자주 쓰는 최소 계산을 코드로 익히기.

이 파일은 "수학을 외우기"보다 "계산이 어떻게 생겼는지"를 먼저 잡는 용도입니다.

내용:
- simple_neuron: y = w*x + b (중학교 일차함수)
- burger_finance: W@x + b (행렬곱)
- relu: max(0, x)
- fit_line_gd: 경사하강법으로 직선(y ≈ w*x + b) 맞추기

이 통합 파일은 week0_dl_basics.py와 demo_week0_dl_basics.py의
모든 코드를 합친 것입니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ============================================================================
# Inline: softmax functions (from softmax.py)
# ============================================================================

def softmax(logits: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """점수(logits)를 확률로 바꾸는 함수입니다.

    큰 수를 먼저 빼서(max-shift) 계산이 터지지 않게 합니다.
    """
    if logits.size == 0:
        raise ValueError("logits must not be empty")
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def log_softmax(logits: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """로그 소프트맥스 함수입니다."""
    if logits.size == 0:
        raise ValueError("logits must not be empty")
    shifted = logits - logits.max(axis=axis, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    return shifted - logsumexp


# ============================================================================
# Section 1: week0_dl_basics.py - Core functions
# ============================================================================

def simple_neuron(x: float, w: float, b: float) -> float:
    """
    뉴런의 기본 연산: y = w*x + b

    Args:
        x: 입력값
        w: 가중치(weight) - x의 영향력을 조절
        b: 편향(bias) - 기본값 조절

    Returns:
        선형 변환의 결과값

    예: x=3, w=10, b=20 -> y = 10*3 + 20 = 50
    """
    return (w * x) + b


def burger_finance(sales: np.ndarray) -> np.ndarray:
    """
    행렬곱을 이용한 다중 선형 변환: y = W@x + b

    햄버거 가게 예제:
    - 버거, 감자튀김, 콜라 3가지 상품의 판매량이 주어지면
    - 총 수익과 이윤을 계산한다

    Args:
        sales: shape (3,) - [버거수, 감자튀김수, 콜라수]

    Returns:
        shape (2,) - [총수익, 이윤]

    수학:
    - W는 2x3 행렬: 각 행이 계산 규칙(수익단가, 이윤마진)
    - sales는 3x1 벡터: 판매량
    - W @ sales = (2x3) @ (3x1) = (2x1) - 결과는 2개 값
    - b는 각 항목에 더할 상수항

    예:
    - 버거 단가: 5000원, 감자튀김: 2000원, 콜라: 1500원
    - 버거 이윤: 2000원, 감자튀김: 1000원, 콜라: 500원
    - sales = [100, 80, 120]이면:
      - 총수익 = 5000*100 + 2000*80 + 1500*120 + 10000 = 1,108,000원
      - 이윤 = 2000*100 + 1000*80 + 500*120 - 50000 = 290,000원
    """
    sales = np.asarray(sales, dtype=np.float64)
    if sales.shape != (3,):
        raise ValueError("sales must have shape (3,)")

    # W: 2x3 행렬 - 각 행이 한 가지 계산(수익 또는 이윤)
    # 행렬곱 W @ sales는 각 행마다 가중합(weighted sum)을 계산한다
    W = np.array(
        [
            [5000.0, 2000.0, 1500.0],  # 각 상품의 단가
            [2000.0, 1000.0, 500.0],   # 각 상품의 이윤 마진
        ],
        dtype=np.float64,
    )
    b = np.array([10000.0, -50000.0], dtype=np.float64)  # 각 항목의 기본값
    return (W @ sales) + b


def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU (Rectified Linear Unit): 비선형 활성화 함수

    정의: ReLU(x) = max(0, x)
    - 양수는 그대로, 음수는 0으로 변환
    - 신경망에 비선형성(nonlinearity)을 추가한다
    - 선형 변환만으로는 복잡한 패턴을 배울 수 없으므로 필수

    특징:
    - 계산이 간단하고 빠름
    - 미분도 간단: x > 0이면 1, x < 0이면 0 (경사하강법 용이)
    - 너무 많은 음수 입력이 있으면 뉴런이 "죽을" 수 있음 (Dying ReLU)

    예:
    - relu([-2, -1, 0, 1, 2]) -> [0, 0, 0, 1, 2]
    """
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(x, 0.0)


@dataclass(frozen=True)
class LinearGDResult:
    w: float
    b: float
    losses: list[float]


def fit_line_gd(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.1,
    steps: int = 200,
    w0: float = 0.0,
    b0: float = 0.0,
) -> LinearGDResult:
    """
    경사하강법(Gradient Descent)으로 직선 y = w*x + b를 데이터에 맞춘다

    핵심 원리:
    1. 손실함수 정의: MSE(w,b) = mean((y_pred - y_true)^2)
    2. 손실을 w, b로 미분하여 그래디언트(기울기) 계산
    3. 반복: w와 b를 그래디언트 반대 방향으로 조정하여 손실 감소

    수학 설명:
    - 예측값: y_pred = w*x + b
    - 오차: err = y_pred - y_true = (w*x + b) - y
    - 손실: L = mean(err^2)
    - w의 그래디언트: dL/dw = mean(2 * err * x) = (2/n) * sum(err * x)
      (chain rule: 제곱의 미분이 2*err, 내부의 (w*x + b)를 w로 미분하면 x)
    - b의 그래디언트: dL/db = mean(2 * err) = (2/n) * sum(err)
      (내부의 (w*x + b)를 b로 미분하면 1)
    - 파라미터 업데이트: w = w - lr * dw, b = b - lr * db
      (음수 그래디언트 방향으로 이동하여 손실 감소)

    Args:
        x: 입력 데이터 (1D 배열)
        y: 목표값 (1D 배열)
        lr: 학습율 (learning rate) - 한 번에 얼마나 크게 이동할지
        steps: 반복 횟수
        w0: 가중치 초기값
        b0: 편향 초기값

    Returns:
        LinearGDResult: 최적 w, b와 각 스텝의 손실값 리스트
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be 1D with same length")
    if len(x) == 0:
        raise ValueError("x and y must not be empty")
    if lr <= 0:
        raise ValueError("lr must be > 0")
    if steps <= 0:
        raise ValueError("steps must be > 0")

    w = float(w0)
    b = float(b0)
    losses: list[float] = []
    n = float(len(x))

    for _ in range(int(steps)):
        y_pred = (w * x) + b
        err = y_pred - y
        loss = float(np.mean(err**2))
        losses.append(loss)

        # 여기서 x, y는 이미 주어진 데이터라 고정이고, 학습 중에 바뀌는 건 w, b뿐이다.
        # 그래서 현재 손실은 사실상 MSE(w, b)로 볼 수 있다.
        # err = (w * x) + b - y, MSE = mean(err**2) = mean(((w * x) + b - y)**2).
        # 이를 w로 미분하면 바깥 제곱에서 2 * err가, 안쪽 (w * x) + b - y 에서는 x가 남는다.
        # 그래서 한 샘플 기준으로는 2 * err * x, 여러 샘플 평균을 내면 아래 dw 식이 된다.
        # b는 안쪽 식을 미분하면 1만 남으므로, 여러 샘플 평균을 내면 아래 db 식이 된다.
        dw = float((2.0 / n) * np.sum(err * x))
        db = float((2.0 / n) * np.sum(err))
        w -= lr * dw
        b -= lr * db

    return LinearGDResult(w=w, b=b, losses=losses)


# ============================================================================
# Section 2: demo_week0_dl_basics.py - Demo and main()
# ============================================================================

def main() -> None:
    """Week 0 데모 메인 함수.

    week0_dl_basics.py에 있는 함수들을 실제로 실행해보고,
    출력이 기대한 대로 나오는지 빠르게 확인합니다.
    """
    print("=" * 60)
    print("Week 0: 딥러닝의 기본 수학 연산")
    print("=" * 60)

    # 1) 뉴런의 기본 연산: y = w*x + b
    # 가장 단순한 선형 변환. 신경망의 기본 단위다.
    print("\n[1] 뉴런 (Neuron)")
    y = simple_neuron(3, 10, 20)
    print(f"    simple_neuron(x=3, w=10, b=20)")
    print(f"    -> y = 10*3 + 20 = {y:.0f}")
    print(f"    (w는 입력의 영향력, b는 기본값)")

    # 2) 행렬곱: 여러 입력으로 여러 출력을 한 번에 계산
    # 신경망 층(layer)의 기본 연산이다.
    print("\n[2] 행렬곱 (Matrix Multiplication)")
    sales = np.array([100, 80, 120], dtype=np.float64)
    revenue, profit = burger_finance(sales)
    print(f"    판매량: 버거={sales[0]:.0f}, 튀김={sales[1]:.0f}, 콜라={sales[2]:.0f}")
    print(f"    행렬곱 결과:")
    print(f"      - 총수익: {revenue:,.0f}원")
    print(f"      - 이윤: {profit:,.0f}원")
    print(f"    (2x3 행렬 @ 3x1 벡터 = 2x1 결과)")

    # 3) 경사하강법: 가장 중요한 학습 알고리즘
    # 손실함수를 최소화하기 위해 파라미터를 반복적으로 조정한다.
    print("\n[3] 경사하강법 (Gradient Descent)")
    x = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    y = np.array([1, 3, 5, 7, 9], dtype=np.float64)  # 실제 데이터: y = 2x + 1
    print(f"    목표: 주어진 데이터에 맞는 직선 y = w*x + b 찾기")
    print(f"    x: {x.tolist()}")
    print(f"    y: {y.tolist()} (실제는 y = 2x + 1)")

    # 초기값에서 경사하강법 실행
    res = fit_line_gd(x, y, lr=0.1, steps=200, w0=0.0, b0=0.0)
    print(f"    학습 결과 (200 스텝 후):")
    print(f"      - 최적 가중치 w: {res.w:.3f} (목표: 2.0)")
    print(f"      - 최적 편향 b: {res.b:.3f} (목표: 1.0)")
    print(f"      - 초기 손실: {res.losses[0]:.4f}")
    print(f"      - 최종 손실: {res.losses[-1]:.4f}")
    print(f"    손실이 감소했으므로 학습이 성공적으로 진행되었다!")

    # 4) ReLU: 비선형성을 추가하는 활성화 함수
    print("\n[4] ReLU 활성화 함수 (선택사항)")
    print("    r = relu([-2, -1, 0, 1, 2])")
    r = relu(np.array([-2, -1, 0, 1, 2], dtype=np.float64))
    print(f"    -> {r.tolist()}")
    print(f"    (음수는 0으로, 양수는 그대로 유지)")


if __name__ == "__main__":
    main()
