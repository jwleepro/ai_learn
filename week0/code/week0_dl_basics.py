"""Week 0: 딥러닝에서 자주 쓰는 최소 계산을 코드로 익히기.

이 파일은 “수학을 외우기”보다 “계산이 어떻게 생겼는지”를 먼저 잡는 용도입니다.

- simple_neuron: y = w*x + b (중학교 일차함수)
- burger_finance: W@x + b (행렬곱)
- relu: max(0, x)
- fit_line_gd: 경사하강법으로 직선(y ≈ w*x + b) 맞추기
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def simple_neuron(x: float, w: float, b: float) -> float:
    return (w * x) + b


def burger_finance(sales: np.ndarray) -> np.ndarray:
    """
    sales: [burgers, fries, cola]
    returns: [revenue, profit]
    """
    sales = np.asarray(sales, dtype=np.float64)
    if sales.shape != (3,):
        raise ValueError("sales must have shape (3,)")

    # W rows: [revenue unit prices], [profit margins]
    W = np.array(
        [
            [5000.0, 2000.0, 1500.0],
            [2000.0, 1000.0, 500.0],
        ],
        dtype=np.float64,
    )
    b = np.array([10000.0, -50000.0], dtype=np.float64)
    return (W @ sales) + b


def relu(x: np.ndarray) -> np.ndarray:
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
    Fit y ~= w*x + b by gradient descent on MSE.
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
