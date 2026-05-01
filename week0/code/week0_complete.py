"""Week 0: 딥러닝에서 자주 쓰는 최소 계산을 코드로 익히기.

수학 공식을 외우기 전에, "계산이 코드로는 어떻게 생겼나" 부터 본다.

내용:
- simple_neuron:  y = w*x + b   (중학교 일차함수)
- burger_finance: y = W @ x + b (행렬곱)
- relu:           max(0, x)     (음수는 0 으로)
- fit_line_gd:    경사하강법으로 직선 y ≈ w*x + b 를 데이터에 맞추기
"""

import numpy as np


# ============================================================================
# Softmax / log_softmax (다른 주차 파일에서 사용)
# ============================================================================

def softmax(logits, axis=-1):
    """점수(logits)를 확률로 바꾼다.

    수식: softmax(x_i) = exp(x_i) / sum(exp(x_j))
    가장 큰 값을 먼저 빼는 max-shift 는 exp 가 너무 커지는 걸 막는 트릭.
    """
    if logits.size == 0:
        raise ValueError("logits must not be empty")

    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def log_softmax(logits, axis=-1):
    """log(softmax(x)) 를 안정적으로 계산."""
    if logits.size == 0:
        raise ValueError("logits must not be empty")

    shifted = logits - logits.max(axis=axis, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    return shifted - logsumexp


# ============================================================================
# 1. 뉴런 한 개: y = w*x + b
# ============================================================================

def simple_neuron(x, w, b):
    """뉴런의 기본 연산: y = w*x + b

    매개변수:
        x: 입력값
        w: 가중치 (weight) - 입력의 영향력
        b: 편향 (bias) - 기본값

    예) x=3, w=10, b=20 -> y = 10*3 + 20 = 50
    """
    return (w * x) + b


# ============================================================================
# 2. 행렬곱: 햄버거 가게 예제
# ============================================================================

def burger_finance(sales):
    """행렬곱을 이용한 다중 선형 변환: y = W @ sales + b

    햄버거 가게 예제:
    - 버거, 감자튀김, 콜라 3가지 상품의 판매량이 주어지면
    - 총 수익과 이윤을 한 번에 계산한다.

    매개변수:
        sales: shape (3,) - [버거수, 감자튀김수, 콜라수]

    반환:
        shape (2,) - [총수익, 이윤]

    수식:
        W (2x3) @ sales (3,) + b (2,) = result (2,)
    """
    sales = np.asarray(sales, dtype=np.float64)
    if sales.shape != (3,):
        raise ValueError("sales must have shape (3,)")

    # W: 2x3 행렬. 각 행이 한 가지 계산(수익 또는 이윤).
    W = np.array(
        [
            [5000.0, 2000.0, 1500.0],  # 각 상품의 단가
            [2000.0, 1000.0, 500.0],   # 각 상품의 이윤 마진
        ],
        dtype=np.float64,
    )
    b = np.array([10000.0, -50000.0], dtype=np.float64)  # 각 항목의 상수항

    return (W @ sales) + b


# ============================================================================
# 3. ReLU: 음수는 0 으로
# ============================================================================

def relu(x):
    """ReLU(x) = max(0, x). 신경망에 비선형성을 추가한다.

    예) relu([-2, -1, 0, 1, 2]) -> [0, 0, 0, 1, 2]

    특징:
    - 계산 간단/빠름
    - 미분도 간단 (양수면 1, 음수면 0)
    - 너무 많은 음수 입력이 들어오면 뉴런이 "죽을" 수 있다 (Dying ReLU).
    """
    x = np.asarray(x, dtype=np.float64)
    return np.maximum(x, 0.0)


# ============================================================================
# 4. 경사하강법: 직선 y = w*x + b 를 데이터에 맞추기
# ============================================================================

class LinearGDResult:
    """fit_line_gd 의 결과: 최적 (w, b) 와 각 스텝의 손실값."""

    def __init__(self, w, b, losses):
        self.w = w
        self.b = b
        self.losses = losses


def fit_line_gd(x, y, lr=0.1, steps=200, w0=0.0, b0=0.0):
    """경사하강법(Gradient Descent)으로 직선 y = w*x + b 를 데이터에 맞춘다.

    원리:
    - 손실: L(w, b) = mean((y_pred - y_true)^2)
            여기서 y_pred = w*x + b
    - 그래디언트:
        dL/dw = mean(2 * err * x)   ← err = y_pred - y_true
        dL/db = mean(2 * err)
    - 업데이트:
        w := w - lr * dw
        b := b - lr * db

    매개변수:
        x, y: 1D numpy 배열, 길이 같음
        lr:   학습률 (한 번에 얼마나 크게 이동할지)
        steps: 반복 횟수
        w0, b0: 초기값
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
    losses = []
    n = float(len(x))

    for _ in range(int(steps)):
        # 예측값과 오차
        y_pred = (w * x) + b
        err = y_pred - y

        # 현재 손실 (MSE) 기록
        loss = float(np.mean(err ** 2))
        losses.append(loss)

        # 그래디언트
        dw = float((2.0 / n) * np.sum(err * x))
        db = float((2.0 / n) * np.sum(err))

        # 업데이트
        w = w - lr * dw
        b = b - lr * db

    return LinearGDResult(w, b, losses)


# ============================================================================
# 5. 데모 main()
# ============================================================================

def main():
    """Week 0 데모: 위 함수들을 차례로 실행해 결과를 확인한다."""
    print("=" * 60)
    print("Week 0: 딥러닝의 기본 수학 연산")
    print("=" * 60)

    # 1) 뉴런 한 개
    print("\n[1] 뉴런 (Neuron)")
    y = simple_neuron(3, 10, 20)
    print(f"    simple_neuron(x=3, w=10, b=20)")
    print(f"    -> y = 10*3 + 20 = {y:.0f}")

    # 2) 행렬곱: 햄버거 가게
    print("\n[2] 행렬곱 (Matrix Multiplication)")
    sales = np.array([100, 80, 120], dtype=np.float64)
    result = burger_finance(sales)
    revenue = result[0]
    profit = result[1]
    print(f"    판매량: 버거={sales[0]:.0f}, 튀김={sales[1]:.0f}, 콜라={sales[2]:.0f}")
    print(f"    -> 총수익: {revenue:,.0f}원")
    print(f"    -> 이윤:   {profit:,.0f}원")

    # 3) 경사하강법
    print("\n[3] 경사하강법 (Gradient Descent)")
    x = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    y = np.array([1, 3, 5, 7, 9], dtype=np.float64)  # 실제 정답: y = 2x + 1
    print(f"    x: {x.tolist()}")
    print(f"    y: {y.tolist()} (실제는 y = 2x + 1)")

    result = fit_line_gd(x, y, lr=0.1, steps=200, w0=0.0, b0=0.0)
    last_loss = result.losses[len(result.losses) - 1]
    first_loss = result.losses[0]
    print(f"    학습 결과 (200 스텝 후):")
    print(f"      - 최적 w: {result.w:.3f} (목표: 2.0)")
    print(f"      - 최적 b: {result.b:.3f} (목표: 1.0)")
    print(f"      - 초기 손실: {first_loss:.4f}")
    print(f"      - 최종 손실: {last_loss:.4f}")

    # 4) ReLU
    print("\n[4] ReLU 활성화 함수")
    r = relu(np.array([-2, -1, 0, 1, 2], dtype=np.float64))
    print(f"    relu([-2, -1, 0, 1, 2]) -> {r.tolist()}")


if __name__ == "__main__":
    main()
