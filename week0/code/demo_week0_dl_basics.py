"""Week 0 데모 스크립트.

`week0_dl_basics.py`에 있는 함수들을 실제로 실행해보고,
출력이 기대한 대로 나오는지 빠르게 확인합니다.
"""

from __future__ import annotations

import numpy as np

from week0_dl_basics import burger_finance, fit_line_gd, relu, simple_neuron


def main() -> None:
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
