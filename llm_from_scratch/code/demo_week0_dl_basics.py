"""Week 0 데모 스크립트.

`week0_dl_basics.py`에 있는 함수들을 실제로 실행해보고,
출력이 기대한 대로 나오는지 빠르게 확인합니다.
"""

from __future__ import annotations

import numpy as np

from week0_dl_basics import burger_finance, fit_line_gd, relu, simple_neuron


def main() -> None:
    # 1) neuron
    y = simple_neuron(3, 10, 20)
    print(f"simple_neuron: x=3 w=10 b=20 -> y={y:.0f}")

    # 2) burger finance (matrix multiplication)
    revenue, profit = burger_finance(np.array([100, 80, 120], dtype=np.float64))
    print(f"burger_finance: revenue={revenue:.0f} profit={profit:.0f}")

    # 3) gradient descent line fit
    x = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    y = np.array([1, 3, 5, 7, 9], dtype=np.float64)  # y = 2x + 1
    res = fit_line_gd(x, y, lr=0.1, steps=200, w0=0.0, b0=0.0)
    print(f"fit_line_gd: w={res.w:.3f} b={res.b:.3f} loss0={res.losses[0]:.4f} lossN={res.losses[-1]:.4f}")

    # # 4) relu demo
    # r = relu(np.array([-2, -1, 0, 1, 2], dtype=np.float64))
    # print(f"relu([-2,-1,0,1,2]) -> {r.tolist()}")


if __name__ == "__main__":
    main()
