"""Week 2 — 퍼셉트론과 경사하강법.

이 주에 배우는 것
- "학습" 이 무엇인지: 손실(loss)을 줄이도록 가중치를 조금씩 옮기는 것.
- 가장 단순한 모델 y = w*x + b 로 직선 회귀(regression) 학습.
- 경사하강법(Gradient Descent) 한 줄 요약:
      w ← w - learning_rate * (손실의 w에 대한 기울기)

왜 이게 필요한가
- LLM 도 결국 같은 일을 한다. 가중치가 수억 개로 늘어났을 뿐.
- "기울기 방향의 반대로 조금씩" 이라는 감각을 한 변수로 먼저 익힌다.

실행:  python week2.py
"""

import numpy as np


# ============================================================
# 1. 문제 설정 — 직선 하나 맞히기
# ============================================================
# 어떤 데이터 (x, y) 가 있다. 우리는 진짜 정답이 y = 2x + 1 임을 알지만
# 모델은 그걸 모른다. 무작위 w, b 에서 시작해서 점점 정답에 가까워져야 한다.
#
# 데이터 생성: 노이즈를 살짝 섞어서 현실적으로 만든다.

print("=" * 60)
print("[1] 데이터 만들기 — 진짜 직선은 y = 2x + 1")
print("=" * 60)

np.random.seed(0)

true_w = 2.0
true_b = 1.0

# x 는 0 ~ 5 사이 30개. (Java 의 double[] 30개 라고 보면 됨)
x_data = np.linspace(0.0, 5.0, 30)
# 약간의 노이즈를 더해서 흩뿌린다.
noise = np.random.randn(30) * 0.5
y_data = true_w * x_data + true_b + noise

print("x_data 처음 5개:", x_data[:5])
print("y_data 처음 5개:", y_data[:5])


# ============================================================
# 2. 손실(loss) — "얼마나 틀렸는지" 를 숫자 하나로
# ============================================================
# 예측: y_pred = w * x + b
# 진짜: y_true
# 평균 제곱 오차 (Mean Squared Error, MSE):
#       loss = (1/N) * sum( (y_pred - y_true)^2 )
#
# 왜 제곱?  부호가 사라져서 절댓값처럼 동작하고, 미분이 깔끔해진다.

def mse_loss(w, b, x, y):
    y_pred = w * x + b
    diff = y_pred - y
    return np.mean(diff * diff)


# ============================================================
# 3. 기울기(gradient) — "w 를 어느 쪽으로 움직여야 손실이 줄어드나"
# ============================================================
# loss = (1/N) * sum( (w*x + b - y)^2 )
#
# w 에 대한 미분 (체인룰):
#   d(loss)/dw = (2/N) * sum( (w*x + b - y) * x )
#
# b 에 대한 미분:
#   d(loss)/db = (2/N) * sum( (w*x + b - y) )
#
# 이 두 줄이 이 주의 핵심이다. 이걸 직접 코드로 옮긴다.

def gradients(w, b, x, y):
    n = len(x)
    y_pred = w * x + b
    diff = y_pred - y                 # (N,)
    grad_w = (2.0 / n) * np.sum(diff * x)
    grad_b = (2.0 / n) * np.sum(diff)
    return grad_w, grad_b


# ============================================================
# 4. 학습 루프 — 한 스텝 = (기울기 계산 → 가중치 갱신)
# ============================================================
# 한 스텝:
#   w ← w - learning_rate * grad_w
#   b ← b - learning_rate * grad_b
#
# learning_rate (학습률) 가 너무 크면 발산, 너무 작으면 너무 느리게 학습된다.
# 적당한 값을 찾는 게 항상 중요하다.

print()
print("=" * 60)
print("[4] 학습 시작")
print("=" * 60)

w = 0.0   # 무작위(또는 0)에서 시작
b = 0.0
learning_rate = 0.05
num_steps = 200

print("시작 w=%.3f, b=%.3f, loss=%.4f" % (w, b, mse_loss(w, b, x_data, y_data)))

for step in range(num_steps):
    grad_w, grad_b = gradients(w, b, x_data, y_data)
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

    # 매 50 스텝마다 진행상황 출력
    if (step + 1) % 50 == 0:
        loss_now = mse_loss(w, b, x_data, y_data)
        print("step %3d  w=%.3f  b=%.3f  loss=%.4f" % (step + 1, w, b, loss_now))


# ============================================================
# 5. 결과 비교
# ============================================================
print()
print("=" * 60)
print("[5] 결과")
print("=" * 60)
print("진짜 값  : w=%.3f, b=%.3f" % (true_w, true_b))
print("학습 결과: w=%.3f, b=%.3f" % (w, b))
print()
print("→ 노이즈 때문에 완전히 똑같진 않아도 거의 같은 직선을 찾아냈다.")


# ============================================================
# 6. 학습률을 바꿔보면?
# ============================================================
# 너무 크면 손실이 진동·발산하고, 너무 작으면 200 스텝으론 못 따라잡는다.
# 이 감각이 신경망 학습에서 가장 중요하다.

def train_once(learning_rate, num_steps):
    w_local, b_local = 0.0, 0.0
    for _ in range(num_steps):
        grad_w, grad_b = gradients(w_local, b_local, x_data, y_data)
        w_local = w_local - learning_rate * grad_w
        b_local = b_local - learning_rate * grad_b
    return w_local, b_local, mse_loss(w_local, b_local, x_data, y_data)

print()
print("=" * 60)
print("[6] 학습률 비교 (200 스텝)")
print("=" * 60)
print("%-10s  %-8s  %-8s  %s" % ("lr", "w", "b", "loss"))

for lr_test in [0.001, 0.01, 0.05, 0.2, 0.3]:
    w_t, b_t, loss_t = train_once(lr_test, 200)
    diverged = np.isnan(loss_t) or loss_t > 1e6
    if diverged:
        # 너무 큰 학습률은 손실이 폭발해서 숫자 자체가 의미 없어진다.
        print("%-10.4f  %-8s  %-8s  %s" % (lr_test, "?", "?", "발산"))
    else:
        print("%-10.4f  %-8.3f  %-8.3f  %.4f" % (lr_test, w_t, b_t, loss_t))


# ============================================================
# 7. 정리
# ============================================================
print()
print("=" * 60)
print("[정리]")
print("=" * 60)
print("- 학습 = 손실의 기울기를 따라 가중치를 조금씩 반대 방향으로 옮기는 것.")
print("- 한 스텝:  w ← w - learning_rate * grad_w")
print("- 학습률은 너무 크면 발산, 너무 작으면 더디다.")
print("- 다음 주: 직선 회귀가 아니라 '글자 다음 글자' 예측 = 언어모델로 간다.")
