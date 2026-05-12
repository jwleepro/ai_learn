"""Week 5 — 크로스엔트로피 손실과 역전파 (직접 유도).

이 주에 배우는 것
- 크로스엔트로피(cross-entropy) 손실이 무엇인지, 왜 그 모양인지.
- "기울기" 가 행렬을 통과할 때 어떻게 흘러가는지 — 역전파(backpropagation).
- 손으로 유도한 미분식을 numerical gradient(수치 미분)와 비교해서 같은지 검증.

왜 이게 필요한가
- 다음 주부터는 PyTorch 가 알아서 미분해주는 척 가르쳐도 되지만,
  딱 한 번은 손으로 유도해봐야 "역전파가 마법이 아니다" 가 몸에 들어온다.
- 이 한 주가 끝나면 모든 신경망 학습 코드가 "다 똑같은 패턴" 으로 보인다.

실행:  python week5.py
"""

import numpy as np


# ============================================================
# 1. 크로스엔트로피 손실 — 왜 -log(p) 인가
# ============================================================
# 모델 출력: probs = [p_0, p_1, ..., p_{V-1}]   (합=1)
# 정답:     target_id (정답 한 칸)
#
# 손실 정의:  loss = -log(p_target)
#
# 직관:
#   - p_target = 1.0  →  -log(1) = 0   (완벽한 확신)
#   - p_target = 0.5  →  -log(0.5) ≈ 0.693
#   - p_target = 0.01 →  -log(0.01) ≈ 4.605  (큰 손실, 정답을 거의 안 줬음)
#
# "정답에 더 큰 확률을 줄수록 손실이 줄어든다." 그게 전부다.
#
# (수학적으로는 KL-divergence 의 한 줄 짜리 형태인데, 직관 위주로만 잡고 간다.)

print("=" * 60)
print("[1] 크로스엔트로피 손실의 모양")
print("=" * 60)

for p in [0.99, 0.5, 0.1, 0.01]:
    print("p_target = %.2f  →  loss = -log(p) = %.4f" % (p, -np.log(p)))


# ============================================================
# 2. softmax + cross-entropy 의 마법 — 미분이 깔끔해진다
# ============================================================
# 입력: logits z  (벡터, shape (V,))
# 단계 1: probs = softmax(z)
# 단계 2: loss  = -log(probs[target])
#
# 손으로 유도한 결과 (정말로 한번 종이에 해보기를 추천):
#
#   d(loss) / d(z_i) = probs[i] - 1   if i == target
#                    = probs[i]       otherwise
#
#   = probs - one_hot(target)
#
# 이 한 줄이 신경망 학습에서 가장 중요한 미분이다.
# 모든 분류 모델이 이걸로 학습한다.
#
# 코드로:

def softmax(z):
    z_safe = z - np.max(z)
    e = np.exp(z_safe)
    return e / np.sum(e)


def loss_and_grad_logits(z, target_id):
    probs = softmax(z)
    loss = -np.log(probs[target_id] + 1e-12)

    grad_z = probs.copy()
    grad_z[target_id] -= 1.0   # probs - one_hot(target)
    return loss, grad_z, probs


print()
print("=" * 60)
print("[2] 손으로 유도한 미분식 검증")
print("=" * 60)

# 실험: 무작위 logits 와 무작위 target 으로 검증
rng = np.random.default_rng(0)
V = 6
z = rng.standard_normal(V)
target_id = 2

loss, grad_z_analytic, probs = loss_and_grad_logits(z, target_id)

print("z       =", z)
print("probs   =", probs)
print("target  =", target_id)
print("loss    = %.4f" % loss)
print("grad_z (분석적)  =", grad_z_analytic)

# 수치 미분(numerical gradient)으로 검증
# d(loss)/d(z_i) ≈ ( loss(z + eps*e_i) - loss(z - eps*e_i) ) / (2*eps)
def numerical_grad(z, target_id, eps=1e-6):
    grad = np.zeros_like(z)
    for i in range(len(z)):
        z_plus = z.copy()
        z_plus[i] += eps
        z_minus = z.copy()
        z_minus[i] -= eps

        loss_plus, _, _ = loss_and_grad_logits(z_plus, target_id)
        loss_minus, _, _ = loss_and_grad_logits(z_minus, target_id)

        grad[i] = (loss_plus - loss_minus) / (2 * eps)
    return grad

grad_z_numerical = numerical_grad(z, target_id)
print("grad_z (수치적)  =", grad_z_numerical)
print("최대 차이       = %.2e" % np.max(np.abs(grad_z_analytic - grad_z_numerical)))
print("→ 거의 0 이면 분석식이 맞다.")


# ============================================================
# 3. 행렬곱을 통과하는 기울기 — 체인룰 (Chain Rule)
# ============================================================
# 더 큰 신경망의 한 층:
#
#   y = W @ x        (W shape: (M, N), x shape: (N,), y shape: (M,))
#   loss = ... y 로 만든 함수 ...
#
# 우리가 이미 "d(loss)/d(y) = grad_y" 를 안다고 하자.
# 그러면 W 와 x 의 기울기는?
#
#   d(loss)/d(W) = outer(grad_y, x)   shape (M, N)
#   d(loss)/d(x) = W.T @ grad_y       shape (N,)
#
# 외울 필요 없다. 모양 맞춰보면 자연히 나온다:
#   - W shape 은 (M, N) 이니 기울기도 (M, N) 모양이어야 한다.
#   - grad_y (M,) 와 x (N,) 의 outer product 가 정확히 (M, N).
#
# 이걸 직접 검증해보자.

print()
print("=" * 60)
print("[3] 행렬곱을 통과하는 기울기 검증")
print("=" * 60)

# 작은 예: y = W @ x, loss = sum(y^2)/2  (단순한 손실)
W_test = rng.standard_normal((4, 3))
x_test = rng.standard_normal(3)

def forward_loss(W, x):
    y = W @ x
    return 0.5 * np.sum(y * y), y

loss_value, y_val = forward_loss(W_test, x_test)
# d(loss)/d(y) = y
grad_y = y_val
# 위 공식 적용
grad_W_analytic = np.outer(grad_y, x_test)
grad_x_analytic = W_test.T @ grad_y

print("loss = %.4f" % loss_value)
print("grad_W shape =", grad_W_analytic.shape, "(W shape =", W_test.shape, ")")
print("grad_x shape =", grad_x_analytic.shape, "(x shape =", x_test.shape, ")")

# 수치 미분으로 검증
def num_grad_W(W, x, eps=1e-6):
    grad = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_plus = W.copy()
            W_plus[i, j] += eps
            W_minus = W.copy()
            W_minus[i, j] -= eps
            l_plus, _ = forward_loss(W_plus, x)
            l_minus, _ = forward_loss(W_minus, x)
            grad[i, j] = (l_plus - l_minus) / (2 * eps)
    return grad

grad_W_numerical = num_grad_W(W_test, x_test)
print("grad_W 최대 차이 = %.2e" % np.max(np.abs(grad_W_analytic - grad_W_numerical)))
print("→ 분석식과 수치 미분이 일치.")


# ============================================================
# 4. 다 합치면 — week4 의 신경망 빅램 한 스텝을 검증
# ============================================================
# week4 에서 우리는 이렇게 썼다:
#
#   embedding = E[prev_id]                   (D,)
#   logits    = embedding @ W                (V,)
#   probs     = softmax(logits)
#   loss      = -log(probs[target_id])
#
# 그래디언트 (체인룰 적용):
#   d_logits = probs - one_hot(target)       (V,)
#   d_W      = outer(embedding, d_logits)    (D, V)
#   d_embed  = W @ d_logits                  (D,)
#   d_E[prev_id] = d_embed   (다른 행은 0)

print()
print("=" * 60)
print("[4] 신경망 빅램 한 스텝 — 분석식 vs 수치 미분")
print("=" * 60)

V_test = 5
D_test = 3
E = rng.standard_normal((V_test, D_test)) * 0.5
W = rng.standard_normal((D_test, V_test)) * 0.5
prev_id = 1
target_id = 4

def step_loss(E, W, prev_id, target_id):
    embedding = E[prev_id]
    logits = embedding @ W
    probs = softmax(logits)
    return -np.log(probs[target_id] + 1e-12), probs, embedding

# 분석적 기울기
loss_v, probs_v, embedding_v = step_loss(E, W, prev_id, target_id)
d_logits = probs_v.copy()
d_logits[target_id] -= 1.0
d_W_analytic = np.outer(embedding_v, d_logits)
d_E_analytic = np.zeros_like(E)
d_E_analytic[prev_id] = W @ d_logits

# 수치 미분 (W 의 한 칸씩 흔들어보기)
def num_grad(get_loss, param, *args, eps=1e-6):
    grad = np.zeros_like(param)
    flat = param.reshape(-1)
    grad_flat = grad.reshape(-1)
    for k in range(flat.size):
        original = flat[k]
        flat[k] = original + eps
        l_plus = get_loss(*args)
        flat[k] = original - eps
        l_minus = get_loss(*args)
        flat[k] = original
        grad_flat[k] = (l_plus - l_minus) / (2 * eps)
    return grad

def loss_only_W():
    l, _, _ = step_loss(E, W, prev_id, target_id)
    return l

def loss_only_E():
    l, _, _ = step_loss(E, W, prev_id, target_id)
    return l

d_W_numerical = num_grad(loss_only_W, W)
d_E_numerical = num_grad(loss_only_E, E)

print("W 기울기 최대 차이 = %.2e" % np.max(np.abs(d_W_analytic - d_W_numerical)))
print("E 기울기 최대 차이 = %.2e" % np.max(np.abs(d_E_analytic - d_E_numerical)))
print("→ 두 값이 거의 0 = 우리가 손으로 유도한 식이 맞다.")


# ============================================================
# 5. 실제 학습 한 번 더 (검증된 미분으로)
# ============================================================
# 한 스텝 진행 전후의 loss 가 진짜로 줄어드는지 확인.

print()
print("=" * 60)
print("[5] 한 스텝 학습 → loss 가 실제로 줄어드는가")
print("=" * 60)

lr = 0.5
loss_before, _, _ = step_loss(E, W, prev_id, target_id)
E = E - lr * d_E_analytic
W = W - lr * d_W_analytic
loss_after, _, _ = step_loss(E, W, prev_id, target_id)
print("학습 전 loss = %.4f" % loss_before)
print("학습 후 loss = %.4f  (작아져야 정상)" % loss_after)


# ============================================================
# 6. 정리
# ============================================================
print()
print("=" * 60)
print("[정리]")
print("=" * 60)
print("- 크로스엔트로피: loss = -log(p_target). 직관: 정답에 자신없으면 큰 벌점.")
print("- softmax + CE 의 마법:  d(loss)/d(logits) = probs - one_hot(target).")
print("- 행렬곱을 통과하는 기울기:  d_W = outer(in, d_out),  d_in = W.T @ d_out.")
print("- 수치 미분과 비교해서 검증할 수 있다 (= gradient check).")
print("- 다음 주: 컨텍스트를 늘린 MLP 언어모델로 확장.")
