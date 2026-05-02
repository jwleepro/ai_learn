"""Week 7 — Self-Attention.

이 주에 배우는 것
- Transformer 의 심장 = self-attention.
- Q(Query), K(Key), V(Value) 가 무엇이고 왜 3개나 필요한지.
- causal mask 가 왜 필요한지 (= "미래의 글자를 미리 보면 안 된다").
- 모든 위치의 글자가 모든 이전 위치의 글자를 "보고" 자기 표현을 갱신한다.

이번 주는 forward 만 한다. 학습은 week9.

직관 한 줄
- MLP(week6) 는 "마지막 N글자" 를 한 덩어리로 본다.
- Self-attention 은 각 위치에서 "지금까지 본 모든 글자 중 어느 게 중요한가" 를 학습한다.
- 그래서 "안녕하세요. ... 안녕히 가세요" 같은 긴 의존도 잡을 수 있다.

실행:  python week7.py
"""

import numpy as np


# ============================================================
# 1. 큰 그림 — self-attention 한 줄로
# ============================================================
# 입력: x  shape (T, D)
#   - T = 시퀀스 길이 (글자 몇 개 들어왔는지)
#   - D = 각 글자의 임베딩 차원
#
# 가중치 3개 (학습 가능):
#   W_Q, W_K, W_V   각각 shape (D, D)
#
# 계산:
#   Q = x @ W_Q       shape (T, D)
#   K = x @ W_K       shape (T, D)
#   V = x @ W_V       shape (T, D)
#
#   scores = Q @ K.T              shape (T, T)   ← "위치 i 가 위치 j 에 얼마나 관심"
#   scores = scores / sqrt(D)     ← 스케일 조정 (분산 폭주 방지)
#   scores[i, j] = -∞ if j > i    ← causal mask: 미래를 가리기
#
#   weights = softmax(scores, axis=1)   shape (T, T)   ← 행마다 합=1
#   out     = weights @ V                shape (T, D)
#
# out[i] = "위치 i 에서 본 새 표현". 이전 위치 V 들의 가중 평균.


# ============================================================
# 2. softmax (행마다)
# ============================================================
def softmax_rows(Z):
    Z_safe = Z - np.max(Z, axis=1, keepdims=True)
    e = np.exp(Z_safe)
    return e / np.sum(e, axis=1, keepdims=True)


# ============================================================
# 3. causal mask 만들기
# ============================================================
# T x T 행렬에서 j > i 인 자리에 -1e9 (사실상 -∞) 를 넣는다.
# softmax 통과하면 그 자리는 거의 0 이 되어 "미래" 위치가 무시된다.
#
# Java 라면 이중 for 루프. numpy 도 그대로.

def make_causal_mask(T):
    mask = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            if j > i:
                mask[i, j] = -1e9
    return mask


# ============================================================
# 4. self-attention 본체
# ============================================================
def self_attention(x, W_Q, W_K, W_V):
    T = x.shape[0]
    D = x.shape[1]

    Q = x @ W_Q                          # (T, D)
    K = x @ W_K                          # (T, D)
    V = x @ W_V                          # (T, D)

    scores = Q @ K.T                     # (T, T)
    scores = scores / np.sqrt(D)         # 스케일

    scores = scores + make_causal_mask(T)   # 미래 가리기

    weights = softmax_rows(scores)       # (T, T)
    out = weights @ V                    # (T, D)

    return out, weights


# ============================================================
# 5. 시연 — 작은 입력에 self-attention 한 번 돌려보기
# ============================================================
print("=" * 60)
print("[5] self-attention 시연")
print("=" * 60)

rng = np.random.default_rng(0)

T = 5      # 시퀀스 길이 (5 글자)
D = 4      # 임베딩 차원 4

# 입력 임베딩 (실제론 글자 ID 를 임베딩 테이블에 넣어 만든 것)
x = rng.standard_normal((T, D))

# 가중치 무작위 초기화 (학습은 다음 주에)
W_Q = rng.standard_normal((D, D)) * 0.3
W_K = rng.standard_normal((D, D)) * 0.3
W_V = rng.standard_normal((D, D)) * 0.3

out, weights = self_attention(x, W_Q, W_K, W_V)

print("입력 x shape    =", x.shape)
print("출력 out shape  =", out.shape)
print()
print("attention weights (T x T) — 행 i 는 위치 i 가 어디에 집중했는가:")
print(weights)
print()
print("관찰:")
print("  - 첫 행은 [1, 0, 0, 0, 0]: 첫 위치는 자기 자신만 볼 수 있다.")
print("  - 윗삼각(미래) 부분이 모두 0: causal mask 가 잘 들었다.")
print("  - 각 행의 합은 1 (softmax 의 성질):", weights.sum(axis=1))


# ============================================================
# 6. 왜 Q/K/V 가 따로 필요한가
# ============================================================
# Q (Query): "내가 어떤 정보를 찾고 있나"
# K (Key)  : "이 위치는 어떤 정보를 갖고 있나"
# V (Value): "그래서 실제로 가져올 내용"
#
# Java 비유: HashMap 검색.
#   key 가 일치하면 value 를 가져온다.
#   여기서는 hard match 가 아니라 soft match (점수 + softmax).
#
# 같은 입력 x 에서 세 가지 다른 역할의 벡터를 뽑아내기 위해 각각 다른 W 를 쓴다.
# 만약 Q 와 K 가 같은 가중치였다면, "나 자신만 잘 매칭" 되는 단조로운 패턴이 됐을 것.


# ============================================================
# 7. dot-product 점수의 의미를 직접 보기
# ============================================================
# 가중치를 일부러 단순하게 만들어, 같은 글자끼리 attention 이 잘 걸리는지 확인.
#
# 만약 입력이 [a, b, a, b, a] 같은 패턴이고 W_Q = W_K = I (항등 행렬) 라면,
# 같은 위치끼리 dot product 가 커지므로 weights 에서 같은 글자끼리 가중치가 높을 것.

print()
print("=" * 60)
print("[7] 같은 패턴이 반복되면 attention 이 어디로 가나")
print("=" * 60)

# 단순한 두 종류 임베딩
emb_a = np.array([1.0, 0.0, 0.0, 0.0])
emb_b = np.array([0.0, 1.0, 0.0, 0.0])
x_pattern = np.stack([emb_a, emb_b, emb_a, emb_b, emb_a])   # (5, 4)

W_Q_id = np.eye(4)        # Q = K = identity
W_K_id = np.eye(4)
W_V_id = np.eye(4)

_, weights_pattern = self_attention(x_pattern, W_Q_id, W_K_id, W_V_id)

print("입력 패턴: a b a b a (위치 0~4)")
print("attention weights:")
print(np.round(weights_pattern, 3))
print()
print("관찰:")
print("  - 마지막 행(위치 4, a) 을 보면, 이전 위치 중에서")
print("    a 위치(0, 2) 의 가중치가 b 위치(1, 3) 보다 높다.")
print("  - causal 때문에 위치 4 를 본인은 못 보지만, 같은 글자 a 인 0, 2 에 집중.")


# ============================================================
# 8. 정리
# ============================================================
print()
print("=" * 60)
print("[정리]")
print("=" * 60)
print("- self-attention 은 각 위치가 '이전 모든 위치 중 누구를 볼지' 학습한다.")
print("- 핵심 5줄:")
print("    Q, K, V = x @ W_Q, x @ W_K, x @ W_V")
print("    scores = Q @ K.T / sqrt(D)")
print("    scores += causal_mask        # 미래 가리기")
print("    weights = softmax(scores)")
print("    out = weights @ V")
print("- Q/K/V 가 다른 역할인 이유: 같은 정보에서 '찾는 것/갖고 있는 것/가져올 것' 을 분리.")
print("- 다음 주: 이걸 헤드 여러 개로 병렬화하고(Multi-Head) Transformer 블록을 조립.")
