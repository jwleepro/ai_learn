"""Week 8 — Transformer 블록 조립.

이 주에 배우는 것
- Multi-Head Attention: self-attention 을 여러 개 병렬로 돌리고 결과 합치기.
- 위치 임베딩(positional embedding): 글자가 "몇 번째 위치" 인지 모델에 알려주기.
- LayerNorm: 학습이 폭주하지 않도록 활성값을 정규화.
- 잔차 연결(residual / skip connection): 깊은 신경망이 잘 학습되게 하는 핵심 트릭.
- FFN(feed-forward): 위치별로 적용되는 작은 MLP.

이번 주도 forward 만 한다. 학습은 week9.

Transformer 블록 한 장면
   x  ──┬──> LayerNorm ──> Multi-Head Self-Attn ──> + ─┬──> LayerNorm ──> FFN ──> + ──> out
        │                                              │                         │
        └──────────── (잔차) ──────────────────────────┘                         │
                                                                                 │
        └──────────────────────────── (잔차) ─────────────────────────────────────┘

블록을 N개 쌓아서 GPT 가 만들어진다 (week9).

실행:  python week8.py
"""

import numpy as np


# ============================================================
# 1. softmax (행마다)
# ============================================================
def softmax_rows(Z):
    Z_safe = Z - np.max(Z, axis=1, keepdims=True)
    e = np.exp(Z_safe)
    return e / np.sum(e, axis=1, keepdims=True)


# ============================================================
# 2. Multi-Head Self-Attention
# ============================================================
# week7 에서는 head 가 하나였다. 그걸 H 개로 쪼개서 병렬로 돌린다.
#
# 가정: D 가 H 로 나누어 떨어진다. head_dim = D / H.
# 각 헤드는 head_dim 차원으로 동작한다. 헤드끼리 다른 패턴을 학습한다.
#
# 흐름:
#   Q_full = x @ W_Q   shape (T, D)   ← 한 번에 만든 다음
#   Q[h]   = Q_full[:, h*head_dim : (h+1)*head_dim]   ← 헤드별로 자른다
#
# 각 헤드마다:
#   scores = Q[h] @ K[h].T / sqrt(head_dim)
#   + causal mask
#   weights = softmax(scores)
#   attn[h] = weights @ V[h]
#
# 마지막에 모든 헤드 결과를 옆으로 이어붙이고(concat) 한 번 더 행렬곱(W_O):
#   concat = [attn[0] | attn[1] | ... | attn[H-1]]   shape (T, D)
#   out    = concat @ W_O                              shape (T, D)

def make_causal_mask(T):
    mask = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            if j > i:
                mask[i, j] = -1e9
    return mask


def multi_head_attention(x, W_Q, W_K, W_V, W_O, num_heads):
    T = x.shape[0]
    D = x.shape[1]
    head_dim = D // num_heads

    # 한 번에 큰 Q, K, V 만들기
    Q_full = x @ W_Q     # (T, D)
    K_full = x @ W_K     # (T, D)
    V_full = x @ W_V     # (T, D)

    # 헤드별로 잘라서 저장 (3D 배열: (H, T, head_dim))
    Q_heads = np.zeros((num_heads, T, head_dim))
    K_heads = np.zeros((num_heads, T, head_dim))
    V_heads = np.zeros((num_heads, T, head_dim))
    for h in range(num_heads):
        col_start = h * head_dim
        col_end = (h + 1) * head_dim
        Q_heads[h] = Q_full[:, col_start:col_end]
        K_heads[h] = K_full[:, col_start:col_end]
        V_heads[h] = V_full[:, col_start:col_end]

    causal = make_causal_mask(T)

    # 각 헤드별로 attention 계산
    attn_per_head = np.zeros((num_heads, T, head_dim))
    for h in range(num_heads):
        scores = Q_heads[h] @ K_heads[h].T          # (T, T)
        scores = scores / np.sqrt(head_dim)
        scores = scores + causal
        weights = softmax_rows(scores)              # (T, T)
        attn_per_head[h] = weights @ V_heads[h]     # (T, head_dim)

    # 헤드들을 옆으로 이어붙이기 → (T, D)
    concat = np.zeros((T, D))
    for h in range(num_heads):
        col_start = h * head_dim
        col_end = (h + 1) * head_dim
        concat[:, col_start:col_end] = attn_per_head[h]

    # 마지막 출력 투영 (output projection)
    out = concat @ W_O                               # (T, D)
    return out


# ============================================================
# 3. LayerNorm
# ============================================================
# 각 토큰 벡터를 그 벡터 안에서 정규화 (평균 0, 분산 1) 한 다음
# 학습 가능한 gamma, beta 로 스케일·이동.
#
#   y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
#
# 왜 필요한가?
#   - 깊은 신경망에서 활성값이 점점 커지거나 작아지는 걸 막아 학습이 안정.

def layer_norm(x, gamma, beta, eps=1e-5):
    # x shape (T, D). 각 행(=각 토큰)별로 mean/var.
    mean = np.mean(x, axis=1, keepdims=True)        # (T, 1)
    var = np.var(x, axis=1, keepdims=True)          # (T, 1)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


# ============================================================
# 4. FFN — 위치별로 똑같이 적용되는 작은 MLP
# ============================================================
# 보통 hidden 을 4D 정도로 키운다. ReLU 또는 GELU 사용.
#   y = relu(x @ W_ff1 + b_ff1) @ W_ff2 + b_ff2

def relu(z):
    return np.maximum(0.0, z)


def feed_forward(x, W_ff1, b_ff1, W_ff2, b_ff2):
    h = relu(x @ W_ff1 + b_ff1)
    return h @ W_ff2 + b_ff2


# ============================================================
# 5. Transformer 블록 한 개 (Pre-LN 형태)
# ============================================================
# Pre-LN = LayerNorm 을 sub-layer 앞에 두는 방식 (현대 GPT 가 쓰는 표준).
# 잔차 연결 두 번:
#
#   1) x = x + MultiHeadAttn( LayerNorm(x) )
#   2) x = x + FFN( LayerNorm(x) )

def transformer_block(x, params, num_heads):
    # 1) Attention sub-layer
    x_norm1 = layer_norm(x, params["ln1_gamma"], params["ln1_beta"])
    attn_out = multi_head_attention(
        x_norm1,
        params["W_Q"], params["W_K"], params["W_V"], params["W_O"],
        num_heads,
    )
    x = x + attn_out                          # 잔차 1

    # 2) FFN sub-layer
    x_norm2 = layer_norm(x, params["ln2_gamma"], params["ln2_beta"])
    ffn_out = feed_forward(
        x_norm2,
        params["W_ff1"], params["b_ff1"],
        params["W_ff2"], params["b_ff2"],
    )
    x = x + ffn_out                           # 잔차 2
    return x


# ============================================================
# 6. 위치 임베딩
# ============================================================
# self-attention 은 입력 순서를 모른다 (행렬곱 + softmax 만 있으니).
# "이게 0번 자리 글자, 저게 1번 자리" 같은 위치 정보를 따로 더해줘야 한다.
#
# 가장 단순한 방법: 위치마다 학습 가능한 벡터 하나씩.
#
#   positional_emb shape (max_T, D)
#   x = token_emb + positional_emb[:T]


# ============================================================
# 7. 한 번 돌려보기 — 입력 → 임베딩 → Transformer 블록 → 출력 logits
# ============================================================
def main():
    rng = np.random.default_rng(0)

    # 모델 크기
    V = 30          # vocab size (글자 가짓수, 가짜 데모)
    D = 16          # 임베딩 차원
    NUM_HEADS = 4   # head 수 (D 가 4 로 나눠 떨어져야 함)
    FFN_HIDDEN = 4 * D
    MAX_T = 32      # 최대 시퀀스 길이

    # 파라미터 초기화
    token_emb = rng.standard_normal((V, D)) * 0.1
    positional_emb = rng.standard_normal((MAX_T, D)) * 0.1

    block_params = {
        "ln1_gamma": np.ones(D),
        "ln1_beta":  np.zeros(D),
        "W_Q": rng.standard_normal((D, D)) * 0.1,
        "W_K": rng.standard_normal((D, D)) * 0.1,
        "W_V": rng.standard_normal((D, D)) * 0.1,
        "W_O": rng.standard_normal((D, D)) * 0.1,
        "ln2_gamma": np.ones(D),
        "ln2_beta":  np.zeros(D),
        "W_ff1": rng.standard_normal((D, FFN_HIDDEN)) * 0.1,
        "b_ff1": np.zeros(FFN_HIDDEN),
        "W_ff2": rng.standard_normal((FFN_HIDDEN, D)) * 0.1,
        "b_ff2": np.zeros(D),
    }
    # 마지막 출력층 (D → V) — "이 위치 다음 글자 logits"
    W_out = rng.standard_normal((D, V)) * 0.1
    b_out = np.zeros(V)

    # 가짜 입력 시퀀스 (글자 ID 8개)
    input_ids = np.array([3, 7, 12, 4, 9, 1, 18, 2])
    T = len(input_ids)
    print("=" * 60)
    print("[7] 한 블록 돌려보기")
    print("=" * 60)
    print("입력 ID :", input_ids)
    print("T = %d, D = %d, NUM_HEADS = %d" % (T, D, NUM_HEADS))

    # 1) 토큰 임베딩 + 위치 임베딩
    #    x shape (T, D)
    x = np.zeros((T, D))
    for i in range(T):
        x[i] = token_emb[input_ids[i]] + positional_emb[i]

    print("임베딩 후 x shape =", x.shape)

    # 2) Transformer 블록 한 번
    x = transformer_block(x, block_params, NUM_HEADS)
    print("블록 후    x shape =", x.shape)

    # 3) 출력 logits (각 위치마다 다음 글자 V 차원 점수)
    logits = x @ W_out + b_out             # (T, V)
    print("logits shape =", logits.shape)

    # 4) 마지막 위치의 다음 글자 확률 분포
    last_probs = softmax_rows(logits)[-1]   # (V,)
    print()
    print("마지막 위치의 다음 글자 확률 (V=%d 칸 중 처음 8개):" % V)
    print(np.round(last_probs[:8], 4))
    print("합 =", last_probs.sum())

    # 학습 안 했으니 의미 있는 분포는 아니다. shape/흐름이 맞는지가 포인트.

    # ========================================================
    # 8. 블록을 두 번 쌓아 보기 (= 더 깊은 모델)
    # ========================================================
    print()
    print("=" * 60)
    print("[8] 같은 블록을 2번 쌓기")
    print("=" * 60)

    # 두 번째 블록 파라미터를 다시 만든다 (실제 GPT 도 블록마다 다른 가중치).
    block_params_2 = {
        "ln1_gamma": np.ones(D),
        "ln1_beta":  np.zeros(D),
        "W_Q": rng.standard_normal((D, D)) * 0.1,
        "W_K": rng.standard_normal((D, D)) * 0.1,
        "W_V": rng.standard_normal((D, D)) * 0.1,
        "W_O": rng.standard_normal((D, D)) * 0.1,
        "ln2_gamma": np.ones(D),
        "ln2_beta":  np.zeros(D),
        "W_ff1": rng.standard_normal((D, FFN_HIDDEN)) * 0.1,
        "b_ff1": np.zeros(FFN_HIDDEN),
        "W_ff2": rng.standard_normal((FFN_HIDDEN, D)) * 0.1,
        "b_ff2": np.zeros(D),
    }

    # 다시 임베딩부터
    x2 = np.zeros((T, D))
    for i in range(T):
        x2[i] = token_emb[input_ids[i]] + positional_emb[i]

    x2 = transformer_block(x2, block_params, NUM_HEADS)
    x2 = transformer_block(x2, block_params_2, NUM_HEADS)

    print("두 블록 통과 후 x2 shape =", x2.shape)
    print("→ 블록을 N번 쌓는 것이 곧 깊은 GPT 다. (다음 주에 학습까지 한다.)")


main()


# ============================================================
# 9. 정리
# ============================================================
print()
print("=" * 60)
print("[정리]")
print("=" * 60)
print("- Multi-Head Attention = 같은 입력에서 H개의 다른 attention 을 병렬로.")
print("- LayerNorm + 잔차 = 깊은 신경망이 학습 가능하게 만드는 두 트릭.")
print("- FFN = 위치별로 적용되는 작은 MLP. 비선형 표현력을 더해준다.")
print("- 위치 임베딩 = 'i 번째 자리' 라는 정보를 토큰 임베딩에 더해주는 것.")
print("- Transformer 블록 = LN→MHA→+잔차→LN→FFN→+잔차.")
print("- 다음 주: 블록을 N번 쌓아 numpy 만으로 MiniGPT 를 학습시킨다.")
