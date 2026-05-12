"""Week 9 — MiniGPT 학습 + 샘플링.

이 주에 배우는 것
- 작은 Transformer LM 을 numpy 로 처음부터 끝까지 학습.
- 모든 위치에서 동시에 다음 토큰을 예측하는 "언어모델 학습" 의 표준 방식.
- 샘플링 옵션: temperature, top-k, top-p 가 어떻게 결과를 바꾸는지.

이 주의 단순화 (의도적)
- LayerNorm 은 빼서 backprop 을 짧게 유지한다 (학습은 잘 된다).
  실제 GPT 에는 들어 있다 — week8 에서 이미 봤고, week10 의 PyTorch 버전엔 추가한다.
- single-head attention. multi-head 는 같은 걸 H 번 반복하는 것이라 핵심 아이디어는 같다.
- Transformer 블록은 1 개. 블록 N 개 쌓는 건 동일 코드 N 번 돌리는 것일 뿐.

모델 구조
   ids                                       (B, T)
     │  embedding lookup
     ▼
   x = token_emb + position_emb              (B, T, D)
     │  attn_out = self_attention(x)
     ▼
   x = x + attn_out                          (잔차)
     │  ffn_out = relu(x W1 + b1) W2 + b2
     ▼
   x = x + ffn_out                           (잔차)
     │  output projection
     ▼
   logits = x W_out + b_out                  (B, T, V)

손실
   각 (b, t) 위치에서 다음 토큰(target_ids[b, t]) 의 cross-entropy 평균.

실행:  python week9.py
"""

import numpy as np


# ============================================================
# 0. 데이터
# ============================================================
TRAINING_TEXT = (
    "안녕하세요. 오늘 날씨가 참 좋네요.\n"
    "안녕하세요. 식사는 하셨나요?\n"
    "안녕하세요. 만나서 반갑습니다.\n"
    "오늘 날씨가 정말 좋습니다.\n"
    "오늘 식사는 무엇을 드셨나요?\n"
    "내일은 비가 올 것 같습니다.\n"
    "내일 만나서 식사 같이 해요.\n"
    "감사합니다. 좋은 하루 되세요.\n"
    "감사합니다. 또 만나요.\n"
    "반갑습니다. 좋은 시간 보내세요.\n"
    "오늘은 좋은 하루 보내세요.\n"
    "내일도 좋은 시간 되세요.\n"
)


class CharTokenizer:
    def __init__(self, text):
        unique_chars = sorted(set(text))
        self.id_to_char = unique_chars
        self.char_to_id = {}
        for i in range(len(unique_chars)):
            self.char_to_id[unique_chars[i]] = i

    def vocab_size(self):
        return len(self.id_to_char)

    def encode(self, text):
        ids = []
        for ch in text:
            ids.append(self.char_to_id[ch])
        return ids

    def decode(self, ids):
        chars = []
        for token_id in ids:
            chars.append(self.id_to_char[token_id])
        return "".join(chars)


# ============================================================
# 1. 데이터셋 만들기
# ============================================================
# 텍스트를 길이 T+1 짜리 윈도우로 자르고 입력/정답으로 분리:
#   input_ids  = window[:T]      (B, T)
#   target_ids = window[1:T+1]   (B, T)   ← 한 칸씩 미래
#
# 한 윈도우 안에서 모든 위치가 학습 신호를 만든다 → "한 시퀀스에 T 개의 학습 쌍".

def make_dataset(ids, T, stride=1):
    inputs = []
    targets = []
    i = 0
    while i + T < len(ids):
        window_input = []
        window_target = []
        for k in range(T):
            window_input.append(ids[i + k])
            window_target.append(ids[i + k + 1])
        inputs.append(window_input)
        targets.append(window_target)
        i = i + stride
    return np.array(inputs), np.array(targets)


# ============================================================
# 2. 헬퍼: softmax (마지막 축)
# ============================================================
def softmax_lastdim(Z):
    Z_safe = Z - np.max(Z, axis=-1, keepdims=True)
    e = np.exp(Z_safe)
    return e / np.sum(e, axis=-1, keepdims=True)


def relu(z):
    return np.maximum(0.0, z)


# ============================================================
# 3. causal mask (한 번 만들고 재사용)
# ============================================================
def make_causal_mask(T):
    mask = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            if j > i:
                mask[i, j] = -1e9
    return mask


# ============================================================
# 4. Forward
# ============================================================
def forward(input_ids, params, causal_mask):
    """input_ids shape (B, T) → logits shape (B, T, V), cache."""
    B = input_ids.shape[0]
    T = input_ids.shape[1]
    D = params["W_Q"].shape[0]
    V = params["W_out"].shape[1]

    # 1) 토큰 + 위치 임베딩
    x = np.zeros((B, T, D))
    for b in range(B):
        for t in range(T):
            token_id = input_ids[b, t]
            x[b, t] = params["token_emb"][token_id] + params["pos_emb"][t]

    # 2) Attention (배치별로)
    # Q, K, V shape (B, T, D)
    Q = np.zeros((B, T, D))
    K = np.zeros((B, T, D))
    V_mat = np.zeros((B, T, D))
    for b in range(B):
        Q[b] = x[b] @ params["W_Q"]
        K[b] = x[b] @ params["W_K"]
        V_mat[b] = x[b] @ params["W_V"]

    # weights[b] shape (T, T), attn_raw[b] shape (T, D)
    weights = np.zeros((B, T, T))
    attn_raw = np.zeros((B, T, D))
    for b in range(B):
        scores = Q[b] @ K[b].T / np.sqrt(D)        # (T, T)
        scores = scores + causal_mask              # mask 미래
        weights[b] = softmax_lastdim(scores)       # 행마다 합=1
        attn_raw[b] = weights[b] @ V_mat[b]        # (T, D)

    # output projection
    attn_out = np.zeros((B, T, D))
    for b in range(B):
        attn_out[b] = attn_raw[b] @ params["W_O"]

    # 잔차 1
    x_after_attn = x + attn_out

    # 3) FFN (위치별로 적용 — 배치/시간 차원은 그냥 모두 통과시킨다)
    pre_hidden = x_after_attn @ params["W_ff1"] + params["b_ff1"]    # (B, T, H)
    hidden = relu(pre_hidden)
    ffn_out = hidden @ params["W_ff2"] + params["b_ff2"]              # (B, T, D)

    # 잔차 2
    x_final = x_after_attn + ffn_out

    # 4) 출력 logits
    logits = x_final @ params["W_out"] + params["b_out"]              # (B, T, V)

    cache = {
        "input_ids": input_ids,
        "x": x,
        "Q": Q, "K": K, "V": V_mat,
        "weights": weights,
        "attn_raw": attn_raw,
        "x_after_attn": x_after_attn,
        "pre_hidden": pre_hidden,
        "hidden": hidden,
        "ffn_out": ffn_out,
        "x_final": x_final,
    }
    return logits, cache


# ============================================================
# 5. Loss + Backward
# ============================================================
# (B, T, V) logits 와 (B, T) target_ids 가 주어지면
# 모든 (b, t) 위치의 cross-entropy 를 평균해 loss 1 개를 만든다.

def loss_and_backward(logits, target_ids, cache, params):
    B = logits.shape[0]
    T = logits.shape[1]
    V = logits.shape[2]
    D = cache["x"].shape[2]

    # ---- 손실 ----
    probs = softmax_lastdim(logits)        # (B, T, V)
    # 정답 칸 확률만 빼와서 평균
    correct_log_probs = 0.0
    count = B * T
    for b in range(B):
        for t in range(T):
            p = probs[b, t, target_ids[b, t]]
            correct_log_probs += -np.log(p + 1e-12)
    loss = correct_log_probs / count

    # ---- d_logits ----
    d_logits = probs.copy()
    for b in range(B):
        for t in range(T):
            d_logits[b, t, target_ids[b, t]] -= 1.0
    d_logits = d_logits / count            # (B, T, V)

    # ---- 출력층 W_out, b_out ----
    x_final = cache["x_final"]                                 # (B, T, D)
    # W_out shape (D, V)
    # 평면화하면 행렬곱: x_final.reshape(B*T, D).T @ d_logits.reshape(B*T, V)
    flat_x = x_final.reshape(B * T, D)
    flat_dlogits = d_logits.reshape(B * T, V)
    d_W_out = flat_x.T @ flat_dlogits                          # (D, V)
    d_b_out = np.sum(flat_dlogits, axis=0)                     # (V,)
    d_x_final = flat_dlogits @ params["W_out"].T               # (B*T, D)
    d_x_final = d_x_final.reshape(B, T, D)

    # ---- 잔차 2 분기 ----
    # x_final = x_after_attn + ffn_out
    d_x_after_attn = d_x_final.copy()
    d_ffn_out = d_x_final.copy()

    # ---- FFN backward ----
    # ffn_out = hidden @ W_ff2 + b_ff2
    hidden = cache["hidden"]
    flat_hidden = hidden.reshape(B * T, -1)                    # (B*T, H)
    flat_dffn = d_ffn_out.reshape(B * T, D)
    d_W_ff2 = flat_hidden.T @ flat_dffn                        # (H, D)
    d_b_ff2 = np.sum(flat_dffn, axis=0)                        # (D,)
    d_hidden = flat_dffn @ params["W_ff2"].T                   # (B*T, H)

    # hidden = relu(pre_hidden)
    pre_hidden = cache["pre_hidden"].reshape(B * T, -1)
    d_pre_hidden = d_hidden * (pre_hidden > 0)                 # (B*T, H)

    # pre_hidden = x_after_attn @ W_ff1 + b_ff1
    flat_xaa = cache["x_after_attn"].reshape(B * T, D)
    d_W_ff1 = flat_xaa.T @ d_pre_hidden                        # (D, H)
    d_b_ff1 = np.sum(d_pre_hidden, axis=0)                     # (H,)
    d_xaa_from_ffn = d_pre_hidden @ params["W_ff1"].T          # (B*T, D)
    d_xaa_from_ffn = d_xaa_from_ffn.reshape(B, T, D)

    # 잔차 2 의 두 갈래 합치기
    d_x_after_attn = d_x_after_attn + d_xaa_from_ffn

    # ---- 잔차 1 분기 ----
    # x_after_attn = x + attn_out
    d_x = d_x_after_attn.copy()
    d_attn_out = d_x_after_attn.copy()

    # ---- attn_out = attn_raw @ W_O ----
    attn_raw = cache["attn_raw"]
    d_W_O = np.zeros_like(params["W_O"])
    d_attn_raw = np.zeros_like(attn_raw)
    for b in range(B):
        d_W_O += attn_raw[b].T @ d_attn_out[b]                 # (D, D)
        d_attn_raw[b] = d_attn_out[b] @ params["W_O"].T        # (T, D)

    # ---- attention backward (per batch) ----
    Q = cache["Q"]; K = cache["K"]; V_mat = cache["V"]
    weights = cache["weights"]

    d_Q = np.zeros_like(Q)
    d_K = np.zeros_like(K)
    d_V = np.zeros_like(V_mat)

    for b in range(B):
        # attn_raw[b] = weights[b] @ V[b]
        d_weights_b = d_attn_raw[b] @ V_mat[b].T               # (T, T)
        d_V[b] = weights[b].T @ d_attn_raw[b]                  # (T, D)

        # softmax backward (행마다)
        # d_scores[i, j] = w[i, j] * (d_w[i, j] - sum_k(d_w[i, k] * w[i, k]))
        d_scores_b = np.zeros_like(weights[b])
        for i in range(T):
            row_sum = np.sum(d_weights_b[i] * weights[b, i])
            d_scores_b[i] = weights[b, i] * (d_weights_b[i] - row_sum)

        # mask 는 더해진 상수라 backprop 에 영향 없음
        # scores = Q[b] @ K[b].T / sqrt(D)
        scale = 1.0 / np.sqrt(D)
        d_Q[b] = (d_scores_b @ K[b]) * scale
        d_K[b] = (d_scores_b.T @ Q[b]) * scale

    # Q, K, V projections: Q = x @ W_Q, etc.
    # → d_W_Q = sum_b x[b].T @ d_Q[b],  d_x += d_Q @ W_Q.T
    x_emb = cache["x"]                                         # (B, T, D)

    d_W_Q = np.zeros_like(params["W_Q"])
    d_W_K = np.zeros_like(params["W_K"])
    d_W_V = np.zeros_like(params["W_V"])

    d_x_from_attn = np.zeros_like(x_emb)
    for b in range(B):
        d_W_Q += x_emb[b].T @ d_Q[b]
        d_W_K += x_emb[b].T @ d_K[b]
        d_W_V += x_emb[b].T @ d_V[b]
        d_x_from_attn[b] = d_Q[b] @ params["W_Q"].T \
                         + d_K[b] @ params["W_K"].T \
                         + d_V[b] @ params["W_V"].T

    # 잔차 1 의 두 갈래 합
    d_x = d_x + d_x_from_attn

    # ---- 임베딩 backward ----
    # x[b, t] = token_emb[ ids[b, t] ] + pos_emb[t]
    d_token_emb = np.zeros_like(params["token_emb"])
    d_pos_emb = np.zeros_like(params["pos_emb"])
    input_ids = cache["input_ids"]
    for b in range(B):
        for t in range(T):
            token_id = input_ids[b, t]
            d_token_emb[token_id] += d_x[b, t]
            d_pos_emb[t] += d_x[b, t]

    grads = {
        "token_emb": d_token_emb,
        "pos_emb": d_pos_emb,
        "W_Q": d_W_Q, "W_K": d_W_K, "W_V": d_W_V, "W_O": d_W_O,
        "W_ff1": d_W_ff1, "b_ff1": d_b_ff1,
        "W_ff2": d_W_ff2, "b_ff2": d_b_ff2,
        "W_out": d_W_out, "b_out": d_b_out,
    }
    return loss, grads


# ============================================================
# 6. 샘플링 — temperature / top-k / top-p
# ============================================================
# logits 한 줄(V,) 에서 다음 토큰 ID 1 개 뽑기.
#
# temperature : logits 를 T로 나눔. 작을수록 sharp(과감), 클수록 평탄(다양).
# top_k       : 상위 k 개만 남기고 나머지 확률 0.
# top_p       : 누적 확률이 p 이상이 될 때까지만 후보로 남김 (nucleus).

def sample_next_id(logits_row, rng, temperature=1.0, top_k=None, top_p=None):
    V = len(logits_row)

    # 1) temperature
    z = logits_row / max(temperature, 1e-6)

    # 2) top_k 가 있으면 top_k 외에는 -inf
    if top_k is not None and top_k < V:
        # 큰 순서대로 인덱스 정렬 → top_k 번째 큰 값
        sorted_idx = np.argsort(z)[::-1]
        keep_set = set()
        for i in range(top_k):
            keep_set.add(int(sorted_idx[i]))
        for i in range(V):
            if i not in keep_set:
                z[i] = -1e9

    probs = softmax_lastdim(z)

    # 3) top_p (nucleus)
    if top_p is not None and top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        cumulative = 0.0
        keep_set = set()
        for idx in sorted_idx:
            cumulative += probs[idx]
            keep_set.add(int(idx))
            if cumulative >= top_p:
                break
        for i in range(V):
            if i not in keep_set:
                probs[i] = 0.0
        probs = probs / np.sum(probs)

    return int(rng.choice(V, p=probs))


# ============================================================
# 7. main — 학습 + 샘플링
# ============================================================
def main():
    tokenizer = CharTokenizer(TRAINING_TEXT)
    V = tokenizer.vocab_size()
    print("vocab size =", V)

    # 모델 크기 (작게 잡아서 numpy 로도 빠르게)
    SEQ_LEN = 12
    D = 32
    FFN_HIDDEN = 64

    ids = tokenizer.encode(TRAINING_TEXT)
    inputs, targets = make_dataset(ids, SEQ_LEN, stride=1)
    print("총 시퀀스 수 =", len(inputs))

    # 가중치 초기화
    rng = np.random.default_rng(0)
    scale = 0.1
    params = {
        "token_emb": rng.standard_normal((V, D)) * scale,
        "pos_emb":   rng.standard_normal((SEQ_LEN, D)) * scale,
        "W_Q": rng.standard_normal((D, D)) * scale,
        "W_K": rng.standard_normal((D, D)) * scale,
        "W_V": rng.standard_normal((D, D)) * scale,
        "W_O": rng.standard_normal((D, D)) * scale,
        "W_ff1": rng.standard_normal((D, FFN_HIDDEN)) * scale,
        "b_ff1": np.zeros(FFN_HIDDEN),
        "W_ff2": rng.standard_normal((FFN_HIDDEN, D)) * scale,
        "b_ff2": np.zeros(D),
        "W_out": rng.standard_normal((D, V)) * scale,
        "b_out": np.zeros(V),
    }

    causal_mask = make_causal_mask(SEQ_LEN)

    # 학습
    learning_rate = 0.05
    num_epochs = 200
    batch_size = 16

    print()
    print("=" * 60)
    print("[학습] D=%d, FFN_HIDDEN=%d, SEQ_LEN=%d" % (D, FFN_HIDDEN, SEQ_LEN))
    print("=" * 60)

    for epoch in range(num_epochs):
        perm = rng.permutation(len(inputs))
        x_shuf = inputs[perm]
        y_shuf = targets[perm]

        epoch_loss = 0.0
        num_batches = 0

        i = 0
        while i < len(x_shuf):
            X_batch = x_shuf[i : i + batch_size]
            Y_batch = y_shuf[i : i + batch_size]

            logits, cache = forward(X_batch, params, causal_mask)
            loss, grads = loss_and_backward(logits, Y_batch, cache, params)

            for key in params:
                params[key] = params[key] - learning_rate * grads[key]

            epoch_loss += loss
            num_batches += 1
            i += batch_size

        avg_loss = epoch_loss / num_batches
        if (epoch + 1) % 20 == 0:
            print("epoch %3d  avg loss = %.4f  ppl = %.2f"
                  % (epoch + 1, avg_loss, float(np.exp(avg_loss))))

    # ---- 샘플링 ----
    print()
    print("=" * 60)
    print("[샘플링]")
    print("=" * 60)

    def generate(seed_text, num_chars, temperature, top_k=None, top_p=None,
                 sample_rng=None):
        generated = tokenizer.encode(seed_text)
        for _ in range(num_chars):
            # 마지막 SEQ_LEN 개를 입력으로 자른다
            context = generated[-SEQ_LEN:]
            # 길이가 SEQ_LEN 이 안 되면 앞쪽을 0으로 채운다 (간단히)
            while len(context) < SEQ_LEN:
                context = [0] + context
            X_input = np.array([context])
            logits, _ = forward(X_input, params, causal_mask)
            # 우리가 원하는 건 "마지막 위치" 의 다음 토큰 logits
            last_logits = logits[0, -1]                            # (V,)
            next_id = sample_next_id(
                last_logits, sample_rng, temperature, top_k, top_p)
            generated.append(next_id)
        return tokenizer.decode(generated)

    seed = "안녕하세요"
    sample_rng = np.random.default_rng(42)

    print("temperature=1.0:")
    print(generate(seed, 40, temperature=1.0, sample_rng=sample_rng))
    print()
    print("temperature=0.3 (sharp):")
    print(generate(seed, 40, temperature=0.3, sample_rng=sample_rng))
    print()
    print("temperature=1.5 (random):")
    print(generate(seed, 40, temperature=1.5, sample_rng=sample_rng))
    print()
    print("top_k=5:")
    print(generate(seed, 40, temperature=1.0, top_k=5, sample_rng=sample_rng))
    print()
    print("top_p=0.9:")
    print(generate(seed, 40, temperature=1.0, top_p=0.9, sample_rng=sample_rng))


main()


# ============================================================
# 8. 정리
# ============================================================
print()
print("=" * 60)
print("[정리]")
print("=" * 60)
print("- MiniGPT = (token+pos 임베딩) → (Attention + 잔차) → (FFN + 잔차) → 출력층.")
print("- 한 시퀀스에서 모든 위치가 동시에 학습 신호를 만든다.")
print("- 샘플링 옵션:")
print("    temperature : 작으면 과감, 크면 다양.")
print("    top_k       : 상위 k 개만 남기기.")
print("    top_p       : 누적 확률 p 까지만 남기기 (nucleus).")
print("- 다음 주: PyTorch 로 같은 모델 옮기기 + autograd/optimizer 자동화 체험.")
