"""Week 6 — MLP 언어모델 (컨텍스트 확장 + 학습/검증 분리 + 퍼플렉서티).

이 주에 배우는 것
- 컨텍스트 윈도우(context window): 직전 1글자가 아니라 직전 N글자로 다음을 예측.
- 학습/검증 데이터 분리(train/val split): 진짜 일반화가 되는지 확인하는 방법.
- 퍼플렉서티(perplexity, PPL): 언어모델 품질의 표준 지표.
- 미니배치(mini-batch): 하나씩이 아니라 묶어서 학습 → 빠르고 안정적.

모델 구조 (week4 보다 한 단계 큰 신경망)
   prev N개 글자 ID → 임베딩 N개 (이어붙이기) → 은닉층(ReLU) → 출력층 → softmax

이번 주의 가장 중요한 메시지
- 신경망의 "층(layer)" 을 한 개 더 쌓는 것 자체는 코드 몇 줄 차이일 뿐이다.
- 진짜 어려운 건 "학습이 잘 되는지 어떻게 알지?" 같은 절차다.

실행:  python week6.py
"""

import numpy as np


# ============================================================
# 0. 데이터와 토크나이저
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
# 1. 컨텍스트 윈도우 = N 개 글자로 다음 글자 예측
# ============================================================
# CONTEXT_LEN = 3 이면:
#
#   "안녕하세요" 에서 만들어지는 학습 쌍:
#       ("안녕하", "세")
#       ("녕하세", "요")
#
# 입력은 길이 N 정수 배열, 출력은 정수 1개.
# 데이터 만드는 함수:

def make_dataset(ids, context_len):
    inputs = []
    targets = []
    for i in range(len(ids) - context_len):
        # 입력: ids[i : i+context_len]   ← 길이 N
        # 정답: ids[i + context_len]     ← 그 다음 1개
        window = []
        for k in range(context_len):
            window.append(ids[i + k])
        inputs.append(window)
        targets.append(ids[i + context_len])
    return np.array(inputs), np.array(targets)


# ============================================================
# 2. 모델 — MLP 한 번
# ============================================================
# 가중치 4종:
#   E       (V, D)         글자 임베딩
#   W1, b1  (D*N, H), (H,) 은닉층 (입력은 N개 임베딩을 이어붙인 것)
#   W2, b2  (H, V), (V,)   출력층
#
# Forward:
#   for each sample:
#     embed_concat = concat( E[id_1], E[id_2], ..., E[id_N] )   shape (D*N,)
#     hidden       = relu( embed_concat @ W1 + b1 )              shape (H,)
#     logits       = hidden @ W2 + b2                            shape (V,)
#
# 이걸 한 번에 batch 처리하기 위해 첫 번째 축에 batch 차원을 둔다.

def relu(z):
    return np.maximum(0.0, z)


def softmax_rows(Z):
    # Z shape (B, V) — 각 행에 softmax
    Z_safe = Z - np.max(Z, axis=1, keepdims=True)
    e = np.exp(Z_safe)
    return e / np.sum(e, axis=1, keepdims=True)


def forward_batch(X, params):
    # X shape (B, N) ← B 개 샘플, 각각 길이 N 의 ID 시퀀스
    E = params["E"]
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    B = X.shape[0]
    N = X.shape[1]
    D = E.shape[1]

    # 1) 임베딩 lookup + concat
    #    명시적 for 루프로 한 샘플씩 채워넣음 (advanced indexing 회피).
    embed_concat = np.zeros((B, N * D))
    for i in range(B):
        for k in range(N):
            token_id = X[i, k]
            # k 번째 임베딩을 [k*D : (k+1)*D] 자리에 복사
            embed_concat[i, k * D : (k + 1) * D] = E[token_id]

    # 2) 은닉층
    pre_hidden = embed_concat @ W1 + b1     # (B, H)
    hidden = relu(pre_hidden)               # (B, H)

    # 3) 출력층
    logits = hidden @ W2 + b2               # (B, V)

    cache = {
        "X": X,
        "embed_concat": embed_concat,
        "pre_hidden": pre_hidden,
        "hidden": hidden,
        "logits": logits,
    }
    return logits, cache


# ============================================================
# 3. 손실(batch 평균 cross-entropy) + backward
# ============================================================
def loss_and_backward(logits, targets, cache, params):
    # logits  shape (B, V)
    # targets shape (B,)
    B = logits.shape[0]
    V = logits.shape[1]

    probs = softmax_rows(logits)

    # cross-entropy 평균
    # 각 샘플의 정답 칸 확률을 명시적으로 모은다.
    correct_probs = np.zeros(B)
    for i in range(B):
        correct_probs[i] = probs[i, targets[i]]
    loss = -np.mean(np.log(correct_probs + 1e-12))

    # ===== Backward =====
    # week5 에서 배운 그대로:
    #   d_logits = (probs - one_hot(target)) / B
    d_logits = probs.copy()
    for i in range(B):
        d_logits[i, targets[i]] -= 1.0
    d_logits = d_logits / B    # batch 평균이니 B로 나눠준다

    hidden = cache["hidden"]
    pre_hidden = cache["pre_hidden"]
    embed_concat = cache["embed_concat"]
    X = cache["X"]

    # 출력층 W2, b2
    d_W2 = hidden.T @ d_logits                        # (H, V)
    d_b2 = np.sum(d_logits, axis=0)                   # (V,)
    d_hidden = d_logits @ params["W2"].T              # (B, H)

    # ReLU 의 미분: 양수면 1, 음수면 0
    d_pre_hidden = d_hidden * (pre_hidden > 0)        # (B, H)

    # 은닉층 W1, b1
    d_W1 = embed_concat.T @ d_pre_hidden              # (D*N, H)
    d_b1 = np.sum(d_pre_hidden, axis=0)               # (H,)
    d_embed_concat = d_pre_hidden @ params["W1"].T    # (B, D*N)

    # 임베딩 E 로 거꾸로 흘려보내기
    # 같은 ID 가 여러 번 들어왔으면 그라디언트도 += 로 누적된다.
    E = params["E"]
    D = E.shape[1]
    N = X.shape[1]
    d_E = np.zeros_like(E)
    for i in range(B):
        for k in range(N):
            token_id = X[i, k]
            d_E[token_id] += d_embed_concat[i, k * D : (k + 1) * D]

    grads = {"E": d_E, "W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}
    return loss, grads


# ============================================================
# 4. 퍼플렉서티(PPL)
# ============================================================
# PPL = exp(평균 cross-entropy)
#
# 직관: "이 모델이 다음 글자를 고를 때 평균적으로 몇 개의 후보를 두고 헷갈리고 있는가."
# 작을수록 좋다. PPL = 5 면 "5지선다" 수준의 불확실성이라는 뜻.

def perplexity(loss):
    return float(np.exp(loss))


# ============================================================
# 5. 학습 루프
# ============================================================
def main():
    # ----- 토크나이즈 -----
    tokenizer = CharTokenizer(TRAINING_TEXT)
    V = tokenizer.vocab_size()
    print("vocab size =", V)

    # ----- 데이터 -----
    CONTEXT_LEN = 4
    EMBED_DIM = 16
    HIDDEN_DIM = 64

    ids = tokenizer.encode(TRAINING_TEXT)
    inputs, targets = make_dataset(ids, CONTEXT_LEN)
    print("총 샘플 수 =", len(inputs))

    # ----- train / val 분리 -----
    # 마지막 15% 를 검증용으로 떼어둔다.
    split_index = int(len(inputs) * 0.85)
    X_train = inputs[:split_index]
    Y_train = targets[:split_index]
    X_val = inputs[split_index:]
    Y_val = targets[split_index:]
    print("train =", len(X_train), "  val =", len(X_val))

    # ----- 가중치 초기화 -----
    rng = np.random.default_rng(0)
    params = {
        "E":  rng.standard_normal((V, EMBED_DIM)) * 0.1,
        "W1": rng.standard_normal((CONTEXT_LEN * EMBED_DIM, HIDDEN_DIM)) * 0.1,
        "b1": np.zeros(HIDDEN_DIM),
        "W2": rng.standard_normal((HIDDEN_DIM, V)) * 0.1,
        "b2": np.zeros(V),
    }

    # ----- 학습 -----
    learning_rate = 0.1
    num_epochs = 300
    batch_size = 32

    print()
    print("=" * 60)
    print("학습 시작 (context_len=%d, embed=%d, hidden=%d)" %
          (CONTEXT_LEN, EMBED_DIM, HIDDEN_DIM))
    print("=" * 60)

    for epoch in range(num_epochs):
        # 한 epoch = train 데이터 한 번 훑기, 미니배치로 나눠서.
        # 매 epoch 마다 인덱스를 셔플 → 같은 패턴에 갇히는 걸 방지.
        perm = rng.permutation(len(X_train))
        X_shuffled = X_train[perm]
        Y_shuffled = Y_train[perm]

        i = 0
        while i < len(X_shuffled):
            X_batch = X_shuffled[i : i + batch_size]
            Y_batch = Y_shuffled[i : i + batch_size]

            logits, cache = forward_batch(X_batch, params)
            loss, grads = loss_and_backward(logits, Y_batch, cache, params)

            # 가중치 갱신
            for key in params:
                params[key] = params[key] - learning_rate * grads[key]

            i = i + batch_size

        # 매 50 epoch 마다 train/val 손실과 PPL 출력
        if (epoch + 1) % 50 == 0:
            train_logits, _ = forward_batch(X_train, params)
            train_loss, _ = loss_and_backward(
                train_logits, Y_train, _make_cache(X_train, params), params)
            val_logits, _ = forward_batch(X_val, params)
            val_loss, _ = loss_and_backward(
                val_logits, Y_val, _make_cache(X_val, params), params)
            print("epoch %3d  train_loss=%.4f  train_ppl=%.2f  val_loss=%.4f  val_ppl=%.2f"
                  % (epoch + 1, train_loss, perplexity(train_loss),
                     val_loss, perplexity(val_loss)))

    # ----- 최종 평가 -----
    print()
    print("=" * 60)
    print("[최종 평가]")
    print("=" * 60)
    train_logits, _ = forward_batch(X_train, params)
    train_loss, _ = loss_and_backward(
        train_logits, Y_train, _make_cache(X_train, params), params)
    val_logits, _ = forward_batch(X_val, params)
    val_loss, _ = loss_and_backward(
        val_logits, Y_val, _make_cache(X_val, params), params)
    print("train: loss=%.4f  ppl=%.2f" % (train_loss, perplexity(train_loss)))
    print("val:   loss=%.4f  ppl=%.2f" % (val_loss, perplexity(val_loss)))
    print("→ val_loss 가 train_loss 보다 너무 크면 과적합 신호.")

    # ----- 텍스트 생성 -----
    print()
    print("=" * 60)
    print("[생성] '안녕하세' 다음 30글자")
    print("=" * 60)

    sample_rng = np.random.default_rng(7)
    seed = "안녕하세"
    generated_ids = tokenizer.encode(seed)

    for _ in range(30):
        # 마지막 CONTEXT_LEN 개를 입력으로
        last_n = generated_ids[-CONTEXT_LEN:]
        X_input = np.array([last_n])
        logits, _ = forward_batch(X_input, params)
        probs = softmax_rows(logits)[0]   # (V,)
        next_id = int(sample_rng.choice(V, p=probs))
        generated_ids.append(next_id)

    print(tokenizer.decode(generated_ids))


def _make_cache(X, params):
    # 평가용 forward 만 다시 돌려 cache 만들기 (loss_and_backward 가 cache 를 요구하므로).
    _, cache = forward_batch(X, params)
    return cache


main()


# ============================================================
# 6. 정리
# ============================================================
print()
print("=" * 60)
print("[정리]")
print("=" * 60)
print("- 컨텍스트 N개로 다음 1개 예측 → 빅램보다 훨씬 자연스러운 텍스트.")
print("- train/val 분리로 '진짜 일반화' 를 측정한다.")
print("- 퍼플렉서티 = exp(loss). 작을수록 좋다.")
print("- 미니배치 = 학습이 빠르고 안정적이다.")
print("- 다음 주: 컨텍스트가 길어질수록 문제가 되는 것을 attention 으로 푼다.")
