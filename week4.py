"""Week 4 — 임베딩 + 소프트맥스 (신경망 빅램).

이 주에 배우는 것
- 임베딩(embedding): "단어를 벡터로 표현" 하는 방법.
  같은 ID 가 들어오면 같은 벡터가 나오는 lookup table 일 뿐이다.
- 소프트맥스(softmax): 신경망의 raw 출력(logits)을 확률 분포로 바꾸는 함수.
- 이 두 개를 합치면 "신경망 버전 빅램 모델" 이 된다.

이번 주 핵심 아이디어
- 지난 주 빅램 카운트 모델: 확률표(V x V) 를 손으로 만들었다.
- 이번 주 신경망 빅램: 같은 V x V 확률표를 가중치로 두고, 손실을 줄이며 학습한다.
- 학습이 끝나면 두 결과가 거의 똑같아진다는 걸 본다.
- 차이는: 신경망 방식은 그대로 더 큰 모델로 확장 가능하다는 것.

실행:  python week4.py
"""

import numpy as np


# ============================================================
# 0. 데이터와 토크나이저 (week3 와 동일)
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
# 1. 임베딩이란?  —  ID → 벡터 변환표
# ============================================================
# 우리는 글자 ID 를 그냥 정수로 모델에 넣을 수 있을까? 안 된다.
# ID 는 그냥 라벨일 뿐, "ID 5 가 ID 4 보다 두 배 크다" 같은 의미가 없다.
#
# 그래서 각 ID 마다 학습 가능한 벡터를 하나씩 둔다.
#
#   E shape = (V, D)
#       V = vocab size (몇 가지 글자가 있는지)
#       D = embedding dim (한 글자를 몇 차원 벡터로 표현할지)
#
# Java 비유: double[V][D] 2차원 배열.
# E[id] 한 줄을 빼오면 그 글자의 벡터다.
#
# 이번 주에서는 신경망 빅램이라 D 는 신경망의 hidden dim 처럼 쓴다.

# 사실 신경망 빅램은 임베딩만 보여주려고 일부러 단순하게 만들 수도 있는데,
# 가장 정직한 형태는 이렇다:
#
#   logits = E[prev_id] @ W       # shape: (D,) @ (D, V) = (V,)
#   probs  = softmax(logits)
#
# 즉 두 행렬: E (V x D), W (D x V) 가 학습 가능한 가중치.
# 사실 이건 "행렬 곱 두 번" 일 뿐이다. ReLU 같은 비선형도 안 끼인다.
# 그래도 "신경망의 핵심 부품" 인 임베딩 + softmax + 손실 + 경사하강 흐름을 모두 본다.


# ============================================================
# 2. softmax (week1 에서 본 그대로)
# ============================================================
def softmax(z):
    z_safe = z - np.max(z)
    e = np.exp(z_safe)
    return e / np.sum(e)


# ============================================================
# 3. forward — 한 글자가 들어오면 다음 글자 확률을 뽑는다
# ============================================================
def forward_one(prev_id, E, W):
    # E[prev_id] : (D,)  ← 그 글자의 임베딩 벡터
    # W          : (D, V)
    # logits     : (V,)
    embedding = E[prev_id]
    logits = embedding @ W       # 행렬곱
    probs = softmax(logits)
    return probs, logits, embedding


# ============================================================
# 4. 손실(loss) — 크로스엔트로피의 한 줄 짜리 형태
# ============================================================
# 정답이 next_id 일 때, 모델이 정답에 매긴 확률 p_correct.
# loss = -log(p_correct)
#   p_correct 가 1.0 (확신) 이면 loss = 0
#   p_correct 가 거의 0   이면 loss = 매우 큰 수
#
# 이게 바로 "정답에 더 큰 확률을 주도록 학습" 하는 신호.
# 자세한 유도와 미분은 다음 주(week5) 에서 다룬다.

def cross_entropy_loss(probs, target_id):
    # 1e-12 는 log(0) 방지용 아주 작은 수
    return -np.log(probs[target_id] + 1e-12)


# ============================================================
# 5. backward — 정답이 무엇이었는지 알려주고 가중치를 어떻게 옮길지 계산
# ============================================================
# 핵심 한 줄 (다음 주에 손으로 유도한다):
#
#   d(loss)/d(logits) = probs - one_hot(target)
#
# 즉 "맞은 칸은 -1 하고 나머지는 0" 인 벡터를 probs 에 더한 것.
# 이걸로 W, E 의 기울기를 구한다.
#
# logits = embedding @ W  (shape: (D,) @ (D, V) = (V,))
# 행렬곱의 미분 규칙으로:
#   d_W     = outer(embedding, d_logits)    shape (D, V)
#   d_embed = W @ d_logits                  shape (D,)
# d_E 는 prev_id 행만 d_embed 로 채우고 나머지는 0.

def gradients(prev_id, target_id, E, W):
    embedding = E[prev_id]
    logits = embedding @ W
    probs = softmax(logits)

    d_logits = probs.copy()
    d_logits[target_id] -= 1.0       # (V,)

    d_W = np.outer(embedding, d_logits)              # (D, V)

    d_E = np.zeros_like(E)
    d_E[prev_id] = W @ d_logits                       # 그 ID 행만 갱신

    loss = -np.log(probs[target_id] + 1e-12)
    return loss, d_E, d_W


# ============================================================
# 6. 학습 루프
# ============================================================
def main():
    tokenizer = CharTokenizer(TRAINING_TEXT)
    V = tokenizer.vocab_size()
    D = 16   # 임베딩 차원

    print("=" * 60)
    print("[설정] V=%d, D=%d" % (V, D))
    print("=" * 60)

    # 가중치 초기화 (작은 무작위 값)
    rng = np.random.default_rng(0)
    E = rng.standard_normal((V, D)) * 0.1
    W = rng.standard_normal((D, V)) * 0.1

    # 학습 데이터 = (prev_id, next_id) 쌍의 리스트
    ids = tokenizer.encode(TRAINING_TEXT)
    pairs = []
    for i in range(len(ids) - 1):
        pairs.append((ids[i], ids[i + 1]))

    print("학습 쌍 개수 =", len(pairs))

    learning_rate = 0.5
    num_epochs = 200

    for epoch in range(num_epochs):
        # 한 epoch = 전체 쌍을 한 번씩 돌기
        # 단순하게 SGD 1개씩 — 실제론 미니배치를 쓰지만 명시성을 위해.
        total_loss = 0.0
        for (prev_id, target_id) in pairs:
            loss, d_E, d_W = gradients(prev_id, target_id, E, W)
            total_loss += loss

            # 가중치 갱신
            E = E - learning_rate * d_E
            W = W - learning_rate * d_W

        avg_loss = total_loss / len(pairs)

        if (epoch + 1) % 20 == 0:
            print("epoch %3d  avg loss = %.4f" % (epoch + 1, avg_loss))

    # ========================================================
    # 7. 결과 비교: 신경망 빅램 vs 카운트 빅램 (week3)
    # ========================================================
    print()
    print("=" * 60)
    print("[비교] '안' 다음 글자 확률 top 5")
    print("=" * 60)

    target_char = "안"
    target_id = tokenizer.char_to_id[target_char]

    # 신경망 빅램의 확률
    probs, _, _ = forward_one(target_id, E, W)

    # 카운트 빅램의 확률 (같은 데이터로 즉석에서 계산)
    counts = np.zeros((V, V))
    for i in range(len(ids) - 1):
        counts[ids[i]][ids[i + 1]] += 1
    counts_smoothed = counts + 1.0
    count_probs = counts_smoothed / counts_smoothed.sum(axis=1, keepdims=True)

    top5 = np.argsort(probs)[::-1][:5]

    print("%-6s  %-10s  %-10s" % ("글자", "신경망", "카운트"))
    for idx in top5:
        ch = tokenizer.id_to_char[idx]
        ch_disp = ch if ch.strip() else repr(ch)  # 공백/개행 보기 좋게
        print("%-6s  %-10.4f  %-10.4f" %
              (ch_disp, probs[idx], count_probs[target_id][idx]))

    print("→ 신경망은 학습 데이터에 더 sharp 하게 맞춘다 (카운트는 +1 스무딩으로 평평).")

    # ========================================================
    # 8. 텍스트 생성
    # ========================================================
    print()
    print("=" * 60)
    print("[8] 신경망 빅램으로 텍스트 생성")
    print("=" * 60)

    sample_rng = np.random.default_rng(42)
    seed_text = "안녕"
    output = list(seed_text)
    current_id = tokenizer.char_to_id[seed_text[-1]]

    for _ in range(40):
        probs, _, _ = forward_one(current_id, E, W)
        next_id = int(sample_rng.choice(V, p=probs))
        output.append(tokenizer.id_to_char[next_id])
        current_id = next_id

    print("".join(output))


main()


# ============================================================
# 9. 정리
# ============================================================
print()
print("=" * 60)
print("[정리]")
print("=" * 60)
print("- 임베딩 = ID → 학습 가능한 벡터 변환표 (V x D 행렬).")
print("- 신경망 빅램 = 임베딩 → 행렬곱 → softmax → 확률.")
print("- 학습 신호: d_logits = probs - one_hot(target).")
print("- 다음 주: 손실/역전파를 한 줄씩 손으로 유도해본다.")
