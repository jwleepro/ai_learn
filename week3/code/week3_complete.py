"""Week 3 Complete: MLP 언어모델.

MLP(Multi-Layer Perceptron) 언어모델: "직전 C 개의 글자(컨텍스트)" 를 보고 "다음 글자" 를 예측.

구조 (입력 -> 출력 순서):
1. Embedding:  각 글자 ID -> D차원 벡터  (룩업)
2. Flatten:    C개 벡터를 이어붙여 (C*D) 차원 벡터로
3. Hidden:     (C*D -> H) 선형 변환 + tanh
4. Output:     (H -> V) 선형 변환 -> logits
5. Softmax:    logits -> 다음 글자 확률

학습은 cross entropy loss 를 줄이는 방향으로 모든 파라미터를 경사하강법으로 갱신.

실행 방법:
- 학습: python week3/code/week3_complete.py --train
- 생성: python week3/code/week3_complete.py --generate
"""

import argparse
from pathlib import Path
import numpy as np


# ============================================================================
# 1. CharTokenizer (week1 과 동일)
# ============================================================================

class CharTokenizer:
    def __init__(self, vocab):
        if len(vocab) == 0:
            raise ValueError("vocab empty")
        self.vocab = vocab

        self.char_to_id = {}
        for i in range(len(vocab)):
            self.char_to_id[vocab[i]] = i

    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        ids = []
        for ch in text:
            ids.append(self.char_to_id[ch])
        return ids

    def decode(self, ids):
        chars = []
        for token_id in ids:
            chars.append(self.vocab[token_id])
        return "".join(chars)


def build_tokenizer_from_text(text):
    return CharTokenizer(sorted(set(text)))


def softmax(logits, axis=-1):
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


# ============================================================================
# 2. 데이터셋 만들기
# ============================================================================

def make_context_dataset(token_ids, context_len):
    """슬라이딩 윈도우로 (입력 컨텍스트, 다음 글자) 쌍을 만든다.

    예) token_ids = [1, 2, 3, 4, 5], context_len = 3 이면
        X[0] = [1,2,3], y[0] = 4
        X[1] = [2,3,4], y[1] = 5
    """
    n = len(token_ids) - context_len
    X = np.empty((n, context_len), dtype=np.int64)
    y = np.empty((n,), dtype=np.int64)

    for i in range(n):
        # 슬라이싱: token_ids[i : i + context_len] = i 번째부터 context_len 개
        for k in range(context_len):
            X[i, k] = token_ids[i + k]
        y[i] = token_ids[i + context_len]

    return X, y


# ============================================================================
# 3. 샘플링 (top-k / top-p)
# ============================================================================

def sample_from_probs(probs, rng, temperature=1.0, top_k=None, top_p=None):
    """확률 분포에서 다음 토큰 ID 한 개를 뽑는다.

    - temperature: 1.0 보다 작으면 확률이 큰 쪽에 더 쏠림(보수적), 크면 평평해짐(창의적)
    - top_k: 상위 K개 글자만 후보로 두기
    - top_p: 누적확률 P 가 될 때까지의 글자만 후보로 두기 (nucleus sampling)
    """
    p = probs.copy()

    if temperature != 1.0:
        p = np.power(p, 1.0 / temperature)
        p = p / p.sum()

    if top_k is not None:
        # 상위 K번째 값의 임계치
        threshold = np.partition(p, -top_k)[-top_k]
        # 임계치보다 작은 값은 0으로
        for i in range(len(p)):
            if p[i] < threshold:
                p[i] = 0.0
        p = p / p.sum()

    if top_p is not None:
        # 확률이 큰 순서대로 인덱스 정렬 (내림차순)
        sorted_indices = np.argsort(p)[::-1]
        sorted_probs = p[sorted_indices]

        # 누적 합
        cumulative = np.cumsum(sorted_probs)

        # 누적이 top_p 를 넘는 위치부터는 후보에서 제외 (단 첫 글자는 항상 살림)
        new_p = np.zeros_like(p)
        for rank in range(len(sorted_indices)):
            if rank == 0 or cumulative[rank - 1] < top_p:
                new_p[sorted_indices[rank]] = sorted_probs[rank]
        p = new_p / new_p.sum()

    return int(rng.choice(len(p), p=p))


# ============================================================================
# 4. MLP 모델 (NumPy 직접 구현)
# ============================================================================

class MLPLMParams:
    """모델의 모든 학습 가능한 파라미터를 담는 컨테이너."""

    def __init__(self, E, W1, b1, W2, b2):
        self.E = E    # Embedding table  (V, D)
        self.W1 = W1  # Hidden weights   (C*D, H)
        self.b1 = b1  # Hidden bias      (H,)
        self.W2 = W2  # Output weights   (H, V)
        self.b2 = b2  # Output bias      (V,)


def init_params(vocab_size, context_len, embed_dim, hidden_dim, rng):
    scale = 0.02
    E = rng.normal(0, scale, (vocab_size, embed_dim))
    W1 = rng.normal(0, scale, (context_len * embed_dim, hidden_dim))
    b1 = np.zeros(hidden_dim)
    W2 = rng.normal(0, scale, (hidden_dim, vocab_size))
    b2 = np.zeros(vocab_size)
    return MLPLMParams(E, W1, b1, W2, b2)


def forward(params, X):
    """순전파: X (B, C) -> logits (B, V)

    중간 계산값(cache)은 역전파 때 다시 써야 하므로 같이 돌려준다.
    """
    # 1) 임베딩 룩업: 각 토큰 ID 를 D차원 벡터로 (B, C, D)
    emb = params.E[X]

    # 2) Flatten: (B, C, D) -> (B, C*D)
    h_in = emb.reshape(len(X), -1)

    # 3) 은닉층: 선형 + tanh
    h_pre = h_in @ params.W1 + params.b1   # `@` 는 행렬곱 (B, H)
    h = np.tanh(h_pre)

    # 4) 출력층: 선형 -> logits (B, V)
    logits = h @ params.W2 + params.b2

    cache = {"X": X, "emb": emb, "h_in": h_in, "h_pre": h_pre, "h": h}
    return logits, cache


def train_step(params, X, y, lr):
    """학습 한 스텝: forward -> loss -> backward -> 파라미터 업데이트."""
    batch_size = len(X)

    # 1) Forward
    logits, cache = forward(params, X)
    probs = softmax(logits, axis=1)

    # 2) Cross Entropy Loss
    correct_probs = np.zeros(batch_size)
    for i in range(batch_size):
        correct_probs[i] = probs[i, y[i]]
    loss = -np.log(correct_probs + 1e-10).mean()

    # 3) Backprop
    # softmax + cross entropy 의 잘 알려진 결과: dL/dlogits = (probs - one_hot) / B
    dlogits = probs.copy()
    for i in range(batch_size):
        dlogits[i, y[i]] -= 1.0
    dlogits = dlogits / batch_size

    # 출력층 (logits = h @ W2 + b2)
    dW2 = cache["h"].T @ dlogits           # (H, V)
    db2 = dlogits.sum(axis=0)              # (V,)

    # 은닉층 역전파
    dh = dlogits @ params.W2.T             # (B, H)
    # tanh 미분: d(tanh(x))/dx = 1 - tanh(x)^2
    dh_pre = dh * (1.0 - cache["h"] ** 2)  # (B, H)

    dW1 = cache["h_in"].T @ dh_pre         # (C*D, H)
    db1 = dh_pre.sum(axis=0)               # (H,)

    # Flatten 의 역연산: (B, C*D) -> (B, C, D)
    dh_in = dh_pre @ params.W1.T
    dEmb = dh_in.reshape(cache["emb"].shape)

    # 임베딩 그래디언트: 같은 토큰 ID 가 여러 번 나오면 모두 더해야 한다.
    dE = np.zeros_like(params.E)
    context_len = X.shape[1]
    for i in range(batch_size):
        for k in range(context_len):
            token_id = X[i, k]
            dE[token_id] += dEmb[i, k]

    # 4) 파라미터 업데이트 (경사하강법)
    params.W2 -= lr * dW2
    params.b2 -= lr * db2
    params.W1 -= lr * dW1
    params.b1 -= lr * db1
    params.E -= lr * dE

    return float(loss)


# ============================================================================
# 5. Main Execution Flow
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="학습 실행")
    parser.add_argument("--generate", action="store_true", help="생성 실행")
    parser.add_argument("--data", default="week3/data/tiny_corpus_ko.txt", help="데이터 경로")
    parser.add_argument("--model_path", default="week3/code/mlp_model.npz", help="모델 저장/불러오기 경로")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"데이터 파일이 없습니다: {data_path}")
        return
    text = data_path.read_text(encoding="utf-8")
    tokenizer = build_tokenizer_from_text(text)
    ids = np.array(tokenizer.encode(text))

    # 하이퍼파라미터
    context_len = 8
    embed_dim = 32
    hidden_dim = 128
    learning_rate = 0.1
    epochs = 10  # 교육용이라 짧게

    if args.train:
        print(f"--- 학습 시작 (Vocab: {tokenizer.vocab_size()}, Context: {context_len}) ---")
        X, y = make_context_dataset(ids, context_len)
        rng = np.random.default_rng(42)
        params = init_params(
            vocab_size=tokenizer.vocab_size(),
            context_len=context_len,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            rng=rng,
        )

        for epoch in range(epochs):
            # 단순화를 위해 배치 분할 없이 한 번에 처리
            loss = train_step(params, X, y, learning_rate)
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        np.savez(
            args.model_path,
            E=params.E, W1=params.W1, b1=params.b1, W2=params.W2, b2=params.b2,
            vocab=tokenizer.vocab,
            context_len=context_len,
        )
        print(f"모델 저장 완료: {args.model_path}")

    elif args.generate:
        if not Path(args.model_path).exists():
            print(f"모델 파일이 없습니다. 먼저 --train 을 실행하세요: {args.model_path}")
            return

        ckpt = np.load(args.model_path, allow_pickle=True)
        vocab = list(ckpt["vocab"])
        tokenizer = CharTokenizer(vocab)
        params = MLPLMParams(ckpt["E"], ckpt["W1"], ckpt["b1"], ckpt["W2"], ckpt["b2"])
        context_len = int(ckpt["context_len"])

        print("--- 텍스트 생성 시작 ---")
        rng = np.random.default_rng()

        # 시작 컨텍스트: 데이터의 앞부분
        context = []
        for k in range(context_len):
            context.append(int(ids[k]))
        print(f"Seed context: '{tokenizer.decode(context)}'")

        generated = []
        for _ in range(100):
            # 최근 context_len 개로 입력 만들기 (모양: 1 x C)
            recent = context[-context_len:]
            x = np.array([recent])

            logits, _ = forward(params, x)
            probs = softmax(logits[0])
            next_id = sample_from_probs(probs, rng, temperature=0.8)

            generated.append(next_id)
            context.append(next_id)

        print(f"Generated: {tokenizer.decode(generated)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
