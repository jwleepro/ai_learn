"""Week 5 Complete: Transformer Forward Pass.

Transformer 한 블록의 forward (학습은 안 함).

흐름:
1. 입력 임베딩 + 위치 임베딩
2. (LayerNorm -> Multi-Head Attention -> 잔차 연결)
3. (LayerNorm -> Feed-Forward -> 잔차 연결)
4. 출력 선형층 -> 다음 글자 logits

Multi-Head Attention 의 핵심:
- D 차원을 H 개 헤드로 나누어 (각 헤드 차원 Dh = D/H) 병렬로 attention.
- 헤드마다 다른 패턴을 학습할 수 있게 됨.

실행 방법:
- 데모: python week5/code/week5_complete.py --input week5/data/tiny_corpus_ko.txt
"""

import argparse
from pathlib import Path
import numpy as np


# ============================================================================
# 1. CharTokenizer + softmax + LayerNorm
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


def layer_norm(x, gain, bias, eps=1e-5):
    """Layer Normalization: 각 행(샘플)의 평균과 분산을 이용해 정규화.

    수식 (한 행에 대해):
        mean = avg(x), var = var(x)
        x_hat = (x - mean) / sqrt(var + eps)
        output = gain * x_hat + bias
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gain * x_hat + bias


# ============================================================================
# 2. Multi-Head Attention (배치 차원 없이 단일 시퀀스)
# ============================================================================

def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads):
    """Multi-Head Attention.

    입력:  x   (T, D)
    출력: out  (T, D)

    내부 단계:
    1) 선형 투영으로 Q/K/V 를 만든다.
    2) D 차원을 H 개 헤드로 쪼갠다 (헤드별 차원 Dh = D / H).
    3) 헤드마다 attention 점수 -> causal mask -> softmax -> 가중평균.
    4) 헤드 결과를 다시 이어붙이고 출력 투영 Wo.
    """
    seq_len = x.shape[0]
    model_dim = x.shape[1]
    head_dim = model_dim // n_heads

    # 1) Q, K, V 만들기 (T, D)
    Q_full = x @ Wq
    K_full = x @ Wk
    V_full = x @ Wv

    # 2) 헤드별 결과를 모아둘 빈 배열 (n_heads, T, Dh)
    Q = np.empty((n_heads, seq_len, head_dim))
    K = np.empty((n_heads, seq_len, head_dim))
    V = np.empty((n_heads, seq_len, head_dim))
    for h in range(n_heads):
        # h번째 헤드는 D 차원 중 [h*Dh : (h+1)*Dh] 구간을 담당
        col_start = h * head_dim
        col_end = (h + 1) * head_dim
        Q[h] = Q_full[:, col_start:col_end]
        K[h] = K_full[:, col_start:col_end]
        V[h] = V_full[:, col_start:col_end]

    # 3) 헤드별로 attention 계산
    attn_outputs = np.empty((n_heads, seq_len, head_dim))
    for h in range(n_heads):
        # (T, Dh) @ (Dh, T) -> (T, T)
        scores = (Q[h] @ K[h].T) / np.sqrt(head_dim)

        # Causal mask: 미래 위치(j > i)는 -1e9 로
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    scores[i, j] = -1e9

        weights = softmax(scores, axis=1)   # (T, T)
        attn_outputs[h] = weights @ V[h]    # (T, Dh)

    # 4) 헤드 결과 이어붙이기 (T, D)
    combined = np.empty((seq_len, model_dim))
    for h in range(n_heads):
        col_start = h * head_dim
        col_end = (h + 1) * head_dim
        combined[:, col_start:col_end] = attn_outputs[h]

    # 출력 투영
    return combined @ Wo


# ============================================================================
# 3. Feed-Forward Network
# ============================================================================

def feed_forward(x, W1, b1, W2, b2):
    """위치별로 동일하게 적용되는 2층 신경망 (선형 -> ReLU -> 선형)."""
    h = x @ W1 + b1
    # ReLU: 음수는 0 으로
    h = np.maximum(0, h)
    return h @ W2 + b2


# ============================================================================
# 4. Main Execution Flow
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="입력 텍스트 파일 경로")
    parser.add_argument("--tokens", type=int, default=32, help="처리할 토큰 수")
    parser.add_argument("--d_model", type=int, default=64, help="모델 차원")
    parser.add_argument("--n_heads", type=int, default=4, help="어텐션 헤드 수")
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        print(f"데이터 파일이 없습니다: {data_path}")
        return
    text = data_path.read_text(encoding="utf-8")
    tokenizer = build_tokenizer_from_text(text)

    ids = tokenizer.encode(text)[:args.tokens]
    seq_len = len(ids)
    model_dim = args.d_model
    n_heads = args.n_heads

    print(f"--- Transformer Forward Demo (T={seq_len}, D={model_dim}, Heads={n_heads}) ---")

    # 1) 파라미터 초기화 (데모용 랜덤 가중치)
    rng = np.random.default_rng(42)
    scale = 0.02

    # 임베딩 + 위치 임베딩
    tok_emb = rng.normal(0, scale, (tokenizer.vocab_size(), model_dim))
    pos_emb = rng.normal(0, scale, (seq_len, model_dim))

    # MHA 파라미터
    Wq = rng.normal(0, scale, (model_dim, model_dim))
    Wk = rng.normal(0, scale, (model_dim, model_dim))
    Wv = rng.normal(0, scale, (model_dim, model_dim))
    Wo = rng.normal(0, scale, (model_dim, model_dim))

    # LayerNorm 파라미터: gain=1, bias=0 으로 시작 (학습 전엔 항등 변환)
    ln1_gain = np.ones(model_dim)
    ln1_bias = np.zeros(model_dim)

    # FFN 파라미터 (보통 hidden 은 4 * D)
    ffn_hidden = model_dim * 4
    W1 = rng.normal(0, scale, (model_dim, ffn_hidden))
    b1 = np.zeros(ffn_hidden)
    W2 = rng.normal(0, scale, (ffn_hidden, model_dim))
    b2 = np.zeros(model_dim)

    ln2_gain = np.ones(model_dim)
    ln2_bias = np.zeros(model_dim)

    # 출력층
    W_out = rng.normal(0, scale, (model_dim, tokenizer.vocab_size()))

    # 2) Forward Pass
    print("\n1) Input Embedding + Positional Encoding")
    x = tok_emb[ids] + pos_emb  # (T, D)

    print("2) Transformer Block: Multi-Head Attention (with residual + LayerNorm)")
    x_norm = layer_norm(x, ln1_gain, ln1_bias)
    attn_out = multi_head_attention(x_norm, Wq, Wk, Wv, Wo, n_heads=n_heads)
    x = x + attn_out  # 잔차 연결

    print("3) Transformer Block: Feed-Forward Network (with residual + LayerNorm)")
    x_norm = layer_norm(x, ln2_gain, ln2_bias)
    ffn_out = feed_forward(x_norm, W1, b1, W2, b2)
    x = x + ffn_out  # 잔차 연결

    print("4) Output Linear Layer")
    logits = x @ W_out  # (T, V)

    # 3) 결과 출력
    last_probs = softmax(logits[-1])
    sorted_indices_desc = np.argsort(last_probs)[::-1]
    top_indices = sorted_indices_desc[:5]

    print(f"\n[마지막 토큰 '{tokenizer.decode([ids[-1]])}' 다음으로 올 확률이 높은 글자]")
    for idx in top_indices:
        print(f"  '{tokenizer.vocab[idx]}': {last_probs[idx]:.4f}")


if __name__ == "__main__":
    main()
