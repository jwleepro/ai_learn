"""Week 4 Complete: Self-Attention (Single Head).

Self-Attention 의 핵심 아이디어:
- 각 토큰이 "다른 토큰들 중 누구를 얼마나 참고할지" 를 학습하는 메커니즘.
- 입력 X (T 개의 토큰 임베딩) 를 세 가지로 변환:
    Q = X @ Wq  (질문, Query)
    K = X @ Wk  (키,   Key)
    V = X @ Wv  (값,   Value)
- 점수: scores = Q @ K^T / sqrt(Dh)  ← "내가 누구와 얼마나 비슷한가"
- causal mask: 미래 토큰을 -무한대로 가려서 못 보게 함 (언어모델 가정)
- 가중치: weights = softmax(scores)
- 출력: out = weights @ V  ← 가중평균

실행 방법:
- 데모: python week4/code/week4_complete.py --input week4/data/tiny_corpus_ko.txt
"""

import argparse
from pathlib import Path
import numpy as np


# ============================================================================
# 1. CharTokenizer + softmax
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
# 2. Self-Attention
# ============================================================================

def make_causal_mask(seq_len):
    """미래 위치를 가리는 (T, T) bool 마스크.

    예) T=3 이면
        [[F, T, T],
         [F, F, T],
         [F, F, F]]
    True 인 자리는 "보면 안 되는 미래" → 점수를 -무한대로 만든다.
    """
    mask = np.zeros((seq_len, seq_len), dtype=bool)
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                mask[i, j] = True
    return mask


def apply_causal_mask(scores):
    """scores 의 미래 위치를 -1e9 (사실상 -무한대) 로 만든다."""
    seq_len = scores.shape[0]
    mask = make_causal_mask(seq_len)

    masked = scores.copy()
    for i in range(seq_len):
        for j in range(seq_len):
            if mask[i, j]:
                masked[i, j] = -1e9
    return masked


def self_attention(X, Wq, Wk, Wv, causal=True):
    """단일 헤드 Self-Attention.

    입력 모양: X (T, D), Wq/Wk/Wv (D, Dh)
    출력: weights (T, T), out (T, Dh)
    """
    # `@` 는 행렬곱.
    Q = X @ Wq  # (T, Dh)
    K = X @ Wk  # (T, Dh)
    V = X @ Wv  # (T, Dh)

    head_dim = Q.shape[1]
    # K.T 는 K 의 transpose. 결과: (T, T) 점수 행렬.
    scores = (Q @ K.T) / np.sqrt(head_dim)

    if causal:
        scores = apply_causal_mask(scores)

    # 행마다 softmax: weights[i] 는 "i번째 토큰이 각 토큰을 얼마나 볼지" 의 분포
    weights = softmax(scores, axis=1)

    # 가중치와 V 의 행렬곱 → 가중평균된 표현 (T, Dh)
    out = weights @ V

    return weights, out


# ============================================================================
# 3. Main Execution Flow
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="입력 텍스트 파일 경로")
    parser.add_argument("--tokens", type=int, default=20, help="분석할 토큰 수")
    parser.add_argument("--pos", type=int, default=-1, help="분석할 특정 위치 (기본: 마지막)")
    parser.add_argument("--no_causal", action="store_true", help="Causal Masking 비활성화")
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        print(f"데이터 파일이 없습니다: {data_path}")
        return
    text = data_path.read_text(encoding="utf-8")

    # 1) 토크나이저 + 임베딩 준비 (데모를 위해 랜덤 임베딩 사용)
    tokenizer = build_tokenizer_from_text(text)
    full_ids = tokenizer.encode(text)

    seq_len = min(args.tokens, len(full_ids))
    ids = full_ids[:seq_len]  # 앞에서 seq_len 개만 사용

    embed_dim = 16
    head_dim = 16
    rng = np.random.default_rng(42)

    # 가상의 임베딩 테이블 + 투영 행렬
    E = rng.normal(0, 0.1, (tokenizer.vocab_size(), embed_dim))

    # X[i] = E[ids[i]] 와 같음 (numpy fancy indexing)
    X = E[ids]  # (T, D)

    Wq = rng.normal(0, 0.1, (embed_dim, head_dim))
    Wk = rng.normal(0, 0.1, (embed_dim, head_dim))
    Wv = rng.normal(0, 0.1, (embed_dim, head_dim))

    # 2) Self-Attention 계산
    causal = not args.no_causal
    weights, out = self_attention(X, Wq, Wk, Wv, causal=causal)

    # 3) 결과 출력
    pos = args.pos if args.pos >= 0 else seq_len - 1
    print(f"--- Self-Attention Demo (T={seq_len}, Causal={causal}) ---")
    print(f"분석 위치 [{pos}]: '{tokenizer.decode([ids[pos]])}'")

    print("\nAttention 가중치 (상위 5개):")
    row = weights[pos]
    # 큰 값부터 정렬한 인덱스 (앞 5개)
    sorted_indices_desc = np.argsort(row)[::-1]
    top_indices = sorted_indices_desc[:5]
    for idx in top_indices:
        target_char = tokenizer.decode([ids[idx]])
        print(f"  to [{idx:2d}] '{target_char}': {row[idx]:.4f}")

    print("\n[전체 문맥]")
    print(f"'{tokenizer.decode(ids)}'")


if __name__ == "__main__":
    main()
