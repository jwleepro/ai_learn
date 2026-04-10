"""Week 5 Complete: Transformer Forward Pass.

이 파일은 트랜스포머(Transformer) 모델의 핵심 구조와 Forward Pass 과정을
하나의 파일에서 순서대로 읽고 실행할 수 있도록 통합한 교육용 코드입니다.

주요 내용:
1. CharTokenizer: 글자 단위 토크나이저
2. Multi-Head Attention (MHA): 병렬 어텐션 계산
3. Feed-Forward Network (FFN): 비선형 변환 층
4. Transformer Block: MHA와 FFN을 결합한 기본 단위 (Residual & LayerNorm 포함)
5. Forward Pass: 입력 토큰으로부터 최종 예측 점수(Logits)까지의 전체 경로

실행 방법:
- 데모: python week5/code/week5_complete.py --input week5/data/tiny_corpus_ko.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np


# ============================================================================
# 1. Utility Functions & Classes
# ============================================================================

@dataclass
class CharTokenizer:
    """글자 단위 토크나이저."""
    vocab: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.vocab) == 0: raise ValueError("vocab empty")
        self.char_to_id = {ch: i for i, ch in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int: return len(self.vocab)

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls(tuple(sorted(set(text))))

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer Normalization: 각 샘플(행)의 평균과 분산을 이용해 정규화합니다."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return g * x_hat + b


# ============================================================================
# 2. Transformer Components Implementation
# ============================================================================

def multi_head_attention(x: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, Wo: np.ndarray, n_heads: int) -> np.ndarray:
    """Multi-Head Attention: 입력을 여러 헤드로 나누어 어텐션을 병렬로 계산합니다."""
    T, D = x.shape
    Dh = D // n_heads # Head dimension
    
    # Q, K, V 투영 및 헤드 분리
    # (T, D) @ (D, D) -> (T, D) -> (T, n_heads, Dh) -> (n_heads, T, Dh)
    Q = (x @ Wq).reshape(T, n_heads, Dh).transpose(1, 0, 2)
    K = (x @ Wk).reshape(T, n_heads, Dh).transpose(1, 0, 2)
    V = (x @ Wv).reshape(T, n_heads, Dh).transpose(1, 0, 2)
    
    # Scaled Dot-Product Attention
    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(Dh)
    
    # Causal Masking
    mask = np.triu(np.ones((T, T)), k=1).astype(bool)
    scores[:, mask] = -1e9
    
    weights = softmax(scores, axis=-1)
    attn_out = weights @ V # (n_heads, T, Dh)
    
    # 헤드 결합 및 최종 투영
    # (n_heads, T, Dh) -> (T, n_heads, Dh) -> (T, D)
    combined = attn_out.transpose(1, 0, 2).reshape(T, D)
    return combined @ Wo


def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Position-wise Feed-Forward Network: 각 위치마다 독립적으로 적용되는 2층 신경망입니다."""
    h = np.maximum(0, x @ W1 + b1) # ReLU activation
    return h @ W2 + b2


# ============================================================================
# 3. Main Execution Flow
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="입력 텍스트 파일 경로")
    parser.add_argument("--tokens", type=int, default=32, help="처리할 토큰 수")
    parser.add_argument("--d_model", type=int, default=64, help="모델 차원")
    parser.add_argument("--n_heads", type=int, default=4, help="어텐션 헤드 수")
    args = parser.parse_args()

    # 데이터 로드
    data_path = Path(args.input)
    if not data_path.exists():
        print(f"데이터 파일이 없습니다: {data_path}")
        return
    text = data_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    ids = tokenizer.encode(text)[:args.tokens]
    T = len(ids)
    D = args.d_model
    H = args.n_heads

    print(f"--- Transformer Forward Demo (T={T}, D={D}, Heads={H}) ---")

    # 1. 파라미터 초기화 (데모를 위해 랜덤 가중치 사용)
    rng = np.random.default_rng(42)
    scale = 0.02
    
    # Embedding & Positional Embedding
    tok_emb = rng.normal(0, scale, (tokenizer.vocab_size, D))
    pos_emb = rng.normal(0, scale, (T, D))
    
    # Layer 1 Params (Simplified)
    # MHA Params
    Wq = rng.normal(0, scale, (D, D))
    Wk = rng.normal(0, scale, (D, D))
    Wv = rng.normal(0, scale, (D, D))
    Wo = rng.normal(0, scale, (D, D))
    ln1_g = np.ones(D); ln1_b = np.zeros(D)
    
    # FFN Params
    W1 = rng.normal(0, scale, (D, D * 4))
    b1 = np.zeros(D * 4)
    W2 = rng.normal(0, scale, (D * 4, D))
    b2 = np.zeros(D)
    ln2_g = np.ones(D); ln2_b = np.zeros(D)

    # Output Layer
    W_out = rng.normal(0, scale, (D, tokenizer.vocab_size))

    # 2. Forward Pass 계산
    print("\n1) Input Embedding + Positional Encoding")
    x = tok_emb[ids] + pos_emb
    
    print("2) Transformer Block: Multi-Head Attention")
    # Residual Connection + LayerNorm
    x_norm = layer_norm(x, ln1_g, ln1_b)
    attn_out = multi_head_attention(x_norm, Wq, Wk, Wv, Wo, n_heads=H)
    x = x + attn_out
    
    print("3) Transformer Block: Feed-Forward Network")
    # Residual Connection + LayerNorm
    x_norm = layer_norm(x, ln2_g, ln2_b)
    ffn_out = feed_forward(x_norm, W1, b1, W2, b2)
    x = x + ffn_out
    
    print("4) Output Linear Layer")
    logits = x @ W_out # (T, V)
    
    # 3. 결과 출력
    last_probs = softmax(logits[-1])
    top_indices = np.argsort(last_probs)[::-1][:5]
    
    print(f"\n[마지막 토큰 '{tokenizer.decode([ids[-1]])}' 다음으로 올 확률이 높은 글자]")
    for idx in top_indices:
        print(f"  '{tokenizer.vocab[idx]}': {last_probs[idx]:.4f}")


if __name__ == "__main__":
    main()
