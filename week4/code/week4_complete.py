"""Week 4 Complete: Self-Attention (Single Head).

이 파일은 Self-Attention 메커니즘의 핵심 계산 과정을 하나의 파일로 통합한 교육용 코드입니다.
데이터 로드부터 토큰화, 어텐션 가중치 계산 및 시각화 데모까지 순서대로 진행됩니다.

주요 내용:
1. CharTokenizer: 글자 단위 토크나이저
2. Causal Masking: 미래 토큰을 가리는 기법
3. Self-Attention: Query, Key, Value를 이용한 어텐션 점수 및 출력 계산
4. Visualization: 특정 위치에서 어떤 토큰을 중요하게 보는지(Attention weights) 출력

실행 방법:
- 데모: python week4/code/week4_complete.py --input week4/data/tiny_corpus_ko.txt
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
        if len(self.vocab) == 0:
            raise ValueError("vocab empty")
        self.char_to_id = {ch: i for i, ch in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

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


# ============================================================================
# 2. Self-Attention Implementation
# ============================================================================

def causal_mask(scores: np.ndarray) -> np.ndarray:
    """미래 토큰 정보를 보이지 않게 -1e9(매우 낮은 값)로 마스킹합니다."""
    T = scores.shape[0]
    mask = np.triu(np.ones((T, T)), k=1).astype(bool)
    masked = scores.copy()
    masked[mask] = -1e9
    return masked


def self_attention(X: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, causal: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Query, Key, Value를 이용한 Self-Attention 계산
    X: (T, D), W: (D, Dh) -> Weights: (T, T), Output: (T, Dh)
    """
    Q = X @ Wq # (T, Dh)
    K = X @ Wk # (T, Dh)
    V = X @ Wv # (T, Dh)
    
    Dh = Q.shape[1]
    # Dot-product 유사도 계산 및 스케일링
    scores = (Q @ K.T) / np.sqrt(Dh)
    
    if causal:
        scores = causal_mask(scores)
    
    # Softmax를 통해 가중치(확률)로 변환
    weights = softmax(scores, axis=1)
    # 가중치와 Value의 결합
    out = weights @ V
    
    return weights, out


# ============================================================================
# 3. Main Execution Flow
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="입력 텍스트 파일 경로")
    parser.add_argument("--tokens", type=int, default=20, help="분석할 토큰 수")
    parser.add_argument("--pos", type=int, default=-1, help="분석할 특정 위치 (기본: 마지막)")
    parser.add_argument("--no_causal", action="store_true", help="Causal Masking 비활성화")
    args = parser.parse_args()

    # 데이터 로드
    data_path = Path(args.input)
    if not data_path.exists():
        print(f"데이터 파일이 없습니다: {data_path}")
        return
    text = data_path.read_text(encoding="utf-8")
    
    # 1. 토크나이저 및 임베딩 준비 (데모를 위해 랜덤 임베딩 사용)
    tokenizer = CharTokenizer.from_text(text)
    full_ids = tokenizer.encode(text)
    T = min(args.tokens, len(full_ids))
    ids = full_ids[:T]
    
    D = 16  # Embedding dimension
    Dh = 16 # Head dimension
    rng = np.random.default_rng(42)
    
    # 가상의 임베딩 테이블 및 투영 행렬
    E = rng.normal(0, 0.1, (tokenizer.vocab_size, D))
    X = E[ids] # (T, D)
    
    Wq = rng.normal(0, 0.1, (D, Dh))
    Wk = rng.normal(0, 0.1, (D, Dh))
    Wv = rng.normal(0, 0.1, (D, Dh))

    # 2. Self-Attention 계산
    causal = not args.no_causal
    weights, out = self_attention(X, Wq, Wk, Wv, causal=causal)
    
    # 3. 결과 출력
    pos = args.pos if args.pos >= 0 else T - 1
    print(f"--- Self-Attention Demo (T={T}, Causal={causal}) ---")
    print(f"분석 위치 [{pos}]: '{tokenizer.decode([ids[pos]])}'")
    
    print("\nAttention 가중치 (상위 5개):")
    row = weights[pos]
    top_indices = np.argsort(row)[::-1][:5]
    for idx in top_indices:
        target_char = tokenizer.decode([ids[idx]])
        print(f"  to [{idx:2d}] '{target_char}': {row[idx]:.4f}")

    print("\n[전체 문맥]")
    print(f"'{tokenizer.decode(ids)}'")


if __name__ == "__main__":
    main()
