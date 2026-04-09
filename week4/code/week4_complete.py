"""Week 4 Complete: Self-Attention (Single Head).

이 통합 파일은 attention_numpy.py와 demo_self_attention.py의
모든 코드를 포함합니다.

(단일 헤드) Self-attention 계산 블록을 numpy로 구현합니다.

Shapes:
- X: (T, D)
- Wq/Wk/Wv: (D, Dh)
- weights: (T, T)
- out: (T, Dh)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from softmax import softmax
from tokenizer_char import CharTokenizer


# ============================================================================
# Section 1: attention_numpy.py - Self-Attention Core
# ============================================================================

def causal_mask(scores: np.ndarray) -> np.ndarray:
    """
    Causal Masking: 미래 토큰 정보를 보이지 않게 마스킹

    배경:
    - 언어모델은 좌에서 우로(left-to-right) 생성: 이전 토큰들만 봐서 다음 토큰 예측
    - Attention은 모든 토큰 쌍을 고려하므로, "미래 토큰을 보는" 문제 발생
    - 해결: 미래 위치의 점수를 매우 낮은 값(-1e9)으로 설정

    Args:
        scores: shape (T, T) - 어텐션 점수 (T = 시퀀스 길이)

    Returns:
        masked: shape (T, T) - 미래 부분이 -1e9로 마스킹된 scores
    """
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be (T, T)")
    T = scores.shape[0]
    masked = scores.copy()
    upper = np.triu(np.ones((T, T), dtype=bool), k=1)
    masked[upper] = -1e9
    return masked


def self_attention(
    X: np.ndarray,
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    *,
    causal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Self-Attention 계산: 각 토큰이 다른 토큰들을 얼마나 "주목"할지 결정합니다

    Self-Attention의 3가지 구성요소:
    1. Query (Q): "이 위치에서 뭘 찾고 있는가?"
    2. Key (K): "이 위치가 제공할 수 있는 것은?"
    3. Value (V): "실제 정보"

    계산 단계:
    1. X를 Q, K, V로 투영 (각각 다른 측면을 학습)
    2. Q와 K의 내적으로 유사도 점수 계산 (scaled dot-product)
    3. 점수를 softmax로 확률로 변환 → 어텐션 가중치
    4. 가중치로 V를 조합 → 최종 출력

    수학:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_h)) @ V

    Args:
        X: shape (T, D) - 입력 시퀀스 (T = 길이, D = 특성 차원)
        Wq, Wk, Wv: shape (D, Dh) - 학습 가능한 투영 행렬 (Dh = 어텐션 차원)
        causal: True면 인과적 마스킹 적용 (미래 토큰 무시)

    Returns:
        weights: shape (T, T) - 어텐션 가중치
        out: shape (T, Dh) - 최종 어텐션 출력
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (T, D)")
    if Wq.shape[0] != X.shape[1] or Wk.shape[0] != X.shape[1] or Wv.shape[0] != X.shape[1]:
        raise ValueError("Wq/Wk/Wv first dim must match X feature dim")

    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    Dh = Q.shape[1]
    scores = (Q @ K.T) / np.sqrt(float(Dh))

    if causal:
        scores = causal_mask(scores)

    weights = softmax(scores, axis=1)
    out = weights @ V

    return weights, out


# ============================================================================
# Section 2: demo_self_attention.py - Demo script
# ============================================================================

def token_label(vocab: tuple[str, ...], token_id: int) -> str:
    ch = vocab[token_id]
    code = ord(ch)
    if ch == "\n":
        shown = "\\n"
    elif ch == "\t":
        shown = "\\t"
    elif ch == " ":
        shown = "<space>"
    else:
        shown = ch
    return f"{shown}(U+{code:04X},id={token_id})"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="(학습 아님) self-attention weights 출력 데모(numpy).")
    p.add_argument("--input", required=True, help="입력 UTF-8 텍스트 파일 경로")
    p.add_argument("--tokens", type=int, default=24, help="앞에서부터 볼 토큰 수(T)")
    p.add_argument("--d_model", type=int, default=16, help="임베딩 차원(d_model)")
    p.add_argument("--d_head", type=int, default=16, help="헤드 차원(d_head)")
    p.add_argument("--seed", type=int, default=0, help="난수 시드(seed)")
    p.add_argument("--no_causal", action="store_true", help="causal mask 끄기(미래 토큰을 볼 수 있음)")
    p.add_argument("--pos", type=int, default=-1, help="설명할 위치(pos). 기본은 마지막(-1).")
    p.add_argument("--top", type=int, default=8, help="가장 크게 보는 위치 top-N 출력")
    p.add_argument("--matrix", action="store_true", help="전체 weights 행렬 출력(T가 작을 때만)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    if not text:
        raise ValueError("Input text is empty")

    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)
    T = min(int(args.tokens), len(ids))
    ids = ids[:T]

    rng = np.random.default_rng(int(args.seed))
    E = rng.normal(0.0, 0.5, size=(tok.vocab_size, int(args.d_model))).astype(np.float64)
    X = E[np.array(ids, dtype=np.int64)]

    Wq = rng.normal(0.0, 0.5, size=(int(args.d_model), int(args.d_head))).astype(np.float64)
    Wk = rng.normal(0.0, 0.5, size=(int(args.d_model), int(args.d_head))).astype(np.float64)
    Wv = rng.normal(0.0, 0.5, size=(int(args.d_model), int(args.d_head))).astype(np.float64)

    causal = not bool(args.no_causal)
    weights, _ = self_attention(X, Wq, Wk, Wv, causal=causal)

    pos = int(args.pos) if int(args.pos) >= 0 else T - 1
    if not (0 <= pos < T):
        raise ValueError("--pos out of range for selected tokens")

    row = weights[pos]
    top_n = min(int(args.top), T)
    top_idx = np.argsort(row)[-top_n:][::-1]

    print(f"T={T}  causal={causal}  pos={pos}")
    print("context tokens:")
    for i, token_id in enumerate(ids):
        print(f"  [{i:02d}] {token_label(tok.vocab, token_id)}")

    print("")
    print(f"Top attends for position {pos}:")
    for j in top_idx:
        print(f"  to [{int(j):02d}] w={float(row[int(j)]):.4f}")

    if args.matrix:
        if T > 32:
            raise ValueError("--matrix is only allowed for T<=32 to keep output readable")
        print("")
        print("Attention weights matrix (rows=from, cols=to):")
        with np.printoptions(precision=3, suppress=True, linewidth=200):
            print(weights)


if __name__ == "__main__":
    main()
