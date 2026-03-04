"""랜덤 가중치 self-attention weight를 출력하는 데모(학습 아님).

목표: causal mask 유무에 따라 weights가 어떻게 달라지는지, shape가 어떻게 생겼는지 확인.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from attention_numpy import self_attention
from tokenizer_char import CharTokenizer


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
    X = E[np.array(ids, dtype=np.int64)]  # (T, D)

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
