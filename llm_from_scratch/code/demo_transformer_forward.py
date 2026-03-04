"""랜덤 가중치 Transformer forward를 실행해 shape 흐름을 확인하는 데모(학습 아님)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from softmax import softmax
from tokenizer_char import CharTokenizer
from transformer_numpy import TransformerConfig, forward, init_params


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="(학습 아님) Transformer forward 데모(numpy).")
    p.add_argument("--input", required=True, help="입력 UTF-8 텍스트 파일 경로")
    p.add_argument("--tokens", type=int, default=64, help="앞에서부터 넣을 토큰 수(T)")
    p.add_argument("--d_model", type=int, default=64, help="모델 차원(d_model)")
    p.add_argument("--heads", type=int, default=4, help="헤드 수(heads)")
    p.add_argument("--layers", type=int, default=2, help="레이어 수(layers)")
    p.add_argument("--seed", type=int, default=0, help="난수 시드(seed)")
    p.add_argument("--top", type=int, default=10, help="마지막 위치에서 top-N 토큰 출력")
    return p.parse_args()


def label(vocab: tuple[str, ...], token_id: int) -> str:
    ch = vocab[token_id]
    if ch == "\n":
        shown = "\\n"
    elif ch == "\t":
        shown = "\\t"
    elif ch == " ":
        shown = "<space>"
    else:
        shown = ch
    return f"{shown}(U+{ord(ch):04X},id={token_id})"


def main() -> None:
    args = parse_args()
    text = Path(args.input).read_text(encoding="utf-8")
    if not text:
        raise ValueError("Input text is empty")

    tok = CharTokenizer.from_text(text)
    ids = np.array(tok.encode(text), dtype=np.int64)
    T = min(int(args.tokens), len(ids))
    ids = ids[:T]

    cfg = TransformerConfig(
        vocab_size=tok.vocab_size,
        max_seq_len=T,
        d_model=int(args.d_model),
        n_heads=int(args.heads),
        d_ff=int(args.d_model) * 4,
        n_layers=int(args.layers),
        seed=int(args.seed),
    )
    params = init_params(cfg)

    logits, _ = forward(params, ids, n_heads=cfg.n_heads, causal=True)
    last_logits = logits[-1]
    probs = softmax(last_logits, axis=0)

    top_n = min(int(args.top), tok.vocab_size)
    top_ids = np.argsort(probs)[-top_n:][::-1]
    print(f"T={T}  d_model={cfg.d_model}  heads={cfg.n_heads}  layers={cfg.n_layers}")
    print("Top predictions (random weights; just shape demo):")
    for token_id in top_ids:
        tid = int(token_id)
        print(f"  {label(tok.vocab, tid)}  p={float(probs[tid]):.4f}")


if __name__ == "__main__":
    main()
