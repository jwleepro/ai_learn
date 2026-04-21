"""Week 4: Self-Attention (초보용 단순 버전)."""

import argparse
from pathlib import Path

import numpy as np


def build_tokenizer(text: str):
    vocab = sorted(set(text))
    char_to_id = {}
    for i, ch in enumerate(vocab):
        char_to_id[ch] = i
    return vocab, char_to_id


def encode_text(text: str, char_to_id: dict[str, int]) -> list[int]:
    return [char_to_id[ch] for ch in text]


def decode_ids(ids: list[int], vocab: list[str]) -> str:
    return "".join(vocab[i] for i in ids)


def softmax_rows(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def apply_causal_mask(scores: np.ndarray) -> np.ndarray:
    t = scores.shape[0]
    masked = scores.copy()
    for i in range(t):
        for j in range(i + 1, t):
            masked[i, j] = -1e9
    return masked


def self_attention(x: np.ndarray, wq: np.ndarray, wk: np.ndarray, wv: np.ndarray, causal: bool):
    q = x @ wq
    k = x @ wk
    v = x @ wv

    scores = (q @ k.T) / np.sqrt(q.shape[1])
    if causal:
        scores = apply_causal_mask(scores)
    weights = softmax_rows(scores)
    out = weights @ v
    return weights, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--pos", type=int, default=-1)
    parser.add_argument("--no_causal", action="store_true")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"파일이 없습니다: {path}")
        return

    text = path.read_text(encoding="utf-8")
    vocab, char_to_id = build_tokenizer(text)
    all_ids = encode_text(text, char_to_id)
    t = min(args.tokens, len(all_ids))
    ids = all_ids[:t]

    d_model = 16
    d_head = 16
    rng = np.random.default_rng(42)
    emb = rng.normal(0, 0.1, (len(vocab), d_model))
    x = emb[ids]
    wq = rng.normal(0, 0.1, (d_model, d_head))
    wk = rng.normal(0, 0.1, (d_model, d_head))
    wv = rng.normal(0, 0.1, (d_model, d_head))

    weights, _ = self_attention(x, wq, wk, wv, causal=not args.no_causal)
    pos = args.pos if args.pos >= 0 else t - 1

    print(f"T={t}, causal={not args.no_causal}")
    print(f"분석 문자: {decode_ids([ids[pos]], vocab)!r}")
    print("가중치 상위 5개")
    row = weights[pos]
    top = np.argsort(row)[::-1][:5]
    for idx in top:
        print(f"to {idx:2d} {decode_ids([ids[idx]], vocab)!r}: {row[idx]:.4f}")
    print(f"문맥: {decode_ids(ids, vocab)!r}")


if __name__ == "__main__":
    main()
