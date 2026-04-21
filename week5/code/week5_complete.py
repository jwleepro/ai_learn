"""Week 5: Transformer Forward (초보용 단순 버전)."""

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


def softmax_last(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def multi_head_attention(x: np.ndarray, wq: np.ndarray, wk: np.ndarray, wv: np.ndarray, wo: np.ndarray, n_heads: int):
    t, d_model = x.shape
    d_head = d_model // n_heads

    q = (x @ wq).reshape(t, n_heads, d_head).transpose(1, 0, 2)
    k = (x @ wk).reshape(t, n_heads, d_head).transpose(1, 0, 2)
    v = (x @ wv).reshape(t, n_heads, d_head).transpose(1, 0, 2)

    scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(d_head)
    mask = np.triu(np.ones((t, t), dtype=bool), k=1)
    scores[:, mask] = -1e9
    weights = softmax_last(scores)
    out = weights @ v
    out = out.transpose(1, 0, 2).reshape(t, d_model)
    return out @ wo


def feed_forward(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray):
    h = np.maximum(0.0, x @ w1 + b1)
    return h @ w2 + b2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"파일이 없습니다: {path}")
        return

    text = path.read_text(encoding="utf-8")
    vocab, char_to_id = build_tokenizer(text)
    ids = encode_text(text, char_to_id)[: args.tokens]
    t = len(ids)
    d_model = args.d_model
    n_heads = args.n_heads

    rng = np.random.default_rng(42)
    scale = 0.02

    tok_emb = rng.normal(0, scale, (len(vocab), d_model))
    pos_emb = rng.normal(0, scale, (t, d_model))
    x = tok_emb[ids] + pos_emb

    wq = rng.normal(0, scale, (d_model, d_model))
    wk = rng.normal(0, scale, (d_model, d_model))
    wv = rng.normal(0, scale, (d_model, d_model))
    wo = rng.normal(0, scale, (d_model, d_model))

    w1 = rng.normal(0, scale, (d_model, d_model * 4))
    b1 = np.zeros(d_model * 4)
    w2 = rng.normal(0, scale, (d_model * 4, d_model))
    b2 = np.zeros(d_model)

    w_out = rng.normal(0, scale, (d_model, len(vocab)))

    x = x + multi_head_attention(layer_norm(x), wq, wk, wv, wo, n_heads=n_heads)
    x = x + feed_forward(layer_norm(x), w1, b1, w2, b2)
    logits = x @ w_out

    probs = softmax_last(logits[-1])
    top = np.argsort(probs)[::-1][:5]
    print("마지막 토큰 다음 글자 확률 상위 5개")
    for idx in top:
        print(f"{vocab[idx]!r}: {probs[idx]:.4f}")


if __name__ == "__main__":
    main()
