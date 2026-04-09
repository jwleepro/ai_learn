"""Week 5 Complete: Transformer Forward Pass (numpy, no training).

이 통합 파일은 transformer_numpy.py와 demo_transformer_forward.py의
모든 코드를 포함합니다.

Transformer의 forward 계산을 numpy로 구현한 데모용 코드입니다.
(가중치는 랜덤 초기화이며, 학습/역전파는 다루지 않습니다)

주요 컴포넌트:
- Multi-Head Attention (MHA)
- Feed-Forward Network (FFN)
- Layer Normalization
- Residual connections
- 위치 임베딩
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from softmax import softmax
from tokenizer_char import CharTokenizer


# ============================================================================
# Section 1: softmax.py - Helper (if not imported)
# ============================================================================

# Already in imports above, but define locally just in case


# ============================================================================
# Section 2: transformer_numpy.py - Transformer Core
# ============================================================================

@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    max_seq_len: int = 64
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 256
    n_layers: int = 2
    seed: int = 0


@dataclass
class TransformerLayerParams:
    ln1_g: np.ndarray
    ln1_b: np.ndarray
    Wq: np.ndarray
    Wk: np.ndarray
    Wv: np.ndarray
    Wo: np.ndarray
    ln2_g: np.ndarray
    ln2_b: np.ndarray
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray


@dataclass
class TransformerParams:
    tok_emb: np.ndarray
    pos_emb: np.ndarray
    layers: list[TransformerLayerParams]
    ln_f_g: np.ndarray
    ln_f_b: np.ndarray
    W_out: np.ndarray
    b_out: np.ndarray


def layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, *, eps: float = 1e-5) -> np.ndarray:
    """Layer Normalization을 적용합니다."""
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return x_hat * g + b


def init_params(cfg: TransformerConfig) -> TransformerParams:
    """Transformer 파라미터를 초기화합니다."""
    if cfg.vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")

    rng = np.random.default_rng(cfg.seed)
    scale = 0.02

    tok_emb = rng.normal(0.0, scale, size=(cfg.vocab_size, cfg.d_model)).astype(np.float64)
    pos_emb = rng.normal(0.0, scale, size=(cfg.max_seq_len, cfg.d_model)).astype(np.float64)

    layers: list[TransformerLayerParams] = []
    for _ in range(cfg.n_layers):
        ln1_g = np.ones((cfg.d_model,), dtype=np.float64)
        ln1_b = np.zeros((cfg.d_model,), dtype=np.float64)
        Wq = rng.normal(0.0, scale, size=(cfg.d_model, cfg.d_model)).astype(np.float64)
        Wk = rng.normal(0.0, scale, size=(cfg.d_model, cfg.d_model)).astype(np.float64)
        Wv = rng.normal(0.0, scale, size=(cfg.d_model, cfg.d_model)).astype(np.float64)
        Wo = rng.normal(0.0, scale, size=(cfg.d_model, cfg.d_model)).astype(np.float64)
        ln2_g = np.ones((cfg.d_model,), dtype=np.float64)
        ln2_b = np.zeros((cfg.d_model,), dtype=np.float64)
        W1 = rng.normal(0.0, scale, size=(cfg.d_model, cfg.d_ff)).astype(np.float64)
        b1 = np.zeros((cfg.d_ff,), dtype=np.float64)
        W2 = rng.normal(0.0, scale, size=(cfg.d_ff, cfg.d_model)).astype(np.float64)
        b2 = np.zeros((cfg.d_model,), dtype=np.float64)
        layers.append(
            TransformerLayerParams(
                ln1_g=ln1_g, ln1_b=ln1_b, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo,
                ln2_g=ln2_g, ln2_b=ln2_b, W1=W1, b1=b1, W2=W2, b2=b2,
            )
        )

    ln_f_g = np.ones((cfg.d_model,), dtype=np.float64)
    ln_f_b = np.zeros((cfg.d_model,), dtype=np.float64)
    W_out = rng.normal(0.0, scale, size=(cfg.d_model, cfg.vocab_size)).astype(np.float64)
    b_out = np.zeros((cfg.vocab_size,), dtype=np.float64)
    return TransformerParams(tok_emb=tok_emb, pos_emb=pos_emb, layers=layers, ln_f_g=ln_f_g, ln_f_b=ln_f_b, W_out=W_out, b_out=b_out)


def mha(x: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, Wo: np.ndarray, *, n_heads: int, causal: bool) -> tuple[np.ndarray, np.ndarray]:
    """Multi-Head Attention (MHA) Forward 계산."""
    T, D = x.shape
    Dh = D // n_heads

    Q = (x @ Wq).reshape(T, n_heads, Dh).transpose(1, 0, 2)
    K = (x @ Wk).reshape(T, n_heads, Dh).transpose(1, 0, 2)
    V = (x @ Wv).reshape(T, n_heads, Dh).transpose(1, 0, 2)

    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(float(Dh))

    if causal:
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = scores.copy()
        scores[:, mask] = -1e9

    weights = softmax(scores, axis=-1)
    out = weights @ V

    out = out.transpose(1, 0, 2).reshape(T, D)
    out = out @ Wo

    return out, weights


def ffn(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Feed-Forward Network (FFN) Forward 계산."""
    h = x @ W1 + b1
    h = np.maximum(h, 0.0)
    return h @ W2 + b2


def forward(params: TransformerParams, token_ids: np.ndarray, *, n_heads: int, causal: bool = True) -> tuple[np.ndarray, list[np.ndarray]]:
    """Transformer Forward Pass를 수행합니다."""
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be 1D")
    T = len(token_ids)
    if T == 0:
        raise ValueError("token_ids must not be empty")
    if T > params.pos_emb.shape[0]:
        raise ValueError("Sequence longer than max_seq_len in params")

    x = params.tok_emb[token_ids] + params.pos_emb[:T]
    attn_weights: list[np.ndarray] = []

    for layer in params.layers:
        x_ln = layer_norm(x, layer.ln1_g, layer.ln1_b)
        attn_out, w = mha(x_ln, layer.Wq, layer.Wk, layer.Wv, layer.Wo, n_heads=n_heads, causal=causal)
        x = x + attn_out
        attn_weights.append(w)

        x_ln2 = layer_norm(x, layer.ln2_g, layer.ln2_b)
        x = x + ffn(x_ln2, layer.W1, layer.b1, layer.W2, layer.b2)

    x = layer_norm(x, params.ln_f_g, params.ln_f_b)
    logits = x @ params.W_out + params.b_out

    return logits, attn_weights


# ============================================================================
# Section 3: demo_transformer_forward.py - Demo script
# ============================================================================

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
