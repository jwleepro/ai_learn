"""(학습 아님) Transformer forward 계산을 numpy로 구현한 데모용 코드.

이 파일의 목적:
- Transformer 블록이 어떤 shape로 이어지는지 확인
- residual/LayerNorm/Attention/FFN 조립 감 잡기

주의:
- 가중치는 랜덤 초기화이며, 학습/역전파는 다루지 않습니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from softmax import softmax


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
    tok_emb: np.ndarray  # (V, D)
    pos_emb: np.ndarray  # (T, D)
    layers: list[TransformerLayerParams]
    ln_f_g: np.ndarray
    ln_f_b: np.ndarray
    W_out: np.ndarray  # (D, V)
    b_out: np.ndarray  # (V,)


def layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, *, eps: float = 1e-5) -> np.ndarray:
    # 벡터를 "평균 0, 퍼진 정도 1" 근처로 맞추는 정규화입니다.
    # x: (..., D)
    mean = x.mean(axis=-1, keepdims=True)                         # 평균
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)          # 분산(값들이 평균에서 얼마나 퍼졌는지)
    x_hat = (x - mean) / np.sqrt(var + eps)                       # 정규화: (값-평균) / 퍼진정도
    return x_hat * g + b  # 학습 가능한 스케일(g)과 시프트(b)로 다시 조정


def init_params(cfg: TransformerConfig) -> TransformerParams:
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
                ln1_g=ln1_g,
                ln1_b=ln1_b,
                Wq=Wq,
                Wk=Wk,
                Wv=Wv,
                Wo=Wo,
                ln2_g=ln2_g,
                ln2_b=ln2_b,
                W1=W1,
                b1=b1,
                W2=W2,
                b2=b2,
            )
        )

    ln_f_g = np.ones((cfg.d_model,), dtype=np.float64)
    ln_f_b = np.zeros((cfg.d_model,), dtype=np.float64)
    W_out = rng.normal(0.0, scale, size=(cfg.d_model, cfg.vocab_size)).astype(np.float64)
    b_out = np.zeros((cfg.vocab_size,), dtype=np.float64)
    return TransformerParams(tok_emb=tok_emb, pos_emb=pos_emb, layers=layers, ln_f_g=ln_f_g, ln_f_b=ln_f_b, W_out=W_out, b_out=b_out)


def mha(x: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, Wo: np.ndarray, *, n_heads: int, causal: bool) -> tuple[np.ndarray, np.ndarray]:
    """Multi-Head Attention (forward).

    Shapes:
    - x: (T, D) where D=d_model
    - returns out: (T, D), weights: (H, T, T)
    """

    # x: (T, D)
    T, D = x.shape
    Dh = D // n_heads

    Q = (x @ Wq).reshape(T, n_heads, Dh).transpose(1, 0, 2)  # (H, T, Dh)
    K = (x @ Wk).reshape(T, n_heads, Dh).transpose(1, 0, 2)  # (H, T, Dh)
    V = (x @ Wv).reshape(T, n_heads, Dh).transpose(1, 0, 2)  # (H, T, Dh)

    # 내적으로 관련도 점수를 매기고, sqrt(Dh)로 나눠서 점수가 과하게 커지는 걸 방지
    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(float(Dh))  # (H, T, T)
    if causal:
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = scores.copy()
        scores[:, mask] = -1e9

    weights = softmax(scores, axis=-1)  # (H, T, T)
    out = weights @ V  # (H, T, Dh)
    out = out.transpose(1, 0, 2).reshape(T, D)  # (T, D)
    out = out @ Wo
    return out, weights


def ffn(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    h = x @ W1 + b1
    h = np.maximum(h, 0.0)  # ReLU
    return h @ W2 + b2


def forward(params: TransformerParams, token_ids: np.ndarray, *, n_heads: int, causal: bool = True) -> tuple[np.ndarray, list[np.ndarray]]:
    """Transformer forward.

    Shapes:
    - token_ids: (T,)
    - returns logits: (T, V)
    """

    # token_ids: (T,)
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be 1D")
    T = len(token_ids)
    if T == 0:
        raise ValueError("token_ids must not be empty")
    if T > params.pos_emb.shape[0]:
        raise ValueError("Sequence longer than max_seq_len in params")

    x = params.tok_emb[token_ids] + params.pos_emb[:T]  # (T, D)
    attn_weights: list[np.ndarray] = []

    for layer in params.layers:
        x_ln = layer_norm(x, layer.ln1_g, layer.ln1_b)
        attn_out, w = mha(x_ln, layer.Wq, layer.Wk, layer.Wv, layer.Wo, n_heads=n_heads, causal=causal)
        x = x + attn_out
        attn_weights.append(w)

        x_ln2 = layer_norm(x, layer.ln2_g, layer.ln2_b)
        x = x + ffn(x_ln2, layer.W1, layer.b1, layer.W2, layer.b2)

    x = layer_norm(x, params.ln_f_g, params.ln_f_b)
    logits = x @ params.W_out + params.b_out  # (T, V)
    return logits, attn_weights
