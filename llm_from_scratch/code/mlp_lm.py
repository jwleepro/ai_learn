"""아주 작은 MLP 언어모델(numpy, 수동 backprop).

“최근 k개 토큰(context)”을 입력으로 받아 다음 토큰을 예측합니다.

기호(자주 쓰는 shape):
- V: vocab 크기
- C: context_len
- D: embed_dim
- H: hidden_dim
- B: batch 크기
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from softmax import log_softmax, softmax


@dataclass(frozen=True)
class MLPLMConfig:
    context_len: int = 8
    embed_dim: int = 32
    hidden_dim: int = 128
    lr: float = 0.1
    epochs: int = 50
    batch_size: int = 256
    seed: int = 0
    init_scale: float = 0.02


@dataclass
class MLPLMParams:
    E: np.ndarray  # (V, D)
    W1: np.ndarray  # (context_len*D, H)
    b1: np.ndarray  # (H,)
    W2: np.ndarray  # (H, V)
    b2: np.ndarray  # (V,)


def init_params(vocab_size: int, *, config: MLPLMConfig, rng: np.random.Generator) -> MLPLMParams:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")
    if config.context_len <= 0:
        raise ValueError("context_len must be > 0")
    if config.embed_dim <= 0 or config.hidden_dim <= 0:
        raise ValueError("embed_dim and hidden_dim must be > 0")
    if config.init_scale <= 0:
        raise ValueError("init_scale must be > 0")

    D = config.embed_dim
    H = config.hidden_dim
    C = config.context_len
    scale = config.init_scale

    E = rng.normal(0.0, scale, size=(vocab_size, D)).astype(np.float64)
    W1 = rng.normal(0.0, scale, size=(C * D, H)).astype(np.float64)
    b1 = np.zeros((H,), dtype=np.float64)
    W2 = rng.normal(0.0, scale, size=(H, vocab_size)).astype(np.float64)
    b2 = np.zeros((vocab_size,), dtype=np.float64)
    return MLPLMParams(E=E, W1=W1, b1=b1, W2=W2, b2=b2)


def forward(params: MLPLMParams, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """전방향(forward) 계산.

    Shapes:
    - X: (B, C)  (토큰 id)
    - returns logits: (B, V)
    """

    # X: (B, C)
    emb = params.E[X]  # (B, C, D)
    h_in = emb.reshape(len(X), -1)  # (B, C*D)
    h_pre = h_in @ params.W1 + params.b1  # (B, H)
    h = np.tanh(h_pre)  # (B, H)
    logits = h @ params.W2 + params.b2  # (B, V)
    cache = {"X": X, "emb": emb, "h_in": h_in, "h_pre": h_pre, "h": h}
    return logits, cache


def loss_and_grads(params: MLPLMParams, X: np.ndarray, y: np.ndarray) -> tuple[float, MLPLMParams]:
    """loss(정답을 얼마나 잘 맞히는지)와 각 파라미터의 기울기를 계산합니다.

    아래 역전파(backpropagation) 코드는 "loss를 줄이려면 각 파라미터를
    어느 방향으로 바꿔야 하는지"를 거꾸로 추적하는 과정입니다.
    공식 유도는 생략하고, 코드의 shape 흐름만 따라가도 괜찮습니다.
    """

    logits, cache = forward(params, X)
    log_probs = log_softmax(logits, axis=1)
    # loss: 정답 토큰의 확률이 높을수록 작아지는 값
    loss = float(-log_probs[np.arange(len(y)), y].mean())

    # --- 역전파: 출력층 → 은닉층 → 임베딩 순으로 기울기를 전파 ---
    # 출력층 기울기 (bigram_nn과 동일한 softmax 성질 활용)
    probs = np.exp(log_probs)
    dlogits = probs
    dlogits[np.arange(len(y)), y] -= 1.0
    dlogits /= float(len(y))

    h = cache["h"]
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(axis=0)

    # 은닉층 기울기
    dh = dlogits @ params.W2.T
    # tanh 활성화의 기울기: tanh 출력이 0에 가까우면 잘 통과, ±1에 가까우면 거의 차단
    dh_pre = dh * (1.0 - np.tanh(cache["h_pre"]) ** 2)

    h_in = cache["h_in"]
    dW1 = h_in.T @ dh_pre
    db1 = dh_pre.sum(axis=0)

    # 임베딩 기울기
    dh_in = dh_pre @ params.W1.T  # (B, C*D)
    dEmb = dh_in.reshape(cache["emb"].shape)  # (B, C, D)

    dE = np.zeros_like(params.E)
    X_ids = cache["X"].reshape(-1)
    dEmb_flat = dEmb.reshape(-1, dEmb.shape[-1])
    np.add.at(dE, X_ids, dEmb_flat)

    grads = MLPLMParams(E=dE, W1=dW1, b1=db1, W2=dW2, b2=db2)
    return loss, grads


def apply_grads(params: MLPLMParams, grads: MLPLMParams, *, lr: float) -> None:
    params.E -= lr * grads.E
    params.W1 -= lr * grads.W1
    params.b1 -= lr * grads.b1
    params.W2 -= lr * grads.W2
    params.b2 -= lr * grads.b2


def eval_loss(params: MLPLMParams, X: np.ndarray, y: np.ndarray, *, batch_size: int = 4096) -> float:
    if len(X) == 0:
        raise ValueError("eval set is empty")
    total = 0.0
    count = 0
    for start in range(0, len(X), batch_size):
        end = min(len(X), start + batch_size)
        logits, _ = forward(params, X[start:end])
        log_probs = log_softmax(logits, axis=1)
        loss = -log_probs[np.arange(end - start), y[start:end]]
        total += float(loss.sum())
        count += int(end - start)
    return total / count


def train_mlp_lm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    vocab_size: int,
    *,
    config: MLPLMConfig,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> tuple[MLPLMParams, list[dict[str, float]]]:
    rng = np.random.default_rng(config.seed)
    params = init_params(vocab_size, config=config, rng=rng)

    history: list[dict[str, float]] = []
    n = len(X_train)
    if n == 0:
        raise ValueError("train set is empty")

    for epoch in range(1, config.epochs + 1):
        perm = rng.permutation(n)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        epoch_loss = 0.0
        steps = 0
        for start in range(0, n, config.batch_size):
            end = min(n, start + config.batch_size)
            loss, grads = loss_and_grads(params, X_shuf[start:end], y_shuf[start:end])
            apply_grads(params, grads, lr=config.lr)
            epoch_loss += loss
            steps += 1

        train_loss = epoch_loss / max(steps, 1)
        metrics: dict[str, float] = {"epoch": float(epoch), "train_loss": float(train_loss)}

        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_loss = eval_loss(params, X_val, y_val)
            metrics["val_loss"] = float(val_loss)
        history.append(metrics)

    return params, history


def next_token_probs(params: MLPLMParams, context_ids: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    if context_ids.ndim != 1:
        raise ValueError("context_ids must be 1D")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits, _ = forward(params, context_ids.reshape(1, -1))
    logits = logits[0] / float(temperature)
    return softmax(logits, axis=0)
