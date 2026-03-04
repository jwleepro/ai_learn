"""신경망 빅램(Neural Bigram) 언어모델 (numpy, 수동 backprop).

카운트 대신, 학습 가능한 가중치 행렬 W를 둡니다.

- vocab 크기: V
- W: (V, V)
  - prev 토큰 id가 i일 때, logits = W[i]  (다음 토큰 점수)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from softmax import log_softmax, softmax


@dataclass(frozen=True)
class BigramNNConfig:
    lr: float = 1.0
    epochs: int = 20
    batch_size: int = 1024
    seed: int = 0
    init_scale: float = 0.01


def init_W(vocab_size: int, rng: np.random.Generator, *, scale: float = 0.01) -> np.ndarray:
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")
    if scale <= 0:
        raise ValueError("scale must be > 0")
    return rng.normal(0.0, scale, size=(vocab_size, vocab_size)).astype(np.float64)


def loss_and_grad_W(W: np.ndarray, prev_ids: np.ndarray, next_ids: np.ndarray) -> tuple[float, np.ndarray]:
    """배치에 대한 cross-entropy loss와 W의 gradient를 계산합니다.

    Shapes:
    - W: (V, V)
    - prev_ids: (B,)
    - next_ids: (B,)
    - returns grad_W: (V, V)  (prev에 해당하는 행(row)만 업데이트에 사용됨)
    """

    if prev_ids.ndim != 1 or next_ids.ndim != 1:
        raise ValueError("prev_ids and next_ids must be 1D")
    if len(prev_ids) != len(next_ids):
        raise ValueError("prev_ids and next_ids must have same length")
    if len(prev_ids) == 0:
        raise ValueError("batch is empty")

    logits = W[prev_ids]  # (B, V)
    log_probs = log_softmax(logits, axis=1)
    loss = float(-log_probs[np.arange(len(next_ids)), next_ids].mean())

    probs = np.exp(log_probs)  # (B, V)
    grad_logits = probs
    grad_logits[np.arange(len(next_ids)), next_ids] -= 1.0
    grad_logits /= float(len(next_ids))

    grad_W = np.zeros_like(W)
    np.add.at(grad_W, prev_ids, grad_logits)
    return loss, grad_W


def eval_loss(W: np.ndarray, prev_ids: np.ndarray, next_ids: np.ndarray, *, batch_size: int = 4096) -> float:
    if len(prev_ids) == 0:
        raise ValueError("eval set is empty")
    total = 0.0
    count = 0
    for start in range(0, len(prev_ids), batch_size):
        end = min(len(prev_ids), start + batch_size)
        logits = W[prev_ids[start:end]]
        log_probs = log_softmax(logits, axis=1)
        loss = -log_probs[np.arange(end - start), next_ids[start:end]]
        total += float(loss.sum())
        count += int(end - start)
    return total / count


def train_bigram_nn(
    prev_train: np.ndarray,
    next_train: np.ndarray,
    vocab_size: int,
    *,
    config: BigramNNConfig,
    prev_val: np.ndarray | None = None,
    next_val: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    rng = np.random.default_rng(config.seed)
    W = init_W(vocab_size, rng, scale=config.init_scale)

    history: list[dict[str, float]] = []
    n = len(prev_train)
    if n == 0:
        raise ValueError("train set is empty")

    for epoch in range(1, config.epochs + 1):
        perm = rng.permutation(n)
        prev_shuf = prev_train[perm]
        next_shuf = next_train[perm]

        epoch_loss = 0.0
        steps = 0
        for start in range(0, n, config.batch_size):
            end = min(n, start + config.batch_size)
            loss, grad_W = loss_and_grad_W(W, prev_shuf[start:end], next_shuf[start:end])
            W -= config.lr * grad_W
            epoch_loss += loss
            steps += 1

        train_loss = epoch_loss / max(steps, 1)
        metrics: dict[str, float] = {"epoch": float(epoch), "train_loss": float(train_loss)}

        if prev_val is not None and next_val is not None and len(prev_val) > 0:
            val_loss = eval_loss(W, prev_val, next_val)
            metrics["val_loss"] = float(val_loss)
        history.append(metrics)

    return W, history


def bigram_probs(W: np.ndarray, prev_id: int, *, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = W[prev_id] / float(temperature)
    return softmax(logits, axis=0)
