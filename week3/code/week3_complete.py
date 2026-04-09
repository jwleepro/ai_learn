"""Week 3 Complete: MLP 언어모델.

이 통합 파일은 mlp_lm.py, generate_mlp_lm.py, train_mlp_lm.py의
모든 코드를 포함합니다.

"최근 k개 토큰(context)"을 입력으로 받아 다음 토큰을 예측하는 MLP 모델입니다.

기호(자주 쓰는 shape):
- V: vocab 크기
- C: context_len
- D: embed_dim
- H: hidden_dim
- B: batch 크기
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from softmax import log_softmax, softmax
from tokenizer_char import CharTokenizer
from model_io import MLPLMCheckpoint, save_mlp_lm, load_mlp_lm
from dataset_lm import make_context_dataset
from sampling import SamplingConfig, sample_from_probs


# ============================================================================
# Section 1: mlp_lm.py - MLP Language Model Core
# ============================================================================

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
    E: np.ndarray
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray


def init_params(vocab_size: int, *, config: MLPLMConfig, rng: np.random.Generator) -> MLPLMParams:
    """모든 파라미터를 초기화합니다."""
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
    """전방향(forward) 계산"""
    emb = params.E[X]
    h_in = emb.reshape(len(X), -1)
    h_pre = h_in @ params.W1 + params.b1
    h = np.tanh(h_pre)
    logits = h @ params.W2 + params.b2

    cache = {"X": X, "emb": emb, "h_in": h_in, "h_pre": h_pre, "h": h}
    return logits, cache


def loss_and_grads(params: MLPLMParams, X: np.ndarray, y: np.ndarray) -> tuple[float, MLPLMParams]:
    """손실함수(loss)와 각 파라미터의 기울기를 계산합니다(역전파)."""
    logits, cache = forward(params, X)

    log_probs = log_softmax(logits, axis=1)
    loss = float(-log_probs[np.arange(len(y)), y].mean())

    probs = np.exp(log_probs)
    dlogits = probs.copy()
    dlogits[np.arange(len(y)), y] -= 1.0
    dlogits /= float(len(y))

    h = cache["h"]
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(axis=0)

    dh = dlogits @ params.W2.T
    dh_pre = dh * (1.0 - np.tanh(cache["h_pre"]) ** 2)

    h_in = cache["h_in"]
    dW1 = h_in.T @ dh_pre
    db1 = dh_pre.sum(axis=0)

    dh_in = dh_pre @ params.W1.T
    dEmb = dh_in.reshape(cache["emb"].shape)

    dE = np.zeros_like(params.E)
    X_ids = cache["X"].reshape(-1)
    dEmb_flat = dEmb.reshape(-1, dEmb.shape[-1])
    np.add.at(dE, X_ids, dEmb_flat)

    grads = MLPLMParams(E=dE, W1=dW1, b1=db1, W2=dW2, b2=db2)
    return loss, grads


def apply_grads(params: MLPLMParams, grads: MLPLMParams, *, lr: float) -> None:
    """경사하강법(Gradient Descent)을 이용해 파라미터를 업데이트합니다."""
    params.E -= lr * grads.E
    params.W1 -= lr * grads.W1
    params.b1 -= lr * grads.b1
    params.W2 -= lr * grads.W2
    params.b2 -= lr * grads.b2


def eval_loss(params: MLPLMParams, X: np.ndarray, y: np.ndarray, *, batch_size: int = 4096) -> float:
    """평가/검증 집합에서 평균 손실을 계산합니다."""
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
    """MLP 언어모델을 SGD로 훈련합니다."""
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
    """주어진 컨텍스트에서 다음 토큰의 확률분포를 계산합니다."""
    if context_ids.ndim != 1:
        raise ValueError("context_ids must be 1D")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits, _ = forward(params, context_ids.reshape(1, -1))
    logits = logits[0] / float(temperature)
    return softmax(logits, axis=0)


# ============================================================================
# Section 2: train_mlp_lm.py - Training script
# ============================================================================

def train_parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="미니 MLP 언어모델 학습(numpy).")
    p.add_argument("--input", required=True, help="입력 UTF-8 텍스트 파일 경로")
    p.add_argument("--out", default="llm_from_scratch/models/mlp_lm.npz", help="체크포인트 저장 경로(.npz)")
    p.add_argument("--context", type=int, default=8, help="컨텍스트 길이(k)")
    p.add_argument("--embed", type=int, default=32, help="임베딩 차원(D)")
    p.add_argument("--hidden", type=int, default=128, help="은닉 차원(H)")
    p.add_argument("--epochs", type=int, default=60, help="epoch 수")
    p.add_argument("--lr", type=float, default=0.2, help="학습률(learning rate)")
    p.add_argument("--batch", type=int, default=256, help="배치 크기(batch size)")
    p.add_argument("--seed", type=int, default=0, help="난수 시드(seed)")
    p.add_argument("--val_frac", type=float, default=0.1, help="검증 데이터 비율(0~0.5)")
    return p.parse_args()


def train_main() -> None:
    args = train_parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    if not text:
        raise ValueError("Input text is empty")

    tok = CharTokenizer.from_text(text)
    ids = np.array(tok.encode(text), dtype=np.int64)

    X, y = make_context_dataset(ids, int(args.context))

    if not (0.0 <= args.val_frac < 0.5):
        raise ValueError("--val_frac must be in [0, 0.5)")
    split = int(len(X) * (1.0 - args.val_frac))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    config = MLPLMConfig(
        context_len=int(args.context),
        embed_dim=int(args.embed),
        hidden_dim=int(args.hidden),
        lr=float(args.lr),
        epochs=int(args.epochs),
        batch_size=int(args.batch),
        seed=int(args.seed),
    )

    params, history = train_mlp_lm(
        X_train, y_train, tok.vocab_size,
        config=config,
        X_val=X_val, y_val=y_val
    )

    default_start_ids = ids[: config.context_len]
    save_mlp_lm(
        args.out,
        MLPLMCheckpoint(
            tokenizer=tok,
            context_len=config.context_len,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            params=params,
            default_start_ids=default_start_ids,
        ),
    )

    last = history[-1]
    if "val_loss" in last:
        print(f"saved={args.out}  train_loss={last['train_loss']:.4f}  val_loss={last['val_loss']:.4f}")
    else:
        print(f"saved={args.out}  train_loss={last['train_loss']:.4f}")


# ============================================================================
# Section 3: generate_mlp_lm.py - Generation script
# ============================================================================

def generate_parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLP LM으로 텍스트 생성(numpy).")
    p.add_argument("--model", default="llm_from_scratch/models/mlp_lm.npz", help="체크포인트 경로(.npz)")
    p.add_argument("--length", type=int, default=400, help="생성할 글자 수")
    p.add_argument("--seed", type=int, default=0, help="난수 시드(seed)")
    p.add_argument("--temperature", type=float, default=1.0, help="샘플링 온도(>0)")
    p.add_argument("--top_k", type=int, default=None, help="top-k 샘플링(선택)")
    p.add_argument("--top_p", type=float, default=None, help="top-p 샘플링(선택)")
    p.add_argument(
        "--start_ids",
        type=str,
        default="",
        help='시작 컨텍스트를 토큰 id로 직접 지정(쉼표 구분). 예: "1,2,3,4"',
    )
    return p.parse_args()


def generate_main() -> None:
    args = generate_parse_args()

    ckpt = load_mlp_lm(args.model)

    rng = np.random.default_rng(args.seed)
    cfg = SamplingConfig(
        temperature=float(args.temperature),
        top_k=args.top_k,
        top_p=args.top_p
    )

    if args.start_ids:
        start_ids = [int(x.strip()) for x in args.start_ids.split(",") if x.strip() != ""]
        if len(start_ids) != ckpt.context_len:
            raise ValueError(f"--start_ids must have exactly {ckpt.context_len} ids")
        if not all(0 <= token_id < ckpt.tokenizer.vocab_size for token_id in start_ids):
            raise ValueError("--start_ids contains out-of-range token id")
        context = np.array(start_ids, dtype=np.int64)
    else:
        context = ckpt.default_start_ids.copy()
    initial_context = context.copy()

    out_ids: list[int] = []
    for _ in range(args.length):
        probs = next_token_probs(ckpt.params, context, temperature=1.0)
        next_id = sample_from_probs(probs, rng, cfg=cfg)
        out_ids.append(next_id)
        context = np.roll(context, -1)
        context[-1] = next_id

    print(ckpt.tokenizer.decode(initial_context.tolist()) + ckpt.tokenizer.decode(out_ids))


# ============================================================================
# Main entry point
# ============================================================================

def main() -> None:
    import sys
    if len(sys.argv) > 1 and ("--input" in sys.argv or "train" in " ".join(sys.argv)):
        train_main()
    else:
        generate_main()


if __name__ == "__main__":
    main()
