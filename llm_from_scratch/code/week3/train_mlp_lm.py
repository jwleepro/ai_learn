"""MLP LM을 학습하고 체크포인트(.npz)로 저장하는 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dataset_lm import make_context_dataset
from mlp_lm import MLPLMConfig, train_mlp_lm
from model_io import MLPLMCheckpoint, save_mlp_lm
from tokenizer_char import CharTokenizer


def parse_args() -> argparse.Namespace:
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


def main() -> None:
    args = parse_args()
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
    params, history = train_mlp_lm(X_train, y_train, tok.vocab_size, config=config, X_val=X_val, y_val=y_val)

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


if __name__ == "__main__":
    main()
