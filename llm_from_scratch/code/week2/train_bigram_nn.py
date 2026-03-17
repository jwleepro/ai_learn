"""신경망 빅램 모델을 학습하고 체크포인트(.npz)로 저장하는 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bigram_nn import BigramNNConfig, train_bigram_nn
from model_io import BigramNNCheckpoint, save_bigram_nn
from tokenizer_char import CharTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="신경망 빅램 LM 학습(numpy).")
    p.add_argument("--input", required=True, help="입력 UTF-8 텍스트 파일 경로")
    p.add_argument("--out", default="llm_from_scratch/models/bigram_nn.npz", help="체크포인트 저장 경로(.npz)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2.0, help="학습률(learning rate)")
    p.add_argument("--batch", type=int, default=2048, help="배치 크기(batch size)")
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
    prev_ids = ids[:-1]
    next_ids = ids[1:]

    if not (0.0 <= args.val_frac < 0.5):
        raise ValueError("--val_frac must be in [0, 0.5)")
    split = int(len(prev_ids) * (1.0 - args.val_frac))
    prev_train, prev_val = prev_ids[:split], prev_ids[split:]
    next_train, next_val = next_ids[:split], next_ids[split:]

    config = BigramNNConfig(
        lr=float(args.lr),
        epochs=int(args.epochs),
        batch_size=int(args.batch),
        seed=int(args.seed),
    )
    W, history = train_bigram_nn(
        prev_train,
        next_train,
        tok.vocab_size,
        config=config,
        prev_val=prev_val,
        next_val=next_val,
    )

    save_bigram_nn(args.out, BigramNNCheckpoint(tokenizer=tok, W=W, default_start_id=int(ids[0])))

    last = history[-1]
    if "val_loss" in last:
        print(f"saved={args.out}  train_loss={last['train_loss']:.4f}  val_loss={last['val_loss']:.4f}")
    else:
        print(f"saved={args.out}  train_loss={last['train_loss']:.4f}")


if __name__ == "__main__":
    main()
