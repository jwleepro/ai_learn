"""언어모델 평가(loss, perplexity) CLI.

지원:
- counts_bigram: 카운트 기반 빅램
- bigram_nn: 신경망 빅램 체크포인트(.npz)
- mlp_lm: MLP LM 체크포인트(.npz)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bigram_counts import build_bigram_counts, counts_to_probs
from bigram_nn import eval_loss as eval_loss_bigram_nn
from dataset_lm import make_context_dataset
from mlp_lm import eval_loss as eval_loss_mlp
from model_io import load_bigram_nn, load_mlp_lm
from tokenizer_char import CharTokenizer


def perplexity(loss: float) -> float:
    return float(np.exp(loss))


def add_subcommands(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="cmd", required=True)

    p_counts = sub.add_parser("counts_bigram", help="카운트 기반 빅램 모델 평가")
    p_counts.add_argument("--train", required=True, help="학습 텍스트(카운트/확률표를 이 파일로 만듦)")
    p_counts.add_argument("--eval", required=True, help="평가(eval) 텍스트 파일")
    p_counts.add_argument("--smoothing", type=float, default=0.0, help="Add-k smoothing (0=끄기)")

    p_bnn = sub.add_parser("bigram_nn", help="신경망 빅램 체크포인트 평가")
    p_bnn.add_argument("--model", required=True, help="bigram_nn 체크포인트 경로(.npz)")
    p_bnn.add_argument("--eval", required=True, help="평가(eval) 텍스트 파일")

    p_mlp = sub.add_parser("mlp_lm", help="MLP LM 체크포인트 평가")
    p_mlp.add_argument("--model", required=True, help="mlp_lm 체크포인트 경로(.npz)")
    p_mlp.add_argument("--eval", required=True, help="평가(eval) 텍스트 파일")


def main() -> None:
    p = argparse.ArgumentParser(description="언어모델 평가(loss + perplexity).")
    add_subcommands(p)
    args = p.parse_args()

    if args.cmd == "counts_bigram":
        train_text = Path(args.train).read_text(encoding="utf-8")
        eval_text = Path(args.eval).read_text(encoding="utf-8")
        tok = CharTokenizer.from_text(train_text)
        train_ids = np.array(tok.encode(train_text), dtype=np.int64)
        counts = build_bigram_counts(train_ids, tok.vocab_size)
        probs = counts_to_probs(counts, smoothing=float(args.smoothing))

        eval_ids = np.array(tok.encode(eval_text), dtype=np.int64)
        if len(eval_ids) < 2:
            raise ValueError("Eval text must contain at least 2 tokens/characters")
        prev_ids = eval_ids[:-1]
        next_ids = eval_ids[1:]
        p_next = probs[prev_ids, next_ids]
        if np.any(p_next == 0.0):
            zero = int((p_next == 0.0).sum())
            print(f"loss=inf  ppl=inf  (zero_prob_pairs={zero}; try --smoothing 1)")
            return
        loss = float(-np.log(p_next).mean())
        print(f"loss={loss:.4f}  ppl={perplexity(loss):.2f}")
        return

    if args.cmd == "bigram_nn":
        ckpt = load_bigram_nn(args.model)
        eval_text = Path(args.eval).read_text(encoding="utf-8")
        eval_ids = np.array(ckpt.tokenizer.encode(eval_text), dtype=np.int64)
        if len(eval_ids) < 2:
            raise ValueError("Eval text must contain at least 2 tokens/characters")
        prev_ids = eval_ids[:-1]
        next_ids = eval_ids[1:]
        loss = float(eval_loss_bigram_nn(ckpt.W, prev_ids, next_ids))
        print(f"loss={loss:.4f}  ppl={perplexity(loss):.2f}")
        return

    if args.cmd == "mlp_lm":
        ckpt = load_mlp_lm(args.model)
        eval_text = Path(args.eval).read_text(encoding="utf-8")
        eval_ids = np.array(ckpt.tokenizer.encode(eval_text), dtype=np.int64)
        X, y = make_context_dataset(eval_ids, ckpt.context_len)
        loss = float(eval_loss_mlp(ckpt.params, X, y))
        print(f"loss={loss:.4f}  ppl={perplexity(loss):.2f}")
        return

    raise AssertionError("unreachable")


if __name__ == "__main__":
    main()
