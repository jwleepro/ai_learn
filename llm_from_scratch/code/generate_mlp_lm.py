"""학습된 MLP LM 체크포인트(.npz)로 텍스트를 생성하는 CLI."""

from __future__ import annotations

import argparse

import numpy as np

from mlp_lm import next_token_probs
from model_io import load_mlp_lm
from sampling import SamplingConfig, sample_from_probs


def parse_args() -> argparse.Namespace:
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


def main() -> None:
    args = parse_args()
    ckpt = load_mlp_lm(args.model)
    rng = np.random.default_rng(args.seed)
    cfg = SamplingConfig(temperature=float(args.temperature), top_k=args.top_k, top_p=args.top_p)

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
        probs = next_token_probs(ckpt.params, context, temperature=1.0)  # temperature handled in sampler
        next_id = sample_from_probs(probs, rng, cfg=cfg)
        out_ids.append(next_id)
        context = np.roll(context, -1)
        context[-1] = next_id

    print(ckpt.tokenizer.decode(initial_context.tolist()) + ckpt.tokenizer.decode(out_ids))


if __name__ == "__main__":
    main()
