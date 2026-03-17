"""학습된 신경망 빅램 체크포인트(.npz)로 텍스트를 생성하는 CLI."""

from __future__ import annotations

import argparse

import numpy as np

from bigram_nn import bigram_probs
from model_io import load_bigram_nn
from sampling import SamplingConfig, sample_from_probs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="신경망 빅램 LM으로 텍스트 생성(numpy).")
    p.add_argument("--model", default="llm_from_scratch/models/bigram_nn.npz", help="체크포인트 경로(.npz)")
    p.add_argument("--length", type=int, default=400, help="생성할 글자 수")
    p.add_argument("--seed", type=int, default=0, help="난수 시드(seed)")
    p.add_argument("--start_id", type=int, default=None, help="시작 토큰 id(콘솔 인코딩 이슈 회피용)")
    p.add_argument("--temperature", type=float, default=1.0, help="샘플링 온도(>0)")
    p.add_argument("--top_k", type=int, default=None, help="top-k 샘플링(선택)")
    p.add_argument("--top_p", type=float, default=None, help="top-p 샘플링(선택)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = load_bigram_nn(args.model)

    rng = np.random.default_rng(args.seed)
    cfg = SamplingConfig(temperature=float(args.temperature), top_k=args.top_k, top_p=args.top_p)
    if args.start_id is None:
        prev_id = int(ckpt.default_start_id)
        start_text = ckpt.tokenizer.vocab[prev_id]
    else:
        prev_id = int(args.start_id)
        if not (0 <= prev_id < ckpt.tokenizer.vocab_size):
            raise ValueError("--start_id out of range for vocab")
        start_text = ckpt.tokenizer.vocab[prev_id]

    out_ids: list[int] = []
    for _ in range(args.length):
        probs = bigram_probs(ckpt.W, prev_id, temperature=1.0)  # temperature handled in sampler
        next_id = sample_from_probs(probs, rng, cfg=cfg)
        out_ids.append(next_id)
        prev_id = next_id

    print(start_text + ckpt.tokenizer.decode(out_ids))


if __name__ == "__main__":
    main()
