"""카운트 기반 빅램(글자 단위)으로 텍스트 생성하는 CLI.

학습(카운트/확률표 만들기)과 생성을 한 번에 수행합니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bigram_counts import build_bigram_counts, counts_to_probs
from sampling import SamplingConfig, sample_from_probs
from tokenizer_char import CharTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="카운트 기반 빅램 언어모델로 텍스트 생성(글자 단위).")
    p.add_argument("--input", required=True, help="입력 UTF-8 텍스트 파일 경로")
    p.add_argument("--length", type=int, default=400, help="생성할 글자 수")
    p.add_argument("--seed", type=int, default=0, help="난수 시드(seed)")
    p.add_argument("--start", type=str, default="", help="시작 텍스트(마지막 글자를 컨텍스트로 사용)")
    p.add_argument("--smoothing", type=float, default=0.0, help="Add-k smoothing (0=끄기)")
    p.add_argument("--temperature", type=float, default=1.0, help="샘플링 온도(>0)")
    p.add_argument("--top_k", type=int, default=None, help="top-k 샘플링(k개 후보만 유지; 선택)")
    p.add_argument("--top_p", type=float, default=None, help="top-p(nucleus) 샘플링(0<p<=1; 선택)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    text_path = Path(args.input)
    text = text_path.read_text(encoding="utf-8")
    if not text:
        raise ValueError("Input text is empty")

    tokenizer = CharTokenizer.from_text(text)
    token_ids = np.array(tokenizer.encode(text), dtype=np.int64)

    counts = build_bigram_counts(token_ids, tokenizer.vocab_size)
    probs = counts_to_probs(counts, smoothing=args.smoothing)

    start_text = args.start if args.start else text[:1]
    start_ids = tokenizer.encode(start_text)
    prev_id = start_ids[-1]

    rng = np.random.default_rng(args.seed)
    cfg = SamplingConfig(temperature=float(args.temperature), top_k=args.top_k, top_p=args.top_p)
    out_ids: list[int] = []
    for _ in range(args.length):
        next_id = sample_from_probs(probs[prev_id], rng, cfg=cfg)
        out_ids.append(next_id)
        prev_id = next_id

    print(start_text + tokenizer.decode(out_ids))


if __name__ == "__main__":
    main()
