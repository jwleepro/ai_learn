"""BPE 토크나이저를 학습하고 JSON으로 저장하는 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from bpe_tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="(학습용) 단순 BPE 토크나이저 학습.")
    p.add_argument("--input", required=True, help="입력 UTF-8 텍스트 파일 경로")
    p.add_argument("--out", default="llm_from_scratch/models/bpe_tokenizer.json", help="출력 JSON 경로")
    p.add_argument("--merges", type=int, default=200, help="merge 반복 횟수")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    text = Path(args.input).read_text(encoding="utf-8")
    tok = BPETokenizer.train(text, num_merges=int(args.merges))
    tok.save_json(args.out)
    print(f"saved={args.out}  vocab_size={tok.vocab_size}  merges={len(tok.merges)}")


if __name__ == "__main__":
    main()
