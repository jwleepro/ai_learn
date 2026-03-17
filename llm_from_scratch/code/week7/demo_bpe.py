"""BPE encode/decode를 눈으로 확인하는 데모 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from bpe_tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BPE encode/decode 데모.")
    p.add_argument("--tokenizer", required=True, help="토크나이저 JSON 경로")
    p.add_argument("--text_file", required=True, help="인코딩할 텍스트 파일 경로")
    p.add_argument("--max_tokens", type=int, default=60, help="앞에서부터 N개 토큰 출력")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tok = BPETokenizer.load_json(args.tokenizer)
    text = Path(args.text_file).read_text(encoding="utf-8")
    tokens = tok.encode_tokens(text)
    ids = tok.encode(text)
    print(f"tokens={len(tokens)}  ids={len(ids)}  vocab_size={tok.vocab_size}")
    shown = tokens[: int(args.max_tokens)]
    print("first tokens:")
    for i, t in enumerate(shown):
        print(f"  [{i:02d}] {t!r}")
    print("")
    print("decode (from ids):")
    print(tok.decode(ids[: 200]))


if __name__ == "__main__":
    main()
