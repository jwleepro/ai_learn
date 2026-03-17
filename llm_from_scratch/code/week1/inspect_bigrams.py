"""빅램 전이 확률 P(next|prev)를 콘솔로 확인하는 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bigram_counts import build_bigram_counts, counts_to_probs
from tokenizer_char import CharTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="빅램 전이 확률을 확인합니다.")
    p.add_argument("--input", required=True, help="입력 UTF-8 텍스트 파일 경로")
    p.add_argument("--char", type=str, default="", help="이전 글자(prev) (정확히 1글자)")
    p.add_argument(
        "--char_u",
        type=str,
        default="",
        help='이전 글자의 유니코드 코드포인트(예: "다"는 0xB2E4)',
    )
    p.add_argument("--char_id", type=int, default=None, help="이전 토큰 id(prev_id)")
    p.add_argument("--top", type=int, default=10, help="상위 N개 next 후보 출력")
    p.add_argument("--smoothing", type=float, default=0.0, help="Add-k smoothing (0=끄기)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    text = Path(args.input).read_text(encoding="utf-8")
    if not text:
        raise ValueError("Input text is empty")

    tok = CharTokenizer.from_text(text)
    ids = np.array(tok.encode(text), dtype=np.int64)
    counts = build_bigram_counts(ids, tok.vocab_size)
    probs = counts_to_probs(counts, smoothing=args.smoothing)

    provided = int(bool(args.char)) + int(bool(args.char_u)) + int(args.char_id is not None)
    if provided > 1:
        raise ValueError("Use only one of --char / --char_u / --char_id")

    if args.char or args.char_u or args.char_id is not None:
        if args.char_u:
            codepoint = int(args.char_u, 0)
            prev_ch = chr(codepoint)
            prev_id = tok.encode(prev_ch)[0]
        elif args.char_id is not None:
            if not (0 <= args.char_id < tok.vocab_size):
                raise ValueError("--char_id out of range for this vocab")
            prev_id = int(args.char_id)
            prev_ch = tok.vocab[prev_id]
        else:
            if len(args.char) != 1:
                raise ValueError("--char must be exactly 1 character")
            prev_ch = args.char
            prev_id = tok.encode(prev_ch)[0]

        row = probs[prev_id]
        top_n = min(args.top, tok.vocab_size)
        top_ids = np.argsort(row)[-top_n:][::-1]
        print(f"prev={prev_ch!r} (U+{ord(prev_ch):04X}, id={prev_id}, vocab_size={tok.vocab_size})")
        for token_id in top_ids:
            ch = tok.vocab[int(token_id)]
            p = float(row[int(token_id)])
            print(f"  next={ch!r}  p={p:.4f}")
        return

    # Global top transitions (roughly): show prev char and best next char.
    best_next = probs.argmax(axis=1)
    best_p = probs.max(axis=1)
    order = np.argsort(best_p)[::-1]
    top_n = min(args.top, tok.vocab_size)
    print(f"Top {top_n} strongest transitions (by max P(next|prev))")
    for i in range(top_n):
        prev_id = int(order[i])
        next_id = int(best_next[prev_id])
        prev_ch = tok.vocab[prev_id]
        next_ch = tok.vocab[next_id]
        p = float(best_p[prev_id])
        print(f"  {prev_ch!r} -> {next_ch!r}  p={p:.4f}")


if __name__ == "__main__":
    main()
