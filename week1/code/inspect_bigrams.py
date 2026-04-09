"""빅램 전이 확률 P(next|prev)를 콘솔로 확인하는 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _path_setup  # noqa: F401  (code/ 하위 폴더 sys.path 등록)
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
    """
    빅램 전이 확률을 콘솔에서 대화형으로 검사

    두 가지 모드:
    1. --char/--char_u/--char_id 지정: 특정 글자 이후 가능한 다음 글자들을 확률과 함께 표시
    2. 지정하지 않음: 전체 전이 중 확률이 가장 높은 것들을 표시
    """
    args = parse_args()
    text = Path(args.input).read_text(encoding="utf-8")
    if not text:
        raise ValueError("Input text is empty")

    # 토크나이저 구축 및 빅램 계산
    tok = CharTokenizer.from_text(text)
    ids = np.array(tok.encode(text), dtype=np.int64)
    counts = build_bigram_counts(ids, tok.vocab_size)
    probs = counts_to_probs(counts, smoothing=args.smoothing)

    # 검증: --char, --char_u, --char_id 중 최대 1개만 지정
    provided = int(bool(args.char)) + int(bool(args.char_u)) + int(args.char_id is not None)
    if provided > 1:
        raise ValueError("Use only one of --char / --char_u / --char_id")

    # ===== 모드 1: 특정 글자 이후의 전이 확률 조회 =====
    if args.char or args.char_u or args.char_id is not None:
        # 3가지 입력 방식 중 하나를 선택된 글자로 변환
        if args.char_u:
            # 유니코드 코드포인트로 지정 (예: 0xB2E4 = '다')
            codepoint = int(args.char_u, 0)
            prev_ch = chr(codepoint)
            prev_id = tok.encode(prev_ch)[0]
        elif args.char_id is not None:
            # 토큰 ID로 직접 지정
            if not (0 <= args.char_id < tok.vocab_size):
                raise ValueError("--char_id out of range for this vocab")
            prev_id = int(args.char_id)
            prev_ch = tok.vocab[prev_id]
        else:
            # 문자 직접 지정 (예: '다')
            if len(args.char) != 1:
                raise ValueError("--char must be exactly 1 character")
            prev_ch = args.char
            prev_id = tok.encode(prev_ch)[0]

        # 선택한 글자 이후의 확률분포 추출
        row = probs[prev_id]

        # 상위 top_n개의 다음 글자 추출
        # argsort: 확률을 오름차순으로 정렬한 인덱스
        # [-top_n:][::-1]: 마지막 top_n개를 역순으로 (내림차순)
        top_n = min(args.top, tok.vocab_size)
        top_ids = np.argsort(row)[-top_n:][::-1]

        # 결과 출력
        print(f"prev={prev_ch!r} (U+{ord(prev_ch):04X}, id={prev_id}, vocab_size={tok.vocab_size})")
        for token_id in top_ids:
            ch = tok.vocab[int(token_id)]
            p = float(row[int(token_id)])
            print(f"  next={ch!r}  p={p:.4f}")
        return

    # ===== 모드 2: 전체 전이 중 확률이 가장 높은 것들 =====
    # 각 행(이전 글자)에 대해 가장 확률 높은 다음 글자 찾기
    best_next = probs.argmax(axis=1)  # 각 이전 글자마다 가장 확률 높은 다음 글자 인덱스
    best_p = probs.max(axis=1)        # 각 이전 글자마다 그 최고 확률값

    # 최고 확률들을 내림차순으로 정렬
    order = np.argsort(best_p)[::-1]

    # 상위 top_n개 전이 표시
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
