"""Week 1: 빅램 모델 - 완전한 예제 코드

이 파일은 Week 1의 모든 코드를 하나의 파일로 통합한 것입니다.
- 빅램 확률분포 계산
- 텍스트 생성 (온도, 스무딩 조절)
- 전이 확률 검사
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

# ============================================================================
# PART 1: 빅램 확률분포 계산 (bigram_counts.py)
# ============================================================================

def build_bigram_counts(token_ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """토큰 시퀀스에서 빅램 등장 횟수 계산"""
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be 1D")
    if len(token_ids) < 2:
        raise ValueError("token_ids must contain at least 2 tokens")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")

    prev_ids = token_ids[:-1]
    next_ids = token_ids[1:]

    counts = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    np.add.at(counts, (prev_ids, next_ids), 1)
    return counts


def counts_to_probs(counts: np.ndarray, *, smoothing: float = 0.0) -> np.ndarray:
    """빅램 등장 횟수를 확률분포로 변환 (Laplace smoothing 지원)"""
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("counts must be a square 2D matrix")
    if smoothing < 0:
        raise ValueError("smoothing must be >= 0")

    counts_f = counts.astype(np.float64, copy=False)
    if smoothing != 0.0:
        counts_f = counts_f + smoothing

    row_sums = counts_f.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze(axis=1) == 0
    if np.any(zero_rows):
        counts_f = counts_f.copy()
        counts_f[zero_rows, :] = 1.0
        row_sums = counts_f.sum(axis=1, keepdims=True)

    return counts_f / row_sums


def sample_next_id(
    prev_id: int,
    probs: np.ndarray,
    rng: np.random.Generator,
    *,
    temperature: float = 1.0,
) -> int:
    """온도를 조절한 표본추출"""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    row = probs[prev_id]
    if temperature != 1.0:
        row = row ** (1.0 / temperature)
        row = row / row.sum()

    return int(rng.choice(len(row), p=row))


# ============================================================================
# PART 2: 텍스트 생성 (generate_bigram.py)
# ============================================================================

def generate_text(
    text: str,
    length: int = 300,
    seed: int = 0,
    smoothing: float = 0.0,
    temperature: float = 1.0,
    start: str = "",
) -> str:
    """빅램으로 텍스트 생성"""
    from tokenizer_char import CharTokenizer

    # 토크나이저 생성
    tokenizer = CharTokenizer.from_text(text)
    token_ids = np.array(tokenizer.encode(text), dtype=np.int64)

    # 빅램 계산
    counts = build_bigram_counts(token_ids, tokenizer.vocab_size)
    probs = counts_to_probs(counts, smoothing=smoothing)

    # 생성 시작
    start_text = start if start else text[:1]
    start_ids = tokenizer.encode(start_text)
    prev_id = start_ids[-1]

    # 생성 루프
    rng = np.random.default_rng(seed)
    out_ids: list[int] = []
    for _ in range(length):
        next_id = sample_next_id(probs, prev_id, rng, temperature=temperature)
        out_ids.append(next_id)
        prev_id = next_id

    return start_text + tokenizer.decode(out_ids)


# ============================================================================
# PART 3: 전이 확률 검사 (inspect_bigrams.py)
# ============================================================================

def inspect_bigrams(text: str, prev_char: str, top_n: int = 10) -> list[tuple[str, float]]:
    """특정 글자 이후의 전이 확률 조회"""
    from tokenizer_char import CharTokenizer

    tokenizer = CharTokenizer.from_text(text)
    token_ids = np.array(tokenizer.encode(text), dtype=np.int64)

    counts = build_bigram_counts(token_ids, tokenizer.vocab_size)
    probs = counts_to_probs(counts, smoothing=0.0)

    # 글자를 ID로 변환
    prev_id = tokenizer.encode(prev_char)[0]

    # 상위 next 후보 찾기
    row = probs[prev_id]
    top_indices = np.argsort(row)[-top_n:][::-1]

    result = []
    for idx in top_indices:
        next_char = tokenizer.vocab[int(idx)]
        prob = float(row[int(idx)])
        result.append((next_char, prob))

    return result


# ============================================================================
# MAIN: CLI 인터페이스
# ============================================================================

def main():
    """커맨드라인 인터페이스"""
    p = argparse.ArgumentParser(description="Week 1: 빅램 모델 완전한 예제")
    subparsers = p.add_subparsers(dest="command", help="명령어")

    # generate 커맨드
    gen_parser = subparsers.add_parser("generate", help="텍스트 생성")
    gen_parser.add_argument("--input", required=True, help="입력 텍스트 파일")
    gen_parser.add_argument("--length", type=int, default=300, help="생성할 글자 수")
    gen_parser.add_argument("--seed", type=int, default=0, help="난수 시드")
    gen_parser.add_argument("--smoothing", type=float, default=0.0, help="스무딩")
    gen_parser.add_argument("--temperature", type=float, default=1.0, help="온도")
    gen_parser.add_argument("--start", type=str, default="", help="시작 텍스트")

    # inspect 커맨드
    insp_parser = subparsers.add_parser("inspect", help="전이 확률 검사")
    insp_parser.add_argument("--input", required=True, help="입력 텍스트 파일")
    insp_parser.add_argument("--char", required=True, help="확인할 글자")
    insp_parser.add_argument("--top", type=int, default=10, help="상위 N개")

    args = p.parse_args()

    if args.command == "generate":
        text = Path(args.input).read_text(encoding="utf-8")
        generated = generate_text(
            text,
            length=args.length,
            seed=args.seed,
            smoothing=args.smoothing,
            temperature=args.temperature,
            start=args.start,
        )
        print(generated)

    elif args.command == "inspect":
        text = Path(args.input).read_text(encoding="utf-8")
        results = inspect_bigrams(text, args.char, top_n=args.top)
        print(f"prev={args.char!r}의 다음 글자 확률 (상위 {args.top}개):")
        for next_char, prob in results:
            print(f"  {next_char!r}: {prob:.4f}")

    else:
        p.print_help()


if __name__ == "__main__":
    main()
