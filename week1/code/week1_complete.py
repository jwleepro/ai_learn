"""Week 1 Complete: Bigram Language Model.

이 파일은 빅램(Bigram) 언어모델의 모든 과정(토크나이저, 빈도수 계산, 확률 변환, 생성)을
하나의 파일에서 순서대로 읽고 실행할 수 있도록 통합한 교육용 코드입니다.

주요 내용:
1. CharTokenizer: 글자 단위 토크나이저
2. Bigram Counts: 인접한 두 글자의 등장 빈도 계산
3. Bigram Probs: 빈도수를 확률로 변환 (Laplace Smoothing 포함)
4. Generation: 확률 분포를 이용한 다음 글자 생성 (Temperature 조절)

실행 방법:
- 생성: python week1/code/week1_complete.py --generate
- 검사: python week1/code/week1_complete.py --inspect --char "가"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np


# ============================================================================
# 1. Utility Functions & Classes
# ============================================================================

@dataclass
class CharTokenizer:
    """글자 단위 토크나이저."""
    vocab: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.vocab) == 0:
            raise ValueError("vocab empty")
        self.char_to_id = {ch: i for i, ch in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls(tuple(sorted(set(text))))

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids)


# ============================================================================
# 2. Bigram Model Implementation
# ============================================================================

def build_bigram_counts(token_ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """토큰 시퀀스에서 빅램(두 글자 쌍) 등장 횟수를 행렬로 계산합니다."""
    counts = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    prev_ids = token_ids[:-1]
    next_ids = token_ids[1:]
    # np.add.at은 중복된 인덱스에 대해서도 올바르게 누적 합산을 수행합니다.
    np.add.at(counts, (prev_ids, next_ids), 1)
    return counts


def counts_to_probs(counts: np.ndarray, smoothing: float = 0.0) -> np.ndarray:
    """빈도수 행렬을 확률 행렬로 변환합니다. (Laplace Smoothing 지원)"""
    counts_f = counts.astype(np.float64)
    if smoothing > 0:
        counts_f += smoothing
    
    # 각 행의 합이 1이 되도록 정규화
    row_sums = counts_f.sum(axis=1, keepdims=True)
    # 합이 0인 행(한 번도 등장하지 않은 글자)은 균등 분포로 설정
    zero_rows = (row_sums.squeeze() == 0)
    if np.any(zero_rows):
        counts_f[zero_rows] = 1.0
        row_sums = counts_f.sum(axis=1, keepdims=True)
    
    return counts_f / row_sums


def sample_next_id(probs_row: np.ndarray, rng: np.random.Generator, temperature: float = 1.0) -> int:
    """주어진 확률 분포에서 다음 토큰 ID를 샘플링합니다."""
    p = probs_row.copy()
    if temperature != 1.0:
        # 온도가 낮을수록(0에 가까울수록) 확률이 높은 쪽에 더 쏠리게 됩니다.
        p = np.power(p, 1.0 / temperature)
        p /= p.sum()
    return int(rng.choice(len(p), p=p))


# ============================================================================
# 3. Main Execution Flow
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="텍스트 생성 실행")
    parser.add_argument("--inspect", action="store_true", help="특정 글자 뒤의 확률 확인")
    parser.add_argument("--char", default=" ", help="검사할 글자")
    parser.add_argument("--data", default="week1/data/tiny_corpus_ko.txt", help="데이터 경로")
    parser.add_argument("--length", type=int, default=100, help="생성할 글자 수")
    parser.add_argument("--temp", type=float, default=1.0, help="샘플링 온도")
    parser.add_argument("--smooth", type=float, default=0.1, help="스무딩 계수")
    args = parser.parse_args()

    # 데이터 로드
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"데이터 파일이 없습니다: {data_path}")
        return
    text = data_path.read_text(encoding="utf-8")
    
    # 1. 토크나이저 준비
    tokenizer = CharTokenizer.from_text(text)
    token_ids = np.array(tokenizer.encode(text))
    V = tokenizer.vocab_size
    print(f"--- 데이터 로드 완료 (글자 수: {len(text)}, Vocab 크기: {V}) ---")

    # 2. 빅램 모델 학습 (빈도수 계산 및 확률 변환)
    counts = build_bigram_counts(token_ids, V)
    probs = counts_to_probs(counts, smoothing=args.smooth)

    if args.generate:
        print(f"--- 텍스트 생성 (Temperature: {args.temp}, Smoothing: {args.smooth}) ---")
        rng = np.random.default_rng(42)
        
        # 시작 글자: 데이터의 첫 글자
        current_id = token_ids[0]
        generated_ids = [current_id]
        
        for _ in range(args.length):
            next_id = sample_next_id(probs[current_id], rng, temperature=args.temp)
            generated_ids.append(next_id)
            current_id = next_id
        
        print(f"결과: {tokenizer.decode(generated_ids)}")

    elif args.inspect:
        if args.char not in tokenizer.char_to_id:
            print(f"글자 '{args.char}'는 학습 데이터에 없습니다.")
            return
        
        char_id = tokenizer.char_to_id[args.char]
        row = probs[char_id]
        
        # 확률이 높은 상위 10개 출력
        top_indices = np.argsort(row)[::-1][:10]
        print(f"--- '{args.char}' 뒤에 올 글자 확률 (상위 10개) ---")
        for idx in top_indices:
            next_char = tokenizer.vocab[idx]
            p = row[idx]
            if p > 0:
                print(f"  '{next_char}': {p:.4f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
