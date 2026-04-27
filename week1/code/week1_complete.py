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

from __future__ import annotations  # 파이썬 전용: 타입 힌트를 문자열로 평가(전방 참조 가능)

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np


# ============================================================================
# 1. Utility Functions & Classes
# ============================================================================

# @dataclass: 데코레이터(자바 어노테이션과 표기는 비슷하지만 클래스를 실제로 변형).
# 아래 `vocab: tuple[str, ...]` 같은 필드 선언만으로 __init__ 등이 자동 생성된다.
# tuple[str, ...] 의 `...`(Ellipsis)는 "길이가 가변인 동질 튜플"을 뜻한다.
@dataclass
class CharTokenizer:
    """글자 단위 토크나이저."""
    vocab: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.vocab) == 0:
            raise ValueError("vocab empty")
        # 딕셔너리 컴프리헨션: 자바/C#에는 없는 문법.
        # enumerate(x) 는 (index, value) 쌍을 돌려준다 → for (i, ch) 형태로 분해(언패킹).
        self.char_to_id = {ch: i for i, ch in enumerate(self.vocab)}

    # @property: 메서드를 속성처럼 호출할 수 있게 한다 (tokenizer.vocab_size, 괄호 없이).
    # C#의 get-only property 와 거의 같음. 자바는 getter 메서드로 풀어야 함.
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # @classmethod: 첫 인자가 인스턴스(self)가 아니라 클래스 자체(cls)인 메서드.
    # 자바/C# 의 static factory 메서드와 비슷하지만, cls 를 통해 서브클래스도 자동 지원.
    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        # set(text) 는 중복 제거, sorted(...) 는 정렬된 리스트로 만든 뒤 tuple(...) 로 불변화.
        return cls(tuple(sorted(set(text))))

    def encode(self, text: str) -> list[int]:
        # 리스트 컴프리헨션 [식 for x in 시퀀스]. 자바/C# 에는 없는 문법.
        # 자바의 stream().map(...).toList() 또는 C# 의 LINQ Select(...).ToList() 와 의미가 같다.
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        # 괄호 없이 쓴 `self.vocab[i] for i in ids` 는 제너레이터 식(lazy 시퀀스).
        # str.join 은 자바 String.join, C# string.Join 과 같다.
        return "".join(self.vocab[i] for i in ids)


# ============================================================================
# 2. Bigram Model Implementation
# ============================================================================

def build_bigram_counts(token_ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """토큰 시퀀스에서 빅램(두 글자 쌍) 등장 횟수를 행렬로 계산합니다."""
    counts = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    # 슬라이싱(파이썬/넘파이 전용): `[:-1]` 은 "마지막 1개 빼고 전부", `[1:]` 은 "첫 1개 빼고 전부".
    # 자바/C# 에는 직접 대응되는 문법이 없어 보통 subList/Skip 등으로 풀어야 한다.
    prev_ids = token_ids[:-1]
    next_ids = token_ids[1:]
    # np.add.at 은 넘파이 함수로, 중복된 인덱스에 대해서도 올바르게 누적 합산을 수행한다.
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
        # `[::-1]` 은 슬라이싱의 step=-1, 즉 "역순". `[:10]` 은 앞에서 10개. 자바/C# 에는 없는 표기.
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
