"""Week 1 Complete: Bigram Language Model.

빅램(Bigram) 언어모델: "직전 글자 한 개" 만 보고 "다음 글자" 를 예측하는
가장 단순한 언어모델이다.

학습:
- 코퍼스(텍스트)에서 인접한 두 글자가 몇 번 같이 등장했는지 센다.
- 이 횟수를 행마다 합이 1이 되게 정규화하면 확률표가 된다.

생성:
- 현재 글자에 해당하는 확률 행을 보고, 그 분포에서 다음 글자를 무작위로 뽑는다.
- 계속 이어붙이면 텍스트가 생성된다.

실행 방법:
- 생성: python week1/code/week1_complete.py --generate
- 검사: python week1/code/week1_complete.py --inspect --char "가"
"""

import argparse
from pathlib import Path
import numpy as np


# ============================================================================
# 1. CharTokenizer: 글자 <-> 정수 ID 변환
# ============================================================================

class CharTokenizer:
    """글자 단위 토크나이저.

    예) vocab = ['a', 'b', 'c'] 라면
        encode("cab") -> [2, 0, 1]
        decode([2, 0, 1]) -> "cab"
    """

    def __init__(self, vocab):
        # vocab: 글자들의 리스트. 자바 List<String>, JS string[] 와 같다.
        # 이 리스트의 인덱스(0,1,2,...) 가 곧 토큰 ID 가 된다.
        if len(vocab) == 0:
            raise ValueError("vocab empty")
        self.vocab = vocab

        # char_to_id: "글자 -> ID" 변환표.
        # 자바 Map<String, Integer>, JS { [key:string]: number } 와 같다.
        self.char_to_id = {}
        for i in range(len(vocab)):
            ch = vocab[i]
            self.char_to_id[ch] = i

    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        """텍스트 -> 정수 ID 리스트"""
        ids = []
        for ch in text:
            ids.append(self.char_to_id[ch])
        return ids

    def decode(self, ids):
        """정수 ID 리스트 -> 텍스트"""
        chars = []
        for token_id in ids:
            chars.append(self.vocab[token_id])
        # 자바: String.join("", chars) / JS: chars.join("")
        return "".join(chars)


def build_tokenizer_from_text(text):
    """텍스트에 등장한 모든 글자로 토크나이저를 만든다."""
    # set(text): 중복 제거 (자바 HashSet, JS new Set([...text]))
    # sorted(...): 정렬된 리스트로 변환
    unique_chars = sorted(set(text))
    return CharTokenizer(unique_chars)


# ============================================================================
# 2. Bigram Model: 빈도수 세기 -> 확률표
# ============================================================================

def build_bigram_counts(token_ids, vocab_size):
    """토큰 시퀀스에서 (이전 글자, 다음 글자) 쌍의 등장 횟수를 V x V 행렬로 만든다.

    counts[i][j] = "글자 i 다음에 글자 j 가 나온 횟수"
    """
    # V x V 크기의 0 으로 채워진 정수 행렬 (자바 long[V][V])
    counts = np.zeros((vocab_size, vocab_size), dtype=np.int64)

    # 토큰 시퀀스를 한 칸씩 밀어가며 (이전 글자, 다음 글자) 쌍을 본다.
    # 예) ids = [3, 7, 2] 이면 (3,7), (7,2) 두 쌍을 본다.
    for i in range(len(token_ids) - 1):
        prev_id = token_ids[i]
        next_id = token_ids[i + 1]
        counts[prev_id][next_id] += 1

    return counts


def counts_to_probs(counts, smoothing=0.0):
    """빈도수 행렬을 확률 행렬로 변환한다 (각 행의 합이 1).

    Laplace smoothing: 0 으로 나오는 칸도 약간의 확률을 주어
    한 번도 못 본 조합에도 작은 확률을 부여한다.
    """
    # 정수 -> 실수 변환 (자바 (double[][]) 캐스트와 비슷)
    counts_f = counts.astype(np.float64)

    if smoothing > 0:
        # 모든 칸에 smoothing 값을 더함 (브로드캐스팅)
        counts_f = counts_f + smoothing

    # 각 행의 합 (V x 1 모양 행렬)
    row_sums = counts_f.sum(axis=1, keepdims=True)

    # 합이 0인 행(한 번도 등장하지 않은 글자)은 균등 분포로 채움
    for row in range(len(row_sums)):
        if row_sums[row, 0] == 0:
            counts_f[row] = 1.0
            row_sums[row, 0] = float(len(counts_f[row]))

    # 행마다 합이 1이 되도록 나눔 (각 행이 확률 분포가 됨)
    return counts_f / row_sums


def sample_next_id(probs_row, rng, temperature=1.0):
    """확률 분포에서 다음 토큰 ID 한 개를 무작위로 뽑는다.

    temperature:
    - 1.0: 학습된 확률 그대로
    - <1.0: 큰 확률에 더 쏠림 (보수적)
    - >1.0: 분포가 평평해짐 (창의적)
    """
    p = probs_row.copy()

    if temperature != 1.0:
        # 각 확률을 (1/temperature) 제곱한 뒤 다시 정규화
        p = np.power(p, 1.0 / temperature)
        p = p / p.sum()

    # rng.choice(N, p=p): 0..N-1 중 확률 p 에 따라 하나를 뽑음
    return int(rng.choice(len(p), p=p))


# ============================================================================
# 3. Main Execution Flow
# ============================================================================

def main():
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
    tokenizer = build_tokenizer_from_text(text)
    token_ids = np.array(tokenizer.encode(text))
    vocab_size = tokenizer.vocab_size()
    print(f"--- 데이터 로드 완료 (글자 수: {len(text)}, Vocab 크기: {vocab_size}) ---")

    # 2. 빅램 모델 학습 (빈도수 -> 확률)
    counts = build_bigram_counts(token_ids, vocab_size)
    probs = counts_to_probs(counts, smoothing=args.smooth)

    if args.generate:
        print(f"--- 텍스트 생성 (Temperature: {args.temp}, Smoothing: {args.smooth}) ---")
        rng = np.random.default_rng(42)

        # 시작 글자: 데이터의 첫 글자
        current_id = int(token_ids[0])
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
        # np.argsort(row): 정렬했을 때의 원래 인덱스들 (오름차순)
        # [::-1]: 역순으로 뒤집기 -> 내림차순 인덱스
        # [:10]: 앞에서 10개만 자르기
        sorted_indices_desc = np.argsort(row)[::-1]
        top_indices = sorted_indices_desc[:10]

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
