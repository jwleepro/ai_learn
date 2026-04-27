"""Week 6 Complete: Language Model Evaluation.

이 파일은 학습된 언어모델의 성능을 측정하는 과정(Loss, Perplexity)을
하나의 파일에서 순서대로 읽고 실행할 수 있도록 통합한 교육용 코드입니다.

주요 내용:
1. CharTokenizer: 글자 단위 토크나이저
2. Metrics: Cross Entropy Loss 및 Perplexity (PPL) 계산
3. Evaluation: 
   - Count-based Bigram 평가
   - Neural Bigram 평가
   - MLP LM 평가

실행 방법:
- Count Bigram 평가: python week6/code/week6_complete.py --counts --data week6/data/tiny_corpus_ko.txt
- Neural Bigram 평가: python week6/code/week6_complete.py --bigram_nn --model_path week2/code/bigram_nn_model.npz --data week6/data/tiny_corpus_ko.txt
- MLP LM 평가: python week6/code/week6_complete.py --mlp_lm --model_path week3/code/mlp_model.npz --data week6/data/tiny_corpus_ko.txt
"""

from __future__ import annotations  # 파이썬 전용: 타입 힌트를 문자열로 평가

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np


# ============================================================================
# 1. Utility Functions & Classes
# ============================================================================

# CharTokenizer: 파이썬 전용 문법(@dataclass, @property, @classmethod,
# tuple[str, ...] 빌트인 제네릭, 컴프리헨션, 튜플 언패킹)을 사용한다.
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


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def calculate_metrics(logits: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """Loss와 Perplexity를 계산합니다.

    반환 타입 `tuple[float, float]` 는 두 값을 묶어 돌려준다는 의미.
    호출 측에서 `loss, ppl = calculate_metrics(...)` 처럼 튜플 언패킹으로 받을 수 있다.
    """
    probs = softmax(logits, axis=-1)
    # 정답 토큰의 확률 추출 (넘파이 advanced indexing: 행 인덱스와 열 인덱스를 동시에 지정)
    target_probs = probs[np.arange(len(targets)), targets]
    # Numerical stability를 위해 아주 작은 값을 더해줌
    loss = -np.log(target_probs + 1e-10).mean()
    ppl = np.exp(loss)
    return float(loss), float(ppl)


# ============================================================================
# 2. Main Execution Flow
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", action="store_true", help="Count Bigram 평가")
    parser.add_argument("--bigram_nn", action="store_true", help="Neural Bigram 평가")
    parser.add_argument("--mlp_lm", action="store_true", help="MLP LM 평가")
    parser.add_argument("--data", default="week6/data/tiny_corpus_ko.txt", help="평가용 데이터 경로")
    parser.add_argument("--model_path", help="평가할 모델 파일 경로 (.npz)")
    args = parser.parse_args()

    # 평가 데이터 로드
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"데이터 파일이 없습니다: {data_path}")
        return
    eval_text = data_path.read_text(encoding="utf-8")

    if args.counts:
        print("--- Count-based Bigram Evaluation ---")
        tokenizer = CharTokenizer.from_text(eval_text)
        ids = np.array(tokenizer.encode(eval_text))
        V = tokenizer.vocab_size

        # 훈련 데이터로부터 카운트를 구해야 하지만, 데모를 위해 동일 데이터를 씀
        counts = np.zeros((V, V))
        # 슬라이싱: `ids[:-1]` 은 마지막 빼고 전부, `ids[1:]` 은 첫 번째 빼고 전부.
        # 두 배열을 함께 인덱스로 넘기면 (이전 글자, 다음 글자) 위치들에 1씩 누적된다.
        np.add.at(counts, (ids[:-1], ids[1:]), 1)

        # 확률로 변환 (Laplace smoothing 적용하여 0 확률 방지)
        probs = (counts + 0.1) / (counts + 0.1).sum(axis=1, keepdims=True)

        # 정답 글자들의 확률 로그 평균 (넘파이 advanced indexing)
        eval_probs = probs[ids[:-1], ids[1:]]
        loss = -np.log(eval_probs).mean()
        ppl = np.exp(loss)
        print(f"Loss: {loss:.4f}, Perplexity: {ppl:.2f}")

    elif args.bigram_nn:
        if not args.model_path or not Path(args.model_path).exists():
            print("올바른 모델 경로를 지정해주세요.")
            return
        
        ckpt = np.load(args.model_path, allow_pickle=True)
        W = ckpt["W"]
        tokenizer = CharTokenizer(tuple(ckpt["vocab"]))
        
        ids = np.array(tokenizer.encode(eval_text))
        logits = W[ids[:-1]]
        loss, ppl = calculate_metrics(logits, ids[1:])
        print(f"--- Neural Bigram Evaluation (Model: {args.model_path}) ---")
        print(f"Loss: {loss:.4f}, Perplexity: {ppl:.2f}")

    elif args.mlp_lm:
        if not args.model_path or not Path(args.model_path).exists():
            print("올바른 모델 경로를 지정해주세요.")
            return
        
        ckpt = np.load(args.model_path, allow_pickle=True)
        tokenizer = CharTokenizer(tuple(ckpt["vocab"]))
        # 다중 대입(튜플 언패킹): 오른쪽이 5개 짜리 튜플로 묶여 5개 변수에 한 번에 할당된다.
        # 자바에는 없는 문법, C# 의 ValueTuple deconstruction 과 비슷.
        E, W1, b1, W2, b2 = ckpt["E"], ckpt["W1"], ckpt["b1"], ckpt["W2"], ckpt["b2"]
        C = int(ckpt["context_len"])
        
        ids = np.array(tokenizer.encode(eval_text))
        # Dataset 생성
        n = len(ids) - C
        X = np.empty((n, C), dtype=np.int64)
        y = np.empty((n,), dtype=np.int64)
        for i in range(n):
            X[i] = ids[i : i + C]
            y[i] = ids[i + C]
        
        # Forward pass (평가용이므로 배치 처리 생략하고 전체 계산)
        emb = E[X]
        h = np.tanh(emb.reshape(len(X), -1) @ W1 + b1)
        logits = h @ W2 + b2
        
        loss, ppl = calculate_metrics(logits, y)
        print(f"--- MLP LM Evaluation (Model: {args.model_path}) ---")
        print(f"Loss: {loss:.4f}, Perplexity: {ppl:.2f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
