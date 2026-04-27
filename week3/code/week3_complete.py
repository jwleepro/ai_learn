"""Week 3 Complete: MLP 언어모델.

이 파일은 MLP 기반 언어모델의 모든 과정(토크나이저, 데이터셋 생성, 모델 구현, 학습, 생성)을
하나의 파일에서 순서대로 읽고 실행할 수 있도록 통합한 교육용 코드입니다.

주요 내용:
1. CharTokenizer: 글자 단위 토크나이저
2. Dataset: 슬라이딩 윈도우 기반 컨텍스트 데이터셋 생성
3. MLP Model: 임베딩, 은닉층, 출력층으로 구성된 신경망 (NumPy 구현)
4. Training: SGD(확률적 경사 하강법)를 이용한 학습
5. Generation: 학습된 모델을 이용한 텍스트 생성 (Temperature, Top-k/p 샘플링)

실행 방법:
- 학습: python week3/code/week3_complete.py --train
- 생성: python week3/code/week3_complete.py --generate
"""

from __future__ import annotations  # 파이썬 전용: 타입 힌트를 문자열로 평가(전방 참조)

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


# ============================================================================
# 1. Utility Functions & Classes (Formerly external modules)
# ============================================================================

# CharTokenizer: 파이썬 전용 문법 정리
# - @dataclass : 필드 선언만으로 __init__ 자동 생성 (자바 record 와 비슷)
# - tuple[str, ...] : 가변 길이 동질 튜플 타입 표기
# - @property / @classmethod : 속성 접근 메서드 / 클래스 팩토리 메서드
# - {ch: i for i, ch in enumerate(...)} : 딕셔너리 컴프리헨션 + 튜플 언패킹
# - [x for x in y] : 리스트 컴프리헨션
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


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def make_context_dataset(token_ids: np.ndarray, context_len: int) -> tuple[np.ndarray, np.ndarray]:
    """슬라이딩 윈도우로 (입력 컨텍스트, 다음 글자) 쌍을 만든다.

    예) token_ids=[1,2,3,4,5], context_len=3 이면
        X = [[1,2,3], [2,3,4]]
        y = [4, 5]
    """
    n = len(token_ids) - context_len
    X = np.empty((n, context_len), dtype=np.int64)
    y = np.empty((n,), dtype=np.int64)
    for i in range(n):
        X[i] = token_ids[i : i + context_len]
        y[i] = token_ids[i + context_len]
    return X, y


# `int | None` 은 파이썬 3.10+ 의 union 타입 표기로 "int 또는 None" 을 뜻한다.
# 자바의 Optional<Integer>, C# 의 int? 와 비슷한 의미.
@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None


def sample_from_probs(probs: np.ndarray, rng: np.random.Generator, cfg: SamplingConfig) -> int:
    p = probs.copy()
    if cfg.temperature != 1.0:
        p = np.power(p, 1.0 / cfg.temperature)
        p /= p.sum()
    
    if cfg.top_k is not None:
        indices_to_remove = p < np.partition(p, -cfg.top_k)[-cfg.top_k]
        p[indices_to_remove] = 0
        p /= p.sum()
        
    if cfg.top_p is not None:
        # `[::-1]` 은 슬라이싱 step=-1 → "역순". 자바/C# 에는 없는 표기.
        sorted_indices = np.argsort(p)[::-1]
        # 넘파이 fancy indexing: 인덱스 배열로 한꺼번에 골라낸다.
        sorted_probs = p[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        indices_to_remove = cumulative_probs > cfg.top_p
        indices_to_remove[1:] = indices_to_remove[:-1].copy()
        indices_to_remove[0] = False
        p[sorted_indices[indices_to_remove]] = 0
        p /= p.sum()
        
    return int(rng.choice(len(p), p=p))


# ============================================================================
# 2. MLP Model Implementation
# ============================================================================

@dataclass
class MLPLMParams:
    E: np.ndarray   # Embedding table
    W1: np.ndarray  # Hidden weights
    b1: np.ndarray  # Hidden bias
    W2: np.ndarray  # Output weights
    b2: np.ndarray  # Output bias


def init_params(vocab_size: int, context_len: int, embed_dim: int, hidden_dim: int, rng: np.random.Generator) -> MLPLMParams:
    scale = 0.02
    return MLPLMParams(
        E = rng.normal(0, scale, (vocab_size, embed_dim)),
        W1 = rng.normal(0, scale, (context_len * embed_dim, hidden_dim)),
        b1 = np.zeros(hidden_dim),
        W2 = rng.normal(0, scale, (hidden_dim, vocab_size)),
        b2 = np.zeros(vocab_size)
    )


def forward(params: MLPLMParams, X: np.ndarray) -> tuple[np.ndarray, dict]:
    # 함수가 두 값을 한 번에 반환 → 호출 측에서 `a, b = forward(...)` 형태의 튜플 언패킹으로 받는다.
    # 자바에는 없고 C# 의 ValueTuple deconstruction 과 비슷.
    emb = params.E[X]  # (B, C, D)  ← 넘파이 fancy indexing
    h_in = emb.reshape(len(X), -1)  # (B, C*D); reshape 의 -1 은 "남은 차원 자동 계산"
    h_pre = h_in @ params.W1 + params.b1  # `@` 는 행렬곱 전용 연산자(파이썬 3.5+)
    h = np.tanh(h_pre)
    logits = h @ params.W2 + params.b2
    return logits, {"X": X, "h_in": h_in, "h_pre": h_pre, "h": h, "emb": emb}


def train_step(params: MLPLMParams, X: np.ndarray, y: np.ndarray, lr: float) -> float:
    logits, cache = forward(params, X)
    probs = softmax(logits, axis=1)
    
    # Loss (Cross Entropy)
    loss = -np.log(probs[np.arange(len(y)), y] + 1e-10).mean()
    
    # Backprop
    dlogits = probs.copy()
    dlogits[np.arange(len(y)), y] -= 1.0
    dlogits /= len(y)
    
    dW2 = cache["h"].T @ dlogits
    db2 = dlogits.sum(axis=0)
    
    dh = dlogits @ params.W2.T
    dh_pre = dh * (1.0 - cache["h"]**2) # d/dx tanh = 1 - tanh^2
    
    dW1 = cache["h_in"].T @ dh_pre
    db1 = dh_pre.sum(axis=0)
    
    dh_in = dh_pre @ params.W1.T
    dEmb = dh_in.reshape(cache["emb"].shape)

    # 임베딩 그래디언트는 토큰 ID마다 등장 횟수만큼 누적되어야 한다.
    # X에 같은 ID가 여러 번 나오면 dEmb의 해당 위치들이 모두 더해진다.
    # np.add.at은 일반 += 와 달리 인덱스 중복도 정확히 누적해 준다.
    dE = np.zeros_like(params.E)
    np.add.at(dE, cache["X"], dEmb)

    # 모든 파라미터를 학습률 lr 만큼 그래디언트 반대 방향으로 이동시킨다.
    params.W2 -= lr * dW2
    params.b2 -= lr * db2
    params.W1 -= lr * dW1
    params.b1 -= lr * db1
    params.E -= lr * dE
    
    return float(loss)


# ============================================================================
# 3. Main Execution Flow
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="학습 실행")
    parser.add_argument("--generate", action="store_true", help="생성 실행")
    parser.add_argument("--data", default="week3/data/tiny_corpus_ko.txt", help="데이터 경로")
    parser.add_argument("--model_path", default="week3/code/mlp_model.npz", help="모델 저장/불러오기 경로")
    args = parser.parse_args()

    # 데이터 로드
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"데이터 파일이 없습니다: {data_path}")
        return
    text = data_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    ids = np.array(tokenizer.encode(text))

    # 하이퍼파라미터
    C = 8      # Context length
    D = 32     # Embedding dim
    H = 128    # Hidden dim
    LR = 0.1
    EPOCHS = 10 # 교육용이므로 짧게 설정

    if args.train:
        print(f"--- 학습 시작 (Vocab: {tokenizer.vocab_size}, Context: {C}) ---")
        X, y = make_context_dataset(ids, C)
        rng = np.random.default_rng(42)
        params = init_params(tokenizer.vocab_size, C, D, H, rng)

        for epoch in range(EPOCHS):
            # 단순화를 위해 전체 데이터를 한 번에 처리 (또는 배치를 나눌 수 있음)
            loss = train_step(params, X, y, LR)
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")

        # 모델 저장
        np.savez(args.model_path, E=params.E, W1=params.W1, b1=params.b1, W2=params.W2, b2=params.b2, vocab=tokenizer.vocab, context_len=C)
        print(f"모델 저장 완료: {args.model_path}")

    elif args.generate:
        if not Path(args.model_path).exists():
            print(f"모델 파일이 없습니다. 먼저 --train을 실행하세요: {args.model_path}")
            return

        # 모델 로드
        ckpt = np.load(args.model_path, allow_pickle=True)
        # tuple(어떤_시퀀스) 처럼 "타입 이름을 함수로 호출"하는 변환 패턴은 파이썬 관용구.
        # 자바/C# 의 명시적 캐스트나 new ArrayList<>(seq) 와 비슷한 역할.
        vocab = tuple(ckpt["vocab"])
        tokenizer = CharTokenizer(vocab)
        params = MLPLMParams(ckpt["E"], ckpt["W1"], ckpt["b1"], ckpt["W2"], ckpt["b2"])
        context_len = int(ckpt["context_len"])

        print("--- 텍스트 생성 시작 ---")
        rng = np.random.default_rng()
        # 시작 컨텍스트: 데이터의 앞부분 사용
        context = ids[:context_len].tolist()
        print(f"Seed context: '{tokenizer.decode(context)}'")
        
        generated = []
        for _ in range(100):
            x = np.array([context[-context_len:]])
            logits, _ = forward(params, x)
            probs = softmax(logits[0])
            next_id = sample_from_probs(probs, rng, SamplingConfig(temperature=0.8))
            generated.append(next_id)
            context.append(next_id)
        
        print(f"Generated: {tokenizer.decode(generated)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
