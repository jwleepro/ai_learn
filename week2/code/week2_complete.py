"""Week 2 Complete: Neural Bigram Language Model.

이 파일은 신경망 기반 빅램 언어모델의 모든 과정(토크나이저, 데이터셋 생성, 모델 구현, 학습, 생성)을
하나의 파일에서 순서대로 읽고 실행할 수 있도록 통합한 교육용 코드입니다.

주요 내용:
1. CharTokenizer: 글자 단위 토크나이저
2. Neural Bigram Model: 학습 가능한 가중치 행렬 W (V, V)를 이용한 모델
3. Training: SGD(확률적 경사 하강법)를 이용한 학습 (Numpy 구현)
4. Generation: 학습된 모델을 이용한 텍스트 생성 (Temperature 조절)

실행 방법:
- 학습: python week2/code/week2_complete.py --train
- 생성: python week2/code/week2_complete.py --generate
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
        if len(self.vocab) == 0: raise ValueError("vocab empty")
        self.char_to_id = {ch: i for i, ch in enumerate(self.vocab)}

    @property
    def vocab_size(self) -> int: return len(self.vocab)

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


# ============================================================================
# 2. Neural Bigram Model Implementation
# ============================================================================

def init_W(vocab_size: int, rng: np.random.Generator) -> np.ndarray:
    """가중치 행렬 W를 작은 난수로 초기화합니다."""
    scale = 0.01
    return rng.normal(0.0, scale, size=(vocab_size, vocab_size))


def train_step(W: np.ndarray, prev_ids: np.ndarray, next_ids: np.ndarray, lr: float) -> float:
    """한 번의 학습 스텝(순전파, 손실 계산, 역전파, 가중치 업데이트)을 수행합니다."""
    # 1. Forward Pass
    logits = W[prev_ids] # (B, V)
    probs = softmax(logits, axis=1)
    
    # 2. Loss (Cross Entropy)
    loss = -np.log(probs[np.arange(len(next_ids)), next_ids] + 1e-10).mean()
    
    # 3. Backward Pass (Gradient)
    dlogits = probs.copy()
    dlogits[np.arange(len(next_ids)), next_ids] -= 1.0
    dlogits /= len(next_ids)
    
    # 4. Update W
    # dL/dW[i, :] = dL/dlogits를 prev_id=i인 샘플들에 대해 누적
    grad_W = np.zeros_like(W)
    np.add.at(grad_W, prev_ids, dlogits)
    W -= lr * grad_W
    
    return float(loss)


# ============================================================================
# 3. Main Execution Flow
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="학습 실행")
    parser.add_argument("--generate", action="store_true", help="생성 실행")
    parser.add_argument("--data", default="week2/data/tiny_corpus_ko.txt", help="데이터 경로")
    parser.add_argument("--model_path", default="week2/code/bigram_nn_model.npz", help="모델 저장/불러오기 경로")
    args = parser.parse_args()

    # 데이터 로드
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"데이터 파일이 없습니다: {data_path}")
        return
    text = data_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    ids = np.array(tokenizer.encode(text))
    
    # (이전 글자, 다음 글자) 쌍 생성
    prev_ids = ids[:-1]
    next_ids = ids[1:]

    # 하이퍼파라미터
    V = tokenizer.vocab_size
    LR = 2.0
    EPOCHS = 40
    BATCH_SIZE = 2048

    if args.train:
        print(f"--- 신경망 빅램 학습 시작 (Vocab: {V}) ---")
        rng = np.random.default_rng(42)
        W = init_W(V, rng)

        for epoch in range(EPOCHS):
            # 데이터를 섞음
            perm = rng.permutation(len(prev_ids))
            p_shuf = prev_ids[perm]
            n_shuf = next_ids[perm]
            
            epoch_loss = 0.0
            steps = 0
            for start in range(0, len(p_shuf), BATCH_SIZE):
                end = min(len(p_shuf), start + BATCH_SIZE)
                loss = train_step(W, p_shuf[start:end], n_shuf[start:end], LR)
                epoch_loss += loss
                steps += 1
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/steps:.4f}")

        # 모델 저장
        np.savez(args.model_path, W=W, vocab=tokenizer.vocab)
        print(f"모델 저장 완료: {args.model_path}")

    elif args.generate:
        if not Path(args.model_path).exists():
            print(f"모델 파일이 없습니다. 먼저 --train을 실행하세요: {args.model_path}")
            return
        
        # 모델 로드
        ckpt = np.load(args.model_path, allow_pickle=True)
        W = ckpt["W"]
        vocab = tuple(ckpt["vocab"])
        tokenizer = CharTokenizer(vocab)

        print("--- 텍스트 생성 시작 ---")
        rng = np.random.default_rng()
        current_id = ids[0] # 데이터의 첫 글자로 시작
        generated_ids = [current_id]
        
        for _ in range(100):
            logits = W[current_id]
            probs = softmax(logits)
            # 온도는 1.0으로 고정하거나 옵션으로 뺄 수 있음
            next_id = int(rng.choice(len(probs), p=probs))
            generated_ids.append(next_id)
            current_id = next_id
        
        print(f"Generated: {tokenizer.decode(generated_ids)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
