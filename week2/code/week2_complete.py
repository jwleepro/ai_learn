"""Week 2 Complete: Neural Bigram Language Model.

신경망 빅램 언어모델: week1 의 빅램과 같은 입력/출력이지만,
"빈도수 표"를 직접 만드는 대신 "학습 가능한 가중치 행렬 W (V x V)" 를 둔다.

학습:
- 손실(cross entropy)이 낮아지는 쪽으로 W 를 조금씩 조정한다.
- 결국 W[i] 의 softmax 가 "글자 i 다음에 올 글자들의 확률" 이 된다.
- 빅램 카운트 방식과 수학적으로 같은 답에 수렴하지만,
  신경망의 학습 절차(forward / loss / backward / update)를 그대로 연습할 수 있다.

실행 방법:
- 학습: python week2/code/week2_complete.py --train
- 생성: python week2/code/week2_complete.py --generate
"""

import argparse
from pathlib import Path
import numpy as np


# ============================================================================
# 1. CharTokenizer + 공통 함수
# ============================================================================

class CharTokenizer:
    """글자 단위 토크나이저 (week1 과 동일, 설명은 week1 파일 참조)."""

    def __init__(self, vocab):
        if len(vocab) == 0:
            raise ValueError("vocab empty")
        self.vocab = vocab

        self.char_to_id = {}
        for i in range(len(vocab)):
            self.char_to_id[vocab[i]] = i

    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        ids = []
        for ch in text:
            ids.append(self.char_to_id[ch])
        return ids

    def decode(self, ids):
        chars = []
        for token_id in ids:
            chars.append(self.vocab[token_id])
        return "".join(chars)


def build_tokenizer_from_text(text):
    unique_chars = sorted(set(text))
    return CharTokenizer(unique_chars)


def softmax(logits, axis=-1):
    """logits(점수) -> 확률.

    수식: softmax(x_i) = exp(x_i) / sum(exp(x_j))
    큰 값을 빼는 max-shift 는 exp 가 너무 커지는 걸 막는 트릭.
    """
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


# ============================================================================
# 2. Neural Bigram Model
# ============================================================================

def init_W(vocab_size, rng):
    """가중치 행렬 W (V x V) 를 작은 난수로 초기화."""
    return rng.normal(loc=0.0, scale=0.01, size=(vocab_size, vocab_size))


def train_step(W, prev_ids, next_ids, lr):
    """학습 한 스텝: forward -> loss -> backward -> 가중치 업데이트.

    배치 크기 B 에 대해
    - prev_ids: 입력 글자 ID 들 (B,)
    - next_ids: 정답 다음 글자 ID 들 (B,)
    - W: (V, V) 가중치 행렬
    """
    batch_size = len(prev_ids)

    # 1) Forward Pass
    # W[prev_ids]: 넘파이 fancy indexing.
    # prev_ids 가 (B,) 모양이면 W[prev_ids] 는 (B, V) 모양이 된다.
    # 자바로 풀면: for i in 0..B-1: logits[i] = W[prev_ids[i]]
    logits = W[prev_ids]
    probs = softmax(logits, axis=1)  # 행마다 softmax -> (B, V)

    # 2) Loss (Cross Entropy)
    # 정답 위치의 확률만 모아서 -log 평균을 낸다.
    correct_probs = np.zeros(batch_size)
    for i in range(batch_size):
        correct_probs[i] = probs[i, next_ids[i]]
    loss = -np.log(correct_probs + 1e-10).mean()

    # 3) Backward Pass (그래디언트 계산)
    # softmax + cross entropy 의 잘 알려진 결과: dL/dlogits = (probs - one_hot) / B
    dlogits = probs.copy()
    for i in range(batch_size):
        dlogits[i, next_ids[i]] -= 1.0
    dlogits = dlogits / batch_size

    # 4) W 의 그래디언트 계산
    # logits[i] = W[prev_ids[i]] 였으므로 dW[prev_ids[i]] += dlogits[i] 가 된다.
    grad_W = np.zeros_like(W)
    for i in range(batch_size):
        grad_W[prev_ids[i]] += dlogits[i]

    # 5) 가중치 업데이트 (경사하강법)
    # W = W - learning_rate * gradient
    W -= lr * grad_W

    return float(loss)


# ============================================================================
# 3. Main Execution Flow
# ============================================================================

def main():
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
    tokenizer = build_tokenizer_from_text(text)
    ids = np.array(tokenizer.encode(text))

    # (이전 글자, 다음 글자) 쌍 만들기
    # ids[:-1]: 마지막 한 개 제외 / ids[1:]: 첫 한 개 제외
    prev_ids = ids[:-1]
    next_ids = ids[1:]

    # 하이퍼파라미터
    vocab_size = tokenizer.vocab_size()
    learning_rate = 2.0
    epochs = 40
    batch_size = 2048

    if args.train:
        print(f"--- 신경망 빅램 학습 시작 (Vocab: {vocab_size}) ---")
        rng = np.random.default_rng(42)
        W = init_W(vocab_size, rng)

        for epoch in range(epochs):
            # 매 에폭마다 데이터 순서를 섞음
            shuffle_idx = rng.permutation(len(prev_ids))
            prev_shuffled = prev_ids[shuffle_idx]
            next_shuffled = next_ids[shuffle_idx]

            epoch_loss = 0.0
            num_steps = 0

            # 배치 단위로 잘라서 학습
            start = 0
            while start < len(prev_shuffled):
                end = min(start + batch_size, len(prev_shuffled))
                batch_prev = prev_shuffled[start:end]
                batch_next = next_shuffled[start:end]

                loss = train_step(W, batch_prev, batch_next, learning_rate)
                epoch_loss += loss
                num_steps += 1

                start = end

            if (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / num_steps
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # 모델 저장
        np.savez(args.model_path, W=W, vocab=tokenizer.vocab)
        print(f"모델 저장 완료: {args.model_path}")

    elif args.generate:
        if not Path(args.model_path).exists():
            print(f"모델 파일이 없습니다. 먼저 --train 을 실행하세요: {args.model_path}")
            return

        # 모델 로드
        ckpt = np.load(args.model_path, allow_pickle=True)
        W = ckpt["W"]
        # ckpt["vocab"] 은 numpy 배열로 저장돼 있어서 다시 list 로 변환
        vocab = list(ckpt["vocab"])
        tokenizer = CharTokenizer(vocab)

        print("--- 텍스트 생성 시작 ---")
        rng = np.random.default_rng()

        # 시작 글자: 데이터의 첫 글자
        current_id = int(ids[0])
        generated_ids = [current_id]

        for _ in range(100):
            logits = W[current_id]
            probs = softmax(logits)
            next_id = int(rng.choice(len(probs), p=probs))
            generated_ids.append(next_id)
            current_id = next_id

        print(f"Generated: {tokenizer.decode(generated_ids)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
