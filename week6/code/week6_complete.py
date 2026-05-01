"""Week 6 Complete: Language Model Evaluation.

학습된 언어모델이 얼마나 좋은지 측정하는 두 가지 지표:

1) Cross Entropy Loss = -log(정답 토큰의 예측 확률) 의 평균
2) Perplexity (PPL) = exp(loss)
   - "모델이 다음 글자를 고를 때 평균적으로 몇 개 후보 사이에서 헤매는지" 라고 보면 됨
   - 작을수록 좋다.

세 가지 모델을 같은 평가 데이터로 비교:
- Count-based Bigram (week1)
- Neural Bigram      (week2)
- MLP LM             (week3)

실행 방법:
- Count Bigram 평가: python week6/code/week6_complete.py --counts --data week6/data/tiny_corpus_ko.txt
- Neural Bigram 평가: python week6/code/week6_complete.py --bigram_nn --model_path week2/code/bigram_nn_model.npz --data week6/data/tiny_corpus_ko.txt
- MLP LM 평가: python week6/code/week6_complete.py --mlp_lm --model_path week3/code/mlp_model.npz --data week6/data/tiny_corpus_ko.txt
"""

import argparse
from pathlib import Path
import numpy as np


# ============================================================================
# 1. CharTokenizer + softmax
# ============================================================================

class CharTokenizer:
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
    return CharTokenizer(sorted(set(text)))


def softmax(logits, axis=-1):
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


# ============================================================================
# 2. 평가 지표
# ============================================================================

def calculate_loss_and_ppl(logits, targets):
    """logits (N, V) 와 정답 targets (N,) 으로 loss 와 perplexity 계산."""
    probs = softmax(logits, axis=-1)

    # 정답 위치의 확률만 모은다
    correct_probs = np.zeros(len(targets))
    for i in range(len(targets)):
        correct_probs[i] = probs[i, targets[i]]

    # cross entropy = -log(정답 확률) 의 평균
    loss = -np.log(correct_probs + 1e-10).mean()
    ppl = np.exp(loss)
    return float(loss), float(ppl)


# ============================================================================
# 3. Main Execution Flow
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", action="store_true", help="Count Bigram 평가")
    parser.add_argument("--bigram_nn", action="store_true", help="Neural Bigram 평가")
    parser.add_argument("--mlp_lm", action="store_true", help="MLP LM 평가")
    parser.add_argument("--data", default="week6/data/tiny_corpus_ko.txt", help="평가용 데이터 경로")
    parser.add_argument("--model_path", help="평가할 모델 파일 경로 (.npz)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"데이터 파일이 없습니다: {data_path}")
        return
    eval_text = data_path.read_text(encoding="utf-8")

    if args.counts:
        print("--- Count-based Bigram Evaluation ---")
        tokenizer = build_tokenizer_from_text(eval_text)
        ids = np.array(tokenizer.encode(eval_text))
        vocab_size = tokenizer.vocab_size()

        # 빈도수 행렬 만들기 (이전 글자, 다음 글자)
        counts = np.zeros((vocab_size, vocab_size))
        for i in range(len(ids) - 1):
            prev_id = ids[i]
            next_id = ids[i + 1]
            counts[prev_id, next_id] += 1

        # Laplace smoothing 적용 후 행 정규화
        smoothed = counts + 0.1
        probs = smoothed / smoothed.sum(axis=1, keepdims=True)

        # 평가: 각 (prev -> next) 쌍에 대해 정답 확률을 모은다
        n_pairs = len(ids) - 1
        eval_probs = np.empty(n_pairs)
        for i in range(n_pairs):
            eval_probs[i] = probs[ids[i], ids[i + 1]]

        loss = -np.log(eval_probs).mean()
        ppl = np.exp(loss)
        print(f"Loss: {loss:.4f}, Perplexity: {ppl:.2f}")

    elif args.bigram_nn:
        if not args.model_path or not Path(args.model_path).exists():
            print("올바른 모델 경로를 지정해주세요.")
            return

        ckpt = np.load(args.model_path, allow_pickle=True)
        W = ckpt["W"]
        vocab = list(ckpt["vocab"])
        tokenizer = CharTokenizer(vocab)

        ids = np.array(tokenizer.encode(eval_text))
        # 입력: 이전 글자들 (logits = W[prev_id])
        prev_ids = ids[:-1]
        next_ids = ids[1:]
        logits = W[prev_ids]  # (N, V)

        loss, ppl = calculate_loss_and_ppl(logits, next_ids)
        print(f"--- Neural Bigram Evaluation (Model: {args.model_path}) ---")
        print(f"Loss: {loss:.4f}, Perplexity: {ppl:.2f}")

    elif args.mlp_lm:
        if not args.model_path or not Path(args.model_path).exists():
            print("올바른 모델 경로를 지정해주세요.")
            return

        ckpt = np.load(args.model_path, allow_pickle=True)
        vocab = list(ckpt["vocab"])
        tokenizer = CharTokenizer(vocab)
        E = ckpt["E"]
        W1 = ckpt["W1"]
        b1 = ckpt["b1"]
        W2 = ckpt["W2"]
        b2 = ckpt["b2"]
        context_len = int(ckpt["context_len"])

        ids = np.array(tokenizer.encode(eval_text))

        # 슬라이딩 윈도우로 (입력 컨텍스트, 정답 다음 글자) 만들기
        n = len(ids) - context_len
        X = np.empty((n, context_len), dtype=np.int64)
        y = np.empty((n,), dtype=np.int64)
        for i in range(n):
            for k in range(context_len):
                X[i, k] = ids[i + k]
            y[i] = ids[i + context_len]

        # Forward (MLP) — 평가용이라 배치 분할 없이 한 번에 계산
        emb = E[X]                              # (N, C, D)
        h_in = emb.reshape(len(X), -1)          # (N, C*D)
        h = np.tanh(h_in @ W1 + b1)             # (N, H)
        logits = h @ W2 + b2                    # (N, V)

        loss, ppl = calculate_loss_and_ppl(logits, y)
        print(f"--- MLP LM Evaluation (Model: {args.model_path}) ---")
        print(f"Loss: {loss:.4f}, Perplexity: {ppl:.2f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
