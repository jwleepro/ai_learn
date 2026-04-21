"""Week 2: Neural Bigram (초보용 단순 버전)."""

import argparse
from pathlib import Path

import numpy as np


def build_tokenizer(text: str):
    vocab = sorted(set(text))
    char_to_id = {}
    for i, ch in enumerate(vocab):
        char_to_id[ch] = i
    return vocab, char_to_id


def encode_text(text: str, char_to_id: dict[str, int]) -> list[int]:
    return [char_to_id[ch] for ch in text]


def decode_ids(ids: list[int], vocab: list[str]) -> str:
    return "".join(vocab[idx] for idx in ids)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_step(W: np.ndarray, prev_ids: np.ndarray, next_ids: np.ndarray, lr: float) -> float:
    logits = W[prev_ids]
    probs = softmax(logits)
    loss = -np.log(probs[np.arange(len(next_ids)), next_ids] + 1e-10).mean()

    grad_logits = probs.copy()
    grad_logits[np.arange(len(next_ids)), next_ids] -= 1.0
    grad_logits /= len(next_ids)

    grad_W = np.zeros_like(W)
    for i in range(len(prev_ids)):
        grad_W[prev_ids[i]] += grad_logits[i]
    W -= lr * grad_W
    return float(loss)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--data", default="week2/data/tiny_corpus_ko.txt")
    parser.add_argument("--model_path", default="week2/code/bigram_nn_model.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2.0)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"파일이 없습니다: {data_path}")
        return

    text = data_path.read_text(encoding="utf-8")
    vocab, char_to_id = build_tokenizer(text)
    ids = np.array(encode_text(text, char_to_id), dtype=np.int64)
    prev_ids = ids[:-1]
    next_ids = ids[1:]

    if args.train:
        rng = np.random.default_rng(42)
        W = rng.normal(0.0, 0.01, size=(len(vocab), len(vocab)))
        for epoch in range(args.epochs):
            loss = train_step(W, prev_ids, next_ids, args.lr)
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss:.4f}")
        np.savez(args.model_path, W=W, vocab=np.array(vocab))
        print(f"저장 완료: {args.model_path}")
        return

    if args.generate:
        if not Path(args.model_path).exists():
            print(f"모델 파일이 없습니다: {args.model_path}")
            return
        ckpt = np.load(args.model_path, allow_pickle=True)
        W = ckpt["W"]
        vocab = list(ckpt["vocab"])
        rng = np.random.default_rng()
        current = int(ids[0])
        out = [current]
        for _ in range(100):
            row = W[current].reshape(1, -1)
            probs = softmax(row)[0]
            nxt = int(rng.choice(len(probs), p=probs))
            out.append(nxt)
            current = nxt
        print(decode_ids(out, vocab))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
