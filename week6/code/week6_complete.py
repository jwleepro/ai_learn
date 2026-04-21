"""Week 6: 언어모델 평가 (초보용 단순 버전)."""

import argparse
from pathlib import Path

import numpy as np


def build_tokenizer(text: str):
    vocab = sorted(set(text))
    char_to_id = {}
    for i, ch in enumerate(vocab):
        char_to_id[ch] = i
    return vocab, char_to_id


def encode_text_safe(text: str, char_to_id: dict[str, int]):
    ids = []
    for ch in text:
        if ch not in char_to_id:
            return None
        ids.append(char_to_id[ch])
    return ids


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def loss_and_ppl(logits: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    probs = softmax(logits)
    target_probs = probs[np.arange(len(targets)), targets]
    loss = -np.log(target_probs + 1e-10).mean()
    return float(loss), float(np.exp(loss))


def evaluate_counts(eval_text: str):
    vocab, char_to_id = build_tokenizer(eval_text)
    ids = np.array([char_to_id[ch] for ch in eval_text], dtype=np.int64)
    v = len(vocab)
    counts = np.zeros((v, v), dtype=np.float64)
    for i in range(len(ids) - 1):
        counts[ids[i], ids[i + 1]] += 1.0
    probs = (counts + 0.1) / (counts + 0.1).sum(axis=1, keepdims=True)
    seq_probs = probs[ids[:-1], ids[1:]]
    loss = -np.log(seq_probs + 1e-10).mean()
    print(f"Loss: {loss:.4f}, Perplexity: {np.exp(loss):.2f}")


def evaluate_bigram_nn(eval_text: str, model_path: str):
    ckpt = np.load(model_path, allow_pickle=True)
    w = ckpt["W"]
    vocab = list(ckpt["vocab"])
    char_to_id = {ch: i for i, ch in enumerate(vocab)}
    ids = encode_text_safe(eval_text, char_to_id)
    if ids is None:
        print("평가 텍스트에 모델 vocab에 없는 문자가 있습니다.")
        return
    ids = np.array(ids, dtype=np.int64)
    logits = w[ids[:-1]]
    loss, ppl = loss_and_ppl(logits, ids[1:])
    print(f"Loss: {loss:.4f}, Perplexity: {ppl:.2f}")


def evaluate_mlp(eval_text: str, model_path: str):
    ckpt = np.load(model_path, allow_pickle=True)
    vocab = list(ckpt["vocab"])
    char_to_id = {ch: i for i, ch in enumerate(vocab)}
    ids = encode_text_safe(eval_text, char_to_id)
    if ids is None:
        print("평가 텍스트에 모델 vocab에 없는 문자가 있습니다.")
        return
    ids = np.array(ids, dtype=np.int64)

    e = ckpt["E"]
    w1 = ckpt["W1"]
    b1 = ckpt["b1"]
    w2 = ckpt["W2"]
    b2 = ckpt["b2"]
    c = int(ckpt["context_len"])

    n = len(ids) - c
    x = np.empty((n, c), dtype=np.int64)
    y = np.empty(n, dtype=np.int64)
    for i in range(n):
        x[i] = ids[i : i + c]
        y[i] = ids[i + c]

    emb = e[x]
    h = np.tanh(emb.reshape(n, -1) @ w1 + b1)
    logits = h @ w2 + b2
    loss, ppl = loss_and_ppl(logits, y)
    print(f"Loss: {loss:.4f}, Perplexity: {ppl:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", action="store_true")
    parser.add_argument("--bigram_nn", action="store_true")
    parser.add_argument("--mlp_lm", action="store_true")
    parser.add_argument("--data", default="week6/data/tiny_corpus_ko.txt")
    parser.add_argument("--model_path")
    args = parser.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"파일이 없습니다: {path}")
        return
    eval_text = path.read_text(encoding="utf-8")

    if args.counts:
        evaluate_counts(eval_text)
        return
    if args.bigram_nn:
        if not args.model_path or not Path(args.model_path).exists():
            print("모델 경로를 확인해주세요.")
            return
        evaluate_bigram_nn(eval_text, args.model_path)
        return
    if args.mlp_lm:
        if not args.model_path or not Path(args.model_path).exists():
            print("모델 경로를 확인해주세요.")
            return
        evaluate_mlp(eval_text, args.model_path)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
