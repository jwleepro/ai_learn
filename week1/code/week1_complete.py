"""Week 1: Bigram 언어모델 (초보용 단순 버전)."""

import argparse
from pathlib import Path

import numpy as np


def build_tokenizer(text: str):
    vocab = sorted(set(text))
    if not vocab:
        raise ValueError("빈 텍스트입니다.")
    char_to_id = {}
    for i, ch in enumerate(vocab):
        char_to_id[ch] = i
    return vocab, char_to_id


def encode_text(text: str, char_to_id: dict[str, int]) -> list[int]:
    ids = []
    for ch in text:
        ids.append(char_to_id[ch])
    return ids


def decode_ids(ids: list[int], vocab: list[str]) -> str:
    out = []
    for idx in ids:
        out.append(vocab[idx])
    return "".join(out)


def build_bigram_counts(token_ids: list[int], vocab_size: int) -> np.ndarray:
    counts = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    for i in range(len(token_ids) - 1):
        prev_id = token_ids[i]
        next_id = token_ids[i + 1]
        counts[prev_id, next_id] += 1
    return counts


def counts_to_probs(counts: np.ndarray, smoothing: float) -> np.ndarray:
    probs = counts.astype(np.float64) + smoothing
    for row in range(probs.shape[0]):
        row_sum = probs[row].sum()
        if row_sum == 0:
            probs[row] = 1.0 / probs.shape[1]
        else:
            probs[row] = probs[row] / row_sum
    return probs


def sample_next_id(prob_row: np.ndarray, rng: np.random.Generator, temperature: float) -> int:
    p = prob_row.copy()
    if temperature != 1.0:
        p = np.power(p, 1.0 / temperature)
        p = p / p.sum()
    return int(rng.choice(len(p), p=p))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--char", default=" ")
    parser.add_argument("--data", default="week1/data/tiny_corpus_ko.txt")
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--smooth", type=float, default=0.1)
    args = parser.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"파일이 없습니다: {path}")
        return

    text = path.read_text(encoding="utf-8")
    vocab, char_to_id = build_tokenizer(text)
    token_ids = encode_text(text, char_to_id)
    counts = build_bigram_counts(token_ids, len(vocab))
    probs = counts_to_probs(counts, smoothing=args.smooth)
    print(f"문자 수: {len(text)}, vocab 크기: {len(vocab)}")

    if args.generate:
        rng = np.random.default_rng(42)
        current_id = token_ids[0]
        generated = [current_id]
        for _ in range(args.length):
            next_id = sample_next_id(probs[current_id], rng, args.temp)
            generated.append(next_id)
            current_id = next_id
        print(decode_ids(generated, vocab))
        return

    if args.inspect:
        if args.char not in char_to_id:
            print(f"'{args.char}' 문자는 데이터에 없습니다.")
            return
        row = probs[char_to_id[args.char]]
        top_indices = np.argsort(row)[::-1][:10]
        print(f"'{args.char}' 뒤에 올 확률 상위 10개")
        for idx in top_indices:
            print(f"{vocab[idx]!r}: {row[idx]:.4f}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
