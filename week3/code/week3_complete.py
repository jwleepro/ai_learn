"""Week 3: MLP 언어모델 (초보용 단순 버전)."""

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


def make_dataset(ids: np.ndarray, context_len: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(ids) - context_len
    X = np.empty((n, context_len), dtype=np.int64)
    y = np.empty(n, dtype=np.int64)
    for i in range(n):
        X[i] = ids[i : i + context_len]
        y[i] = ids[i + context_len]
    return X, y


def init_params(vocab_size: int, context_len: int, embed_dim: int, hidden_dim: int):
    rng = np.random.default_rng(42)
    scale = 0.02
    params = {
        "E": rng.normal(0, scale, (vocab_size, embed_dim)),
        "W1": rng.normal(0, scale, (context_len * embed_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim),
        "W2": rng.normal(0, scale, (hidden_dim, vocab_size)),
        "b2": np.zeros(vocab_size),
    }
    return params


def forward(params: dict, X: np.ndarray):
    emb = params["E"][X]
    h_in = emb.reshape(len(X), -1)
    h = np.tanh(h_in @ params["W1"] + params["b1"])
    logits = h @ params["W2"] + params["b2"]
    cache = {"X": X, "emb": emb, "h_in": h_in, "h": h}
    return logits, cache


def train_step(params: dict, X: np.ndarray, y: np.ndarray, lr: float) -> float:
    logits, cache = forward(params, X)
    probs = softmax(logits)
    loss = -np.log(probs[np.arange(len(y)), y] + 1e-10).mean()

    dlogits = probs.copy()
    dlogits[np.arange(len(y)), y] -= 1.0
    dlogits /= len(y)

    dW2 = cache["h"].T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = dlogits @ params["W2"].T
    dh_pre = dh * (1.0 - cache["h"] ** 2)
    dW1 = cache["h_in"].T @ dh_pre
    db1 = dh_pre.sum(axis=0)

    dflat = dh_pre @ params["W1"].T
    demb = dflat.reshape(cache["emb"].shape)
    dE = np.zeros_like(params["E"])
    for b in range(X.shape[0]):
        for c in range(X.shape[1]):
            token_id = cache["X"][b, c]
            dE[token_id] += demb[b, c]

    params["W2"] -= lr * dW2
    params["b2"] -= lr * db2
    params["W1"] -= lr * dW1
    params["b1"] -= lr * db1
    params["E"] -= lr * dE
    return float(loss)


def sample_one(prob: np.ndarray, rng: np.random.Generator, temperature: float) -> int:
    p = prob.copy()
    if temperature != 1.0:
        p = np.power(p, 1.0 / temperature)
        p = p / p.sum()
    return int(rng.choice(len(p), p=p))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--data", default="week3/data/tiny_corpus_ko.txt")
    parser.add_argument("--model_path", default="week3/code/mlp_model.npz")
    args = parser.parse_args()

    path = Path(args.data)
    if not path.exists():
        print(f"파일이 없습니다: {path}")
        return

    text = path.read_text(encoding="utf-8")
    vocab, char_to_id = build_tokenizer(text)
    ids = np.array(encode_text(text, char_to_id), dtype=np.int64)

    context_len = 8
    embed_dim = 24
    hidden_dim = 64
    lr = 0.1
    epochs = 10

    if args.train:
        X, y = make_dataset(ids, context_len)
        params = init_params(len(vocab), context_len, embed_dim, hidden_dim)
        for epoch in range(epochs):
            loss = train_step(params, X, y, lr)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        np.savez(
            args.model_path,
            E=params["E"],
            W1=params["W1"],
            b1=params["b1"],
            W2=params["W2"],
            b2=params["b2"],
            vocab=np.array(vocab),
            context_len=context_len,
        )
        print(f"저장 완료: {args.model_path}")
        return

    if args.generate:
        if not Path(args.model_path).exists():
            print(f"모델 파일이 없습니다: {args.model_path}")
            return
        ckpt = np.load(args.model_path, allow_pickle=True)
        params = {
            "E": ckpt["E"],
            "W1": ckpt["W1"],
            "b1": ckpt["b1"],
            "W2": ckpt["W2"],
            "b2": ckpt["b2"],
        }
        vocab = list(ckpt["vocab"])
        context_len = int(ckpt["context_len"])

        rng = np.random.default_rng()
        context = ids[:context_len].tolist()
        out = []
        for _ in range(100):
            x = np.array([context[-context_len:]], dtype=np.int64)
            logits, _ = forward(params, x)
            probs = softmax(logits)[0]
            nxt = sample_one(probs, rng, temperature=0.8)
            out.append(nxt)
            context.append(nxt)
        print(decode_ids(out, vocab))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
