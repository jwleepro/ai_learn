"""체크포인트 저장/로드(.npz).

Core 트랙의 학습 결과(토크나이저 + 파라미터)를 파일로 저장하고,
생성/평가에서 다시 불러올 수 있게 합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from tokenizer_char import CharTokenizer
from mlp_lm import MLPLMParams


ModelType = Literal["bigram_nn", "mlp_lm"]


@dataclass(frozen=True)
class BigramNNCheckpoint:
    tokenizer: CharTokenizer
    W: np.ndarray
    default_start_id: int = 0


@dataclass(frozen=True)
class MLPLMCheckpoint:
    tokenizer: CharTokenizer
    context_len: int
    embed_dim: int
    hidden_dim: int
    params: MLPLMParams
    default_start_ids: np.ndarray


def save_bigram_nn(path: str | Path, ckpt: BigramNNCheckpoint) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vocab_arr = np.array(list(ckpt.tokenizer.vocab))
    np.savez(
        path,
        type="bigram_nn",
        vocab=vocab_arr,
        W=ckpt.W,
        default_start_id=np.int64(int(ckpt.default_start_id)),
    )


def load_bigram_nn(path: str | Path) -> BigramNNCheckpoint:
    path = Path(path)
    with np.load(path) as z:
        model_type = str(z["type"])
        if model_type != "bigram_nn":
            raise ValueError(f"Unsupported model type: {model_type!r}")
        vocab = tuple(z["vocab"].tolist())
        W = z["W"]
        default_start_id = int(z["default_start_id"]) if "default_start_id" in z else 0
    tokenizer = CharTokenizer(vocab)
    if W.shape != (tokenizer.vocab_size, tokenizer.vocab_size):
        raise ValueError("Checkpoint W has incompatible shape for vocab")
    if not (0 <= default_start_id < tokenizer.vocab_size):
        default_start_id = 0
    return BigramNNCheckpoint(tokenizer=tokenizer, W=W, default_start_id=default_start_id)


def save_mlp_lm(path: str | Path, ckpt: MLPLMCheckpoint) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vocab_arr = np.array(list(ckpt.tokenizer.vocab))

    default_start_ids = np.array(ckpt.default_start_ids, dtype=np.int64)
    if default_start_ids.ndim != 1 or len(default_start_ids) != int(ckpt.context_len):
        raise ValueError("default_start_ids must be 1D with length=context_len")

    np.savez(
        path,
        type="mlp_lm",
        vocab=vocab_arr,
        context_len=np.int64(int(ckpt.context_len)),
        embed_dim=np.int64(int(ckpt.embed_dim)),
        hidden_dim=np.int64(int(ckpt.hidden_dim)),
        E=ckpt.params.E,
        W1=ckpt.params.W1,
        b1=ckpt.params.b1,
        W2=ckpt.params.W2,
        b2=ckpt.params.b2,
        default_start_ids=default_start_ids,
    )


def load_mlp_lm(path: str | Path) -> MLPLMCheckpoint:
    path = Path(path)
    with np.load(path) as z:
        model_type = str(z["type"])
        if model_type != "mlp_lm":
            raise ValueError(f"Unsupported model type: {model_type!r}")
        vocab = tuple(z["vocab"].tolist())
        context_len = int(z["context_len"])
        embed_dim = int(z["embed_dim"])
        hidden_dim = int(z["hidden_dim"])
        E = z["E"]
        W1 = z["W1"]
        b1 = z["b1"]
        W2 = z["W2"]
        b2 = z["b2"]
        default_start_ids = z["default_start_ids"].astype(np.int64)

    tokenizer = CharTokenizer(vocab)
    vocab_size = tokenizer.vocab_size
    if E.shape != (vocab_size, embed_dim):
        raise ValueError("E has incompatible shape for vocab/embed_dim")
    if W1.shape != (context_len * embed_dim, hidden_dim):
        raise ValueError("W1 has incompatible shape for context/embed/hidden")
    if b1.shape != (hidden_dim,):
        raise ValueError("b1 has incompatible shape for hidden_dim")
    if W2.shape != (hidden_dim, vocab_size):
        raise ValueError("W2 has incompatible shape for hidden/vocab")
    if b2.shape != (vocab_size,):
        raise ValueError("b2 has incompatible shape for vocab")
    if default_start_ids.ndim != 1 or len(default_start_ids) != context_len:
        raise ValueError("default_start_ids has incompatible shape for context_len")

    params = MLPLMParams(E=E, W1=W1, b1=b1, W2=W2, b2=b2)
    return MLPLMCheckpoint(
        tokenizer=tokenizer,
        context_len=context_len,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        params=params,
        default_start_ids=default_start_ids,
    )
