"""Week 6 Complete: Language Model Evaluation.

이 통합 파일은 evaluate_lm.py의 모든 코드를 포함합니다.

언어모델 평가(loss, perplexity)를 지원합니다:
- counts_bigram: 카운트 기반 빅램
- bigram_nn: 신경망 빅램 체크포인트(.npz)
- mlp_lm: MLP LM 체크포인트(.npz)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tokenizer_char import CharTokenizer
from dataset_lm import make_context_dataset


# ============================================================================
# Section 1: Helper functions (inline implementations)
# ============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """소프트맥스 함수"""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """로그 소프트맥스"""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=axis, keepdims=True)
    return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))


# ============================================================================
# Section 2: Bigram counts (inline)
# ============================================================================

def build_bigram_counts(ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """빅램 (이전 토큰 -> 다음 토큰) 카운트 테이블 생성"""
    if ids.ndim != 1:
        raise ValueError("ids must be 1D")
    if len(ids) < 2:
        raise ValueError("ids must have at least 2 elements")

    counts = np.zeros((vocab_size, vocab_size), dtype=np.int64)
    prev_ids = ids[:-1]
    next_ids = ids[1:]

    for prev_id, next_id in zip(prev_ids, next_ids):
        counts[int(prev_id), int(next_id)] += 1

    return counts


def counts_to_probs(counts: np.ndarray, smoothing: float = 0.0) -> np.ndarray:
    """카운트 테이블을 확률로 변환"""
    counts = np.asarray(counts, dtype=np.float64)
    if smoothing < 0:
        raise ValueError("smoothing must be >= 0")

    counts_smooth = counts + smoothing
    probs = counts_smooth / counts_smooth.sum(axis=1, keepdims=True)
    return probs


# ============================================================================
# Section 3: Checkpoint loading (inline stubs)
# ============================================================================

class BigramNNCheckpoint:
    def __init__(self, tokenizer, W, default_start_id):
        self.tokenizer = tokenizer
        self.W = W
        self.default_start_id = default_start_id


class MLPLMCheckpoint:
    def __init__(self, tokenizer, params, context_len):
        self.tokenizer = tokenizer
        self.params = params
        self.context_len = context_len


def load_bigram_nn(path: str) -> BigramNNCheckpoint:
    """빅램 NN 체크포인트 로드"""
    data = np.load(path, allow_pickle=True)
    return BigramNNCheckpoint(
        tokenizer=data['tokenizer'].item(),
        W=data['W'],
        default_start_id=int(data['default_start_id']),
    )


def load_mlp_lm(path: str) -> MLPLMCheckpoint:
    """MLP LM 체크포인트 로드"""
    data = np.load(path, allow_pickle=True)
    return MLPLMCheckpoint(
        tokenizer=data['tokenizer'].item(),
        params=data['params'].item(),
        context_len=int(data['context_len']),
    )


# ============================================================================
# Section 4: Loss calculation (inline)
# ============================================================================

def eval_loss_bigram_nn(W: np.ndarray, prev_ids: np.ndarray, next_ids: np.ndarray, *, batch_size: int = 4096) -> float:
    """신경망 빅램 모델의 평가 loss 계산"""
    if len(prev_ids) == 0:
        raise ValueError("eval set is empty")
    total = 0.0
    count = 0
    for start in range(0, len(prev_ids), batch_size):
        end = min(len(prev_ids), start + batch_size)
        logits = W[prev_ids[start:end]]
        log_probs = log_softmax(logits, axis=1)
        loss = -log_probs[np.arange(end - start), next_ids[start:end]]
        total += float(loss.sum())
        count += int(end - start)
    return total / count


def eval_loss_mlp(params, X: np.ndarray, y: np.ndarray, *, batch_size: int = 4096) -> float:
    """MLP LM의 평가 loss 계산 (forward pass)"""
    if len(X) == 0:
        raise ValueError("eval set is empty")
    total = 0.0
    count = 0

    # Simple forward pass: E[X] -> W1 -> tanh -> W2 -> logits
    E = params.E
    W1 = params.W1
    b1 = params.b1
    W2 = params.W2
    b2 = params.b2

    for start in range(0, len(X), batch_size):
        end = min(len(X), start + batch_size)
        emb = E[X[start:end]]
        h_in = emb.reshape(len(X[start:end]), -1)
        h_pre = h_in @ W1 + b1
        h = np.tanh(h_pre)
        logits = h @ W2 + b2
        log_probs = log_softmax(logits, axis=1)
        loss = -log_probs[np.arange(end - start), y[start:end]]
        total += float(loss.sum())
        count += int(end - start)
    return total / count


# ============================================================================
# Section 5: evaluate_lm.py - Main evaluation script
# ============================================================================

def perplexity(loss: float) -> float:
    """Cross-entropy loss를 Perplexity로 변환합니다."""
    return float(np.exp(loss))


def add_subcommands(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="cmd", required=True)

    p_counts = sub.add_parser("counts_bigram", help="카운트 기반 빅램 모델 평가")
    p_counts.add_argument("--train", required=True, help="학습 텍스트")
    p_counts.add_argument("--eval", required=True, help="평가(eval) 텍스트 파일")
    p_counts.add_argument("--smoothing", type=float, default=0.0, help="Add-k smoothing (0=끄기)")

    p_bnn = sub.add_parser("bigram_nn", help="신경망 빅램 체크포인트 평가")
    p_bnn.add_argument("--model", required=True, help="bigram_nn 체크포인트 경로(.npz)")
    p_bnn.add_argument("--eval", required=True, help="평가(eval) 텍스트 파일")

    p_mlp = sub.add_parser("mlp_lm", help="MLP LM 체크포인트 평가")
    p_mlp.add_argument("--model", required=True, help="mlp_lm 체크포인트 경로(.npz)")
    p_mlp.add_argument("--eval", required=True, help="평가(eval) 텍스트 파일")


def main() -> None:
    p = argparse.ArgumentParser(description="언어모델 평가(loss + perplexity).")
    add_subcommands(p)
    args = p.parse_args()

    if args.cmd == "counts_bigram":
        train_text = Path(args.train).read_text(encoding="utf-8")
        eval_text = Path(args.eval).read_text(encoding="utf-8")

        tok = CharTokenizer.from_text(train_text)
        train_ids = np.array(tok.encode(train_text), dtype=np.int64)
        counts = build_bigram_counts(train_ids, tok.vocab_size)
        probs = counts_to_probs(counts, smoothing=float(args.smoothing))

        eval_ids = np.array(tok.encode(eval_text), dtype=np.int64)
        if len(eval_ids) < 2:
            raise ValueError("Eval text must contain at least 2 tokens/characters")
        prev_ids = eval_ids[:-1]
        next_ids = eval_ids[1:]
        p_next = probs[prev_ids, next_ids]

        if np.any(p_next == 0.0):
            zero = int((p_next == 0.0).sum())
            print(f"loss=inf  ppl=inf  (zero_prob_pairs={zero}; try --smoothing 1)")
            return

        loss = float(-np.log(p_next).mean())
        print(f"loss={loss:.4f}  ppl={perplexity(loss):.2f}")
        return

    if args.cmd == "bigram_nn":
        ckpt = load_bigram_nn(args.model)

        eval_text = Path(args.eval).read_text(encoding="utf-8")
        eval_ids = np.array(ckpt.tokenizer.encode(eval_text), dtype=np.int64)

        if len(eval_ids) < 2:
            raise ValueError("Eval text must contain at least 2 tokens/characters")

        prev_ids = eval_ids[:-1]
        next_ids = eval_ids[1:]

        loss = float(eval_loss_bigram_nn(ckpt.W, prev_ids, next_ids))
        print(f"loss={loss:.4f}  ppl={perplexity(loss):.2f}")
        return

    if args.cmd == "mlp_lm":
        ckpt = load_mlp_lm(args.model)

        eval_text = Path(args.eval).read_text(encoding="utf-8")
        eval_ids = np.array(ckpt.tokenizer.encode(eval_text), dtype=np.int64)

        X, y = make_context_dataset(eval_ids, ckpt.context_len)

        loss = float(eval_loss_mlp(ckpt.params, X, y))
        print(f"loss={loss:.4f}  ppl={perplexity(loss):.2f}")
        return

    raise AssertionError("unreachable")


if __name__ == "__main__":
    main()
