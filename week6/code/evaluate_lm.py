"""언어모델 평가(loss, perplexity) CLI.

지원:
- counts_bigram: 카운트 기반 빅램
- bigram_nn: 신경망 빅램 체크포인트(.npz)
- mlp_lm: MLP LM 체크포인트(.npz)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _path_setup  # noqa: F401  (code/ 하위 폴더 sys.path 등록)
from bigram_counts import build_bigram_counts, counts_to_probs
from bigram_nn import eval_loss as eval_loss_bigram_nn
from dataset_lm import make_context_dataset
from mlp_lm import eval_loss as eval_loss_mlp
from model_io import load_bigram_nn, load_mlp_lm
from tokenizer_char import CharTokenizer


def perplexity(loss: float) -> float:
    """Cross-entropy loss를 Perplexity로 변환합니다.

    Perplexity (PPL)는 언어모델 평가의 표준 지표입니다.

    수학:
    PPL = exp(loss) = exp(-1/N * sum(log P(y_i)))

    직관:
    - loss가 작을수록 (log 확률이 높을수록) PPL도 작아짐
    - PPL이 작을수록 모델이 데이터를 잘 맞히고 있다는 뜻
    - 예: PPL=100이면 "평균적으로 다음 토큰을 100개 선택지에서 고르는 정도의 확률"

    비교:
    - PPL은 매우 직관적 (exp 스케일)
    - loss는 계산 목적으로 사용 (더 작은 숫자)

    Args:
        loss: cross-entropy loss (낮을수록 좋음)

    Returns:
        perplexity 값 (낮을수록 좋음)
    """
    return float(np.exp(loss))


def add_subcommands(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="cmd", required=True)

    p_counts = sub.add_parser("counts_bigram", help="카운트 기반 빅램 모델 평가")
    p_counts.add_argument("--train", required=True, help="학습 텍스트(카운트/확률표를 이 파일로 만듦)")
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
        # ===== 카운트 기반 빅램 모델 평가 =====
        train_text = Path(args.train).read_text(encoding="utf-8")
        eval_text = Path(args.eval).read_text(encoding="utf-8")

        # 훈련 텍스트로부터 토크나이저 및 확률표 생성
        tok = CharTokenizer.from_text(train_text)
        train_ids = np.array(tok.encode(train_text), dtype=np.int64)
        # 빅램 (이전 토큰 -> 다음 토큰) 카운트 테이블 생성
        counts = build_bigram_counts(train_ids, tok.vocab_size)
        # 카운트를 확률로 변환 (smoothing 적용 가능)
        probs = counts_to_probs(counts, smoothing=float(args.smoothing))

        # 평가 텍스트에서 (prev_token, next_token) 쌍의 확률 계산
        eval_ids = np.array(tok.encode(eval_text), dtype=np.int64)
        if len(eval_ids) < 2:
            raise ValueError("Eval text must contain at least 2 tokens/characters")
        prev_ids = eval_ids[:-1]  # 첫 번째부터 마지막 이전까지
        next_ids = eval_ids[1:]   # 두 번째부터 마지막까지
        # 각 쌍의 조건부 확률 조회
        p_next = probs[prev_ids, next_ids]

        # 확률이 0인 경우 (훈련 데이터에 없는 쌍) 처리
        if np.any(p_next == 0.0):
            zero = int((p_next == 0.0).sum())
            print(f"loss=inf  ppl=inf  (zero_prob_pairs={zero}; try --smoothing 1)")
            return

        # Loss 계산: Cross-entropy loss
        # loss = -1/N * sum(log P(next_token | prev_token))
        # 낮을수록 좋음 (모델이 정답에 높은 확률 할당)
        loss = float(-np.log(p_next).mean())
        print(f"loss={loss:.4f}  ppl={perplexity(loss):.2f}")
        return

    if args.cmd == "bigram_nn":
        # ===== 신경망 빅램 모델 평가 =====
        # 체크포인트 로드 (학습된 가중치 W + 토크나이저)
        ckpt = load_bigram_nn(args.model)

        # 평가 텍스트 로드 및 토큰화
        eval_text = Path(args.eval).read_text(encoding="utf-8")
        eval_ids = np.array(ckpt.tokenizer.encode(eval_text), dtype=np.int64)

        if len(eval_ids) < 2:
            raise ValueError("Eval text must contain at least 2 tokens/characters")

        # (prev_id, next_id) 쌍 생성
        prev_ids = eval_ids[:-1]
        next_ids = eval_ids[1:]

        # 평균 loss 계산
        # W[prev_ids]에서 logits을 가져와 softmax -> log 확률
        loss = float(eval_loss_bigram_nn(ckpt.W, prev_ids, next_ids))
        print(f"loss={loss:.4f}  ppl={perplexity(loss):.2f}")
        return

    if args.cmd == "mlp_lm":
        # ===== MLP LM 모델 평가 =====
        # 체크포인트 로드 (파라미터 + 토크나이저 + 설정)
        ckpt = load_mlp_lm(args.model)

        # 평가 텍스트 로드 및 토큰화
        eval_text = Path(args.eval).read_text(encoding="utf-8")
        eval_ids = np.array(ckpt.tokenizer.encode(eval_text), dtype=np.int64)

        # 컨텍스트 데이터셋 생성
        # (컨텍스트, 정답) 쌍으로 변환
        X, y = make_context_dataset(eval_ids, ckpt.context_len)

        # 평균 loss 계산
        loss = float(eval_loss_mlp(ckpt.params, X, y))
        print(f"loss={loss:.4f}  ppl={perplexity(loss):.2f}")
        return

    raise AssertionError("unreachable")


if __name__ == "__main__":
    main()
