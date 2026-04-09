"""Week 2: 신경망 빅램 언어모델(Neural Bigram Language Model).

이 파일은 bigram_nn.py, generate_bigram_nn.py, train_bigram_nn.py의
모든 코드를 합친 통합 파일입니다.

내용:
- BigramNN 모델: 학습 가능한 가중치 행렬 W를 이용한 빅램 모델
- 훈련: SGD(확률적 경사하강법)로 모델 학습
- 생성: 학습된 모델로 텍스트 생성

주요 개념:
- vocab 크기: V
- W: (V, V) - 학습 가능한 가중치 행렬
- prev 토큰 id가 i일 때, logits = W[i] (다음 토큰 점수)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Import soft utilities (these are assumed to be available)
from softmax import log_softmax, softmax


# ============================================================================
# Section 1: bigram_nn.py - Model definition and training
# ============================================================================

@dataclass(frozen=True)
class BigramNNConfig:
    lr: float = 1.0
    epochs: int = 20
    batch_size: int = 1024
    seed: int = 0
    init_scale: float = 0.01


def init_W(vocab_size: int, rng: np.random.Generator, *, scale: float = 0.01) -> np.ndarray:
    """
    가중치 행렬 W를 정규분포로 초기화

    가중치 초기화의 중요성:
    - 너무 크면: 그래디언트 폭발(exploding gradients)
    - 너무 작으면: 그래디언트 소실(vanishing gradients)
    - 정규분포 사용: 평균 0, 표준편차 작은 값으로 균형잡힌 초기화

    Args:
        vocab_size: V (어휘 크기)
        rng: 난수 생성기
        scale: 초기화 표준편차 (기본값 0.01)

    Returns:
        shape (V, V) - W[i, j]는 토큰i 다음에 토큰j의 점수
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")
    if scale <= 0:
        raise ValueError("scale must be > 0")
    return rng.normal(0.0, scale, size=(vocab_size, vocab_size)).astype(np.float64)


def loss_and_grad_W(W: np.ndarray, prev_ids: np.ndarray, next_ids: np.ndarray) -> tuple[float, np.ndarray]:
    """
    배치에 대한 Cross-Entropy Loss와 W의 그래디언트를 계산합니다

    역전파(Backpropagation) 설명:

    1. Forward Pass:
       - logits = W[prev_ids]  (shape: B x V)
       - log_probs = log_softmax(logits, axis=1)  (각 샘플마다 합=0)
       - loss = -mean(log_probs[i, target_i])  (정답 토큰의 로그확률의 음수)

    2. Loss 의미:
       - 정답 토큰에 높은 확률(낮은 음수 로그)을 할당할수록 loss 감소
       - 모든 비정답 토큰에는 낮은 확률을 할당하게 강제

    3. Gradient (역전파):
       - Cross-Entropy + Softmax의 "깔끔한" 결합:
         dL/dlogits = softmax(logits) - one_hot(target)
       - 직관: 각 토큰마다 (예측확률 - 정답지시자)
         - 정답 위치(=1): dL/dlogits = pred_prob - 1  (음수, logit 내림)
         - 비정답 위치(=0): dL/dlogits = pred_prob  (양수, logit 내림)
         - 즉, 정답 확률은 올리고 비정답 확률은 내려라!

       - dL/dW: chain rule로 grad_logits를 prev_ids의 피쳐로 곱함
         - logits[i] = W[prev_ids[i], :] (한 줄씩)
         - 그래서 W의 각 행은 해당 prev_id가 나타날 때만 업데이트됨

    Shapes:
    - W: (V, V) - 파라미터 행렬
    - prev_ids: (B,) - 배치의 이전 토큰 ID들
    - next_ids: (B,) - 배치의 다음 토큰 ID들 (정답)
    - returns grad_W: (V, V) - W의 그래디언트 (prev에 해당하는 행(row)만 비영)
    """

    if prev_ids.ndim != 1 or next_ids.ndim != 1:
        raise ValueError("prev_ids and next_ids must be 1D")
    if len(prev_ids) != len(next_ids):
        raise ValueError("prev_ids and next_ids must have same length")
    if len(prev_ids) == 0:
        raise ValueError("batch is empty")

    # Forward pass
    logits = W[prev_ids]  # (B, V) - 각 샘플의 logit 점수
    log_probs = log_softmax(logits, axis=1)  # (B, V) - 로그확률

    # Loss: Cross-entropy loss
    # -log P(target | prev) 의 평균
    # 낮을수록 좋음 (정답에 높은 확률을 할당했다는 뜻)
    loss = float(-log_probs[np.arange(len(next_ids)), next_ids].mean())

    # Backward pass (역전파)
    # cross-entropy + softmax의 미분은 매우 깔끔하게 나옴:
    # dL/dlogits = softmax(logits) - one_hot(target)
    probs = np.exp(log_probs)  # (B, V) - softmax 확률
    grad_logits = probs.copy()  # 예측 확률로 시작
    grad_logits[np.arange(len(next_ids)), next_ids] -= 1.0  # 정답 위치는 1을 뺌
    grad_logits /= float(len(next_ids))  # 배치 평균

    # W의 그래디언트 계산
    # dL/dW[i, :] = dL/dlogits를 prev_id=i인 샘플들에 대해 누적
    grad_W = np.zeros_like(W)
    np.add.at(grad_W, prev_ids, grad_logits)  # 인덱싱된 누적 덧셈
    return loss, grad_W


def eval_loss(W: np.ndarray, prev_ids: np.ndarray, next_ids: np.ndarray, *, batch_size: int = 4096) -> float:
    """
    평가 집합(validation/test set)에서 평균 loss 계산

    훈련 중에 과적합(overfitting)을 감지하기 위해 사용:
    - train loss는 계속 내려가지만 val loss는 올라가면 과적합 신호

    Args:
        W: 학습된 가중치 행렬
        prev_ids: 평가 집합의 이전 토큰들
        next_ids: 평가 집합의 다음 토큰들 (정답)
        batch_size: 메모리 절약을 위한 배치 크기

    Returns:
        평균 cross-entropy loss
    """
    if len(prev_ids) == 0:
        raise ValueError("eval set is empty")
    total = 0.0
    count = 0
    # 배치 단위로 나누어 처리 (메모리 제약)
    for start in range(0, len(prev_ids), batch_size):
        end = min(len(prev_ids), start + batch_size)
        logits = W[prev_ids[start:end]]
        log_probs = log_softmax(logits, axis=1)
        # 정답 토큰의 로그확률
        loss = -log_probs[np.arange(end - start), next_ids[start:end]]
        total += float(loss.sum())
        count += int(end - start)
    return total / count


def train_bigram_nn(
    prev_train: np.ndarray,
    next_train: np.ndarray,
    vocab_size: int,
    *,
    config: BigramNNConfig,
    prev_val: np.ndarray | None = None,
    next_val: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """
    신경망 빅램 모델을 SGD(확률적 경사하강법)로 훈련

    훈련 과정:
    1. 가중치 W를 무작위로 초기화
    2. 매 에포크마다:
       - 훈련 데이터를 섞고 (shuffling)
       - 배치 단위로 나누어
       - 각 배치에서 loss와 그래디언트 계산
       - W 업데이트: W -= lr * grad_W
    3. 매 에포크 후 평균 손실 기록
    4. 선택사항: 검증 손실도 기록

    Args:
        prev_train: 훈련 집합 이전 토큰 (shape: N,)
        next_train: 훈련 집합 다음 토큰/정답 (shape: N,)
        vocab_size: 어휘 크기 V
        config: BigramNNConfig - 학습율, 에포크 수, 배치 크기 등
        prev_val, next_val: 검증 집합 (선택사항)

    Returns:
        (최종 가중치 W, 훈련 히스토리)
        - W: shape (V, V)
        - history: 각 에포크의 {"epoch": ..., "train_loss": ..., "val_loss": ...}
    """
    rng = np.random.default_rng(config.seed)
    W = init_W(vocab_size, rng, scale=config.init_scale)

    history: list[dict[str, float]] = []
    n = len(prev_train)
    if n == 0:
        raise ValueError("train set is empty")

    # 에포크 루프: 여러 번 데이터를 반복
    for epoch in range(1, config.epochs + 1):
        # 매 에포크마다 데이터를 섞기 (무작위 순서로 처리하여 수렴성 향상)
        perm = rng.permutation(n)
        prev_shuf = prev_train[perm]
        next_shuf = next_train[perm]

        # 배치 루프
        epoch_loss = 0.0
        steps = 0
        for start in range(0, n, config.batch_size):
            end = min(n, start + config.batch_size)
            # Loss와 그래디언트 계산
            loss, grad_W = loss_and_grad_W(W, prev_shuf[start:end], next_shuf[start:end])
            # 파라미터 업데이트 (경사하강법)
            # W의 각 행은 해당 prev_id가 나타날 때만 grad_W에서 비영(non-zero)
            W -= config.lr * grad_W
            epoch_loss += loss
            steps += 1

        # 에포크 평균 손실
        train_loss = epoch_loss / max(steps, 1)
        metrics: dict[str, float] = {"epoch": float(epoch), "train_loss": float(train_loss)}

        # 검증 손실 (선택사항)
        if prev_val is not None and next_val is not None and len(prev_val) > 0:
            val_loss = eval_loss(W, prev_val, next_val)
            metrics["val_loss"] = float(val_loss)
        history.append(metrics)

    return W, history


def bigram_probs(W: np.ndarray, prev_id: int, *, temperature: float = 1.0) -> np.ndarray:
    """
    신경망으로부터 조건부 확률분포를 계산

    Args:
        W: 학습된 가중치 행렬 (V, V)
        prev_id: 이전 토큰 ID
        temperature: 온도값 (1.0 = 표준 softmax)

    Returns:
        shape (V,) - 다음 토큰의 확률분포 P(next | prev_id)

    계산:
    1. logits = W[prev_id] (신경망 출력 점수)
    2. logits /= temperature (온도 조절)
    3. probs = softmax(logits) (확률분포로 변환)
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = W[prev_id] / float(temperature)
    return softmax(logits, axis=0)


# ============================================================================
# Section 2: generate_bigram_nn.py - Generation with CLI
# ============================================================================

def parse_generate_args() -> argparse.Namespace:
    """생성용 CLI 인자를 파싱합니다."""
    p = argparse.ArgumentParser(description="신경망 빅램 LM으로 텍스트 생성(numpy).")
    p.add_argument("--model", default="llm_from_scratch/models/bigram_nn.npz", help="체크포인트 경로(.npz)")
    p.add_argument("--length", type=int, default=400, help="생성할 글자 수")
    p.add_argument("--seed", type=int, default=0, help="난수 시드(seed)")
    p.add_argument("--start_id", type=int, default=None, help="시작 토큰 id(콘솔 인코딩 이슈 회피용)")
    p.add_argument("--temperature", type=float, default=1.0, help="샘플링 온도(>0)")
    p.add_argument("--top_k", type=int, default=None, help="top-k 샘플링(선택)")
    p.add_argument("--top_p", type=float, default=None, help="top-p 샘플링(선택)")
    return p.parse_args()


def generate(args: argparse.Namespace | None = None) -> None:
    """생성 메인 로직.

    처리 순서:
    1. 학습된 체크포인트 로드 (가중치 W + 토크나이저)
    2. 시작 토큰 결정 (지정 또는 기본값)
    3. 자동회귀 루프: 이전 토큰 -> 다음 토큰 샘플링 반복
    4. 생성된 토큰 ID들을 문자로 디코딩하여 출력

    주의: 이 함수는 model_io와 sampling 모듈을 사용합니다.
    standalone으로 실행하려면 이들 모듈이 필요합니다.
    """
    if args is None:
        args = parse_generate_args()

    # 아래는 의사코드: 실제 실행을 위해서는 model_io, sampling 등이 필요
    print("Generate 모드는 별도의 model_io.py와 sampling.py이 필요합니다.")
    print("train_bigram_nn.py를 먼저 실행해 모델을 저장한 후 생성하세요.")


# ============================================================================
# Section 3: train_bigram_nn.py - Training with CLI
# ============================================================================

def parse_train_args() -> argparse.Namespace:
    """훈련용 CLI 인자를 파싱합니다."""
    p = argparse.ArgumentParser(description="신경망 빅램 LM 학습(numpy).")
    p.add_argument("--input", required=True, help="입력 UTF-8 텍스트 파일 경로")
    p.add_argument("--out", default="llm_from_scratch/models/bigram_nn.npz", help="체크포인트 저장 경로(.npz)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2.0, help="학습률(learning rate)")
    p.add_argument("--batch", type=int, default=2048, help="배치 크기(batch size)")
    p.add_argument("--seed", type=int, default=0, help="난수 시드(seed)")
    p.add_argument("--val_frac", type=float, default=0.1, help="검증 데이터 비율(0~0.5)")
    return p.parse_args()


def train(args: argparse.Namespace | None = None) -> None:
    """훈련 메인 로직.

    처리 순서:
    1. 텍스트 파일 읽기
    2. 문자 토크나이저 생성 (사전학습된 토크나이저 없으므로 텍스트에서 직접 생성)
    3. 텍스트를 토큰 ID 배열로 변환
    4. (prev_id, next_id) 쌍 생성 (빅램)
    5. 데이터를 훈련/검증으로 분할
    6. 신경망 빅램 모델 훈련
    7. 체크포인트 저장 (.npz 파일에 토크나이저와 가중치 포함)

    주의: 이 함수는 tokenizer_char와 model_io 모듈을 사용합니다.
    standalone으로 실행하려면 이들 모듈이 필요합니다.
    """
    if args is None:
        args = parse_train_args()

    # 아래는 의사코드: 실제 실행을 위해서는 tokenizer_char, model_io 등이 필요
    print("Train 모드는 별도의 tokenizer_char.py와 model_io.py이 필요합니다.")
    print("example usage:")
    print("  python week2_complete.py train --input data/tiny_shakespeare.txt --epochs 30 --lr 2.0")


def main() -> None:
    """메인 진입점 - 모드 선택."""
    p = argparse.ArgumentParser(description="Week 2: 신경망 빅램 언어모델(통합)")
    p.add_argument("mode", nargs="?", default="demo", help="모드: train, generate, demo")
    args, remaining = p.parse_known_args()

    if args.mode == "train":
        import sys
        sys.argv = [sys.argv[0]] + remaining
        train_args = parse_train_args()
        train(train_args)
    elif args.mode == "generate":
        import sys
        sys.argv = [sys.argv[0]] + remaining
        gen_args = parse_generate_args()
        generate(gen_args)
    else:
        # Demo mode
        print("=" * 60)
        print("Week 2: 신경망 빅램 언어모델(Neural Bigram LM)")
        print("=" * 60)
        print("\n이 통합 파일은 bigram_nn.py, train_bigram_nn.py, generate_bigram_nn.py를")
        print("하나의 파일로 결합한 것입니다.")
        print("\n주요 컴포넌트:")
        print("  - BigramNNConfig: 모델 설정 (학습률, 에포크, 배치 크기)")
        print("  - init_W(): 가중치 행렬 초기화")
        print("  - loss_and_grad_W(): Cross-entropy loss와 그래디언트 계산")
        print("  - train_bigram_nn(): SGD로 모델 훈련")
        print("  - bigram_probs(): 다음 토큰의 확률분포 계산")
        print("\n사용법:")
        print("  python week2_complete.py train --input <text_file> --epochs 30")
        print("  python week2_complete.py generate --model <checkpoint.npz> --length 400")


if __name__ == "__main__":
    main()
