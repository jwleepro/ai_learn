"""아주 작은 MLP 언어모델(numpy, 수동 backprop).

“최근 k개 토큰(context)”을 입력으로 받아 다음 토큰을 예측합니다.

기호(자주 쓰는 shape):
- V: vocab 크기
- C: context_len
- D: embed_dim
- H: hidden_dim
- B: batch 크기
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import _path_setup  # noqa: F401  (code/ 하위 폴더 sys.path 등록)
from softmax import log_softmax, softmax


@dataclass(frozen=True)
class MLPLMConfig:
    context_len: int = 8
    embed_dim: int = 32
    hidden_dim: int = 128
    lr: float = 0.1
    epochs: int = 50
    batch_size: int = 256
    seed: int = 0
    init_scale: float = 0.02


@dataclass
class MLPLMParams:
    E: np.ndarray  # (V, D)
    W1: np.ndarray  # (context_len*D, H)
    b1: np.ndarray  # (H,)
    W2: np.ndarray  # (H, V)
    b2: np.ndarray  # (V,)


def init_params(vocab_size: int, *, config: MLPLMConfig, rng: np.random.Generator) -> MLPLMParams:
    """모든 파라미터를 초기화합니다.

    가중치 초기화 전략:
    - 임베딩 E, W1, W2: 정규분포 N(0, init_scale)로 초기화
      * init_scale은 보통 0.01~0.02 (너무 크면 그래디언트 폭주, 작으면 훈련 느림)
    - 바이어스 b1, b2: 0으로 초기화 (중립적 시작점)

    Args:
        vocab_size: 어휘 크기 V
        config: MLPLMConfig 객체 (context_len, embed_dim, hidden_dim, init_scale)
        rng: numpy 난수 생성기

    Returns:
        MLPLMParams: 초기화된 모든 파라미터

    파라미터 크기:
        - E: (V, D) - 임베딩 테이블
        - W1: (C*D, H) - 첫번째 선형층 가중치
        - b1: (H,) - 첫번째 바이어스
        - W2: (H, V) - 두번째 선형층 가중치
        - b2: (V,) - 두번째 바이어스
    """
    # 입력 유효성 검사
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")
    if config.context_len <= 0:
        raise ValueError("context_len must be > 0")
    if config.embed_dim <= 0 or config.hidden_dim <= 0:
        raise ValueError("embed_dim and hidden_dim must be > 0")
    if config.init_scale <= 0:
        raise ValueError("init_scale must be > 0")

    # 축약 기호
    D = config.embed_dim      # 임베딩 차원
    H = config.hidden_dim     # 은닉층 차원
    C = config.context_len    # 컨텍스트 길이
    scale = config.init_scale # 초기화 표준편차

    # 임베딩 테이블 초기화: (V, D)
    # 각 토큰이 D차원의 밀집 벡터로 표현됨
    # 정규분포로 초기화하면 훈련 초기에 다양한 그래디언트 신호가 발생
    E = rng.normal(0.0, scale, size=(vocab_size, D)).astype(np.float64)

    # 첫번째 선형층 가중치: (C*D, H)
    # 입력: C개 토큰의 임베딩을 연결한 길이 C*D 벡터
    # 출력: 길이 H의 은닉층
    W1 = rng.normal(0.0, scale, size=(C * D, H)).astype(np.float64)

    # 첫번째 바이어스: (H,)
    # 0으로 초기화 (가중치와 달리, 바이어스는 0에서 시작하는 것이 표준)
    b1 = np.zeros((H,), dtype=np.float64)

    # 두번째 선형층 가중치: (H, V)
    # 입력: 길이 H의 은닉층
    # 출력: 길이 V의 logits (다음 토큰 예측)
    W2 = rng.normal(0.0, scale, size=(H, vocab_size)).astype(np.float64)

    # 두번째 바이어스: (V,)
    b2 = np.zeros((vocab_size,), dtype=np.float64)

    return MLPLMParams(E=E, W1=W1, b1=b1, W2=W2, b2=b2)


def forward(params: MLPLMParams, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """전방향(forward) 계산 - 컨텍스트에서 다음 토큰의 logits를 계산합니다.

    네트워크 구조:
    1. 임베딩층: 토큰 ID들을 밀집 벡터로 변환
    2. 평탄화: 임베딩들을 연결하여 1D 벡터 생성
    3. 첫번째 선형층 + tanh 활성화: 비선형 특성 학습
    4. 두번째 선형층: 다음 토큰의 logits 출력

    Args:
        params: MLPLMParams - E, W1, b1, W2, b2를 포함하는 파라미터 객체
        X: shape (B, C) - 배치의 토큰 ID들 (C = context_len)

    Returns:
        (logits, cache) 튜플:
        - logits: shape (B, V) - 다음 토큰의 logit 점수들 (V = vocab_size)
        - cache: 역전파에서 필요한 중간 계산값들

    정보 흐름 (shape 변화):
        (B, C) [토큰ID]
        -> (B, C, D) [임베딩]
        -> (B, C*D) [평탄화]
        -> (B, H) [숨겨진층, tanh 후]
        -> (B, V) [출력층, logits]
    """

    # 1단계: 임베딩 조회
    # X: (B, C) - 각 샘플은 C개의 토큰 ID
    # E: (V, D) - 어휘 임베딩 테이블
    # emb: (B, C, D) - 각 토큰 ID를 D차원 벡터로 변환
    emb = params.E[X]

    # 2단계: 임베딩 평탄화
    # (B, C, D) -> (B, C*D)
    # C개의 D차원 벡터들을 하나의 긴 벡터로 연결
    # 이를 통해 모든 컨텍스트 정보가 다음 층의 입력이 됨
    h_in = emb.reshape(len(X), -1)

    # 3단계: 첫번째 선형 변환 (W1, b1)
    # (B, C*D) @ (C*D, H) + (H,) = (B, H)
    # 평탄화된 입력을 H차원의 은닉층으로 변환
    h_pre = h_in @ params.W1 + params.b1

    # 4단계: 비선형 활성화 (tanh)
    # tanh: [-1, 1] 범위의 비선형 활성화
    # 신경망이 선형이 아닌 패턴을 학습하도록 함
    # tanh는 수치적으로 안정적이고 그래디언트가 잘 전파됨
    h = np.tanh(h_pre)

    # 5단계: 출력층 (W2, b2)
    # (B, H) @ (H, V) + (V,) = (B, V)
    # 은닉층을 어휘 크기의 logits로 변환
    # logits는 아직 정규화되지 않은 점수들 (나중에 softmax 적용)
    logits = h @ params.W2 + params.b2

    # 역전파를 위해 중간값들을 캐시에 저장
    # forward와 backward를 함께 수행하려면 이들 값이 필요
    cache = {"X": X, "emb": emb, "h_in": h_in, "h_pre": h_pre, "h": h}
    return logits, cache


def loss_and_grads(params: MLPLMParams, X: np.ndarray, y: np.ndarray) -> tuple[float, MLPLMParams]:
    """손실함수(loss)와 각 파라미터의 기울기를 계산합니다(역전파).

    크로스엔트로피 손실함수와 소프트맥스를 조합한 경우, 역전파가 깔끔하게 계산되는
    특별한 구조가 있습니다. 이 함수는 그 구조를 이용해 효율적으로 그래디언트를 계산합니다.

    수학적 개요:
    1. Forward pass에서 logits을 계산하고 softmax로 확률분포로 변환
    2. Loss = -log P(정답) = Cross-Entropy loss
    3. Backward pass에서 미분하면 dLogits = softmax - one_hot(정답) 형태가 나옴
    4. Chain rule을 따라 역전파: 출력층 → 은닉층 → 임베딩 → 가중치

    Args:
        params: MLPLMParams - 현재 학습 가능한 모든 파라미터
        X: shape (B, C) - 배치의 컨텍스트 토큰 ID들
        y: shape (B,) - 배치의 정답 토큰 ID들

    Returns:
        (loss, grads): loss는 배치의 평균 cross-entropy loss,
                      grads는 각 파라미터의 그래디언트(같은 shape)
    """

    # ===== Forward pass =====
    logits, cache = forward(params, X)
    # logits: (B, V) - 아직 정규화되지 않은 점수

    log_probs = log_softmax(logits, axis=1)
    # log_probs: (B, V) - 각 샘플마다 로그확률의 합 = 0 (수치적 안정성)

    # Cross-entropy loss 계산
    # 정답 토큰에 대한 로그확률을 가져와서 음수 취하고 평균
    # loss가 작을수록 모델이 정답에 높은 확률을 할당했다는 뜻
    loss = float(-log_probs[np.arange(len(y)), y].mean())

    # ===== Backward pass =====
    # 핵심: cross-entropy + softmax의 미분은 (softmax - one_hot) 형태

    probs = np.exp(log_probs)  # (B, V) - softmax 확률
    dlogits = probs.copy()      # 예측 확률로 시작
    dlogits[np.arange(len(y)), y] -= 1.0  # 정답 위치에서만 1을 뺌
    # 이제 dlogits = (예측확률 - [정답은 1, 나머지 0]) 형태
    # 직관: 정답 확률을 높이고 비정답 확률을 낮출 신호
    dlogits /= float(len(y))  # 배치 평균으로 정규화
    # dlogits: (B, V) - shape: (batch_size, vocab_size)

    # ===== 출력층 역전파 (W2, b2) =====
    h = cache["h"]  # (B, H) - 은닉층 활성화
    # W2: (H, V) 이므로 h.T @ dlogits = (H, B) @ (B, V) = (H, V)
    dW2 = h.T @ dlogits  # (H, V) - W2의 그래디언트
    db2 = dlogits.sum(axis=0)  # (V,) - b2의 그래디언트 (배치에서 합산)

    # ===== 은닉층 역전파 =====
    # dlogits의 신호를 W2를 통해 역전파
    dh = dlogits @ params.W2.T  # (B, V) @ (V, H) = (B, H)
    # 이제 은닉층 활성화 전 신호를 구하려면 tanh의 미분을 적용
    # tanh'(x) = 1 - tanh(x)^2
    # 이것의 의미: tanh 출력이 ±1 근처면 미분값이 0에 가까움(그래디언트 소실),
    #            0 근처면 미분값이 1에 가까움(그래디언트 통과)
    dh_pre = dh * (1.0 - np.tanh(cache["h_pre"]) ** 2)  # (B, H)

    # ===== 첫번째 선형층 역전파 (W1, b1) =====
    h_in = cache["h_in"]  # (B, C*D) - 평탄화된 임베딩 입력
    # W1: (C*D, H) 이므로 h_in.T @ dh_pre = (C*D, B) @ (B, H) = (C*D, H)
    dW1 = h_in.T @ dh_pre  # (C*D, H) - W1의 그래디언트
    db1 = dh_pre.sum(axis=0)  # (H,) - b1의 그래디언트

    # ===== 임베딩 역전파 (E) =====
    # dh_pre의 신호를 W1을 통해 역전파
    dh_in = dh_pre @ params.W1.T  # (B, H) @ (H, C*D) = (B, C*D)
    # 다시 (B, C, D) 형태로 복구
    dEmb = dh_in.reshape(cache["emb"].shape)  # (B, C, D)

    # 임베딩 테이블 E의 그래디언트 계산
    # 같은 토큰 ID가 여러 위치에 나타나면 그래디언트들이 누적되어야 함
    dE = np.zeros_like(params.E)  # (V, D)
    X_ids = cache["X"].reshape(-1)  # 모든 토큰 ID를 평탄화
    dEmb_flat = dEmb.reshape(-1, dEmb.shape[-1])  # (B*C, D)
    # np.add.at: 같은 인덱스에 여러 번 더하기 (안전한 인덱싱된 덧셈)
    np.add.at(dE, X_ids, dEmb_flat)  # (V, D)

    grads = MLPLMParams(E=dE, W1=dW1, b1=db1, W2=dW2, b2=db2)
    return loss, grads


def apply_grads(params: MLPLMParams, grads: MLPLMParams, *, lr: float) -> None:
    """경사하강법(Gradient Descent)을 이용해 파라미터를 업데이트합니다.

    기본 공식: param_new = param_old - learning_rate * gradient

    - learning_rate가 크면: 빠르게 학습하지만 진동하거나 발산 가능
    - learning_rate가 작으면: 안정적이지만 훈련이 느림
    - 적절한 학습률 선택이 훈련의 성패를 결정함

    Args:
        params: MLPLMParams - 현재 파라미터 (제자리 수정)
        grads: MLPLMParams - 각 파라미터의 그래디언트
        lr: 학습률 (learning rate), 보통 0.001~0.1 범위
    """
    params.E -= lr * grads.E
    params.W1 -= lr * grads.W1
    params.b1 -= lr * grads.b1
    params.W2 -= lr * grads.W2
    params.b2 -= lr * grads.b2


def eval_loss(params: MLPLMParams, X: np.ndarray, y: np.ndarray, *, batch_size: int = 4096) -> float:
    """평가/검증 집합에서 평균 손실을 계산합니다(역전파 없음).

    훈련 중에 과적합을 감지하기 위해 사용합니다:
    - train loss는 계속 내려가지만 val loss는 올라가면 과적합 신호
    - 메모리 제약을 고려해 배치 단위로 나누어 처리

    Args:
        params: 학습된 파라미터
        X: shape (N, C) - 평가 집합의 컨텍스트
        y: shape (N,) - 평가 집합의 정답 토큰
        batch_size: 메모리 효율을 위한 배치 크기 (기본값 4096)

    Returns:
        평균 cross-entropy loss (낮을수록 좋음)
    """
    if len(X) == 0:
        raise ValueError("eval set is empty")
    total = 0.0
    count = 0
    # 배치 단위로 나누어 처리 (메모리 절약)
    for start in range(0, len(X), batch_size):
        end = min(len(X), start + batch_size)
        logits, _ = forward(params, X[start:end])
        log_probs = log_softmax(logits, axis=1)
        # 각 샘플의 정답 토큰에 대한 로그확률
        loss = -log_probs[np.arange(end - start), y[start:end]]
        total += float(loss.sum())
        count += int(end - start)
    return total / count


def train_mlp_lm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    vocab_size: int,
    *,
    config: MLPLMConfig,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> tuple[MLPLMParams, list[dict[str, float]]]:
    """MLP 언어모델을 SGD(확률적 경사하강법)로 훈련합니다.

    훈련 과정(Training Loop):
    1. 파라미터 무작위 초기화
    2. 각 에포크(epoch)마다:
       a. 훈련 데이터를 무작위로 섞음 (shuffling) - 데이터 분포가 균등해짐
       b. 배치 단위로 나누어 처리
       c. 배치마다 loss와 그래디언트 계산
       d. 파라미터 업데이트 (경사하강법)
       e. 선택사항: 검증 손실 평가
    3. 각 에포크의 손실 기록

    Shuffling의 중요성:
    - 데이터가 특정 순서로 정렬되어 있으면 배치별 통계가 왜곡됨
    - 무작위 순서 처리로 더 나은 그래디언트 추정

    Args:
        X_train: shape (N, C) - 훈련 집합의 컨텍스트
        y_train: shape (N,) - 훈련 집합의 정답 토큰
        vocab_size: 어휘 크기 V
        config: MLPLMConfig - 학습률, 에포크 수, 배치 크기 등
        X_val, y_val: 검증 집합 (선택사항, 과적합 감시용)

    Returns:
        (params, history): 최종 학습 파라미터와 훈련 히스토리
        - params: MLPLMParams - 훈련된 모든 파라미터
        - history: 각 에포크의 {"epoch": ..., "train_loss": ..., "val_loss": ...}
    """
    rng = np.random.default_rng(config.seed)
    params = init_params(vocab_size, config=config, rng=rng)

    history: list[dict[str, float]] = []
    n = len(X_train)
    if n == 0:
        raise ValueError("train set is empty")

    # 에포크 루프: 데이터를 여러 번 반복 처리
    for epoch in range(1, config.epochs + 1):
        # 데이터 섞기 (매 에포크마다 무작위 순열 생성)
        perm = rng.permutation(n)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        epoch_loss = 0.0
        steps = 0
        # 배치 루프: 에포크 내 배치 단위 처리
        for start in range(0, n, config.batch_size):
            end = min(n, start + config.batch_size)
            # Loss와 그래디언트 계산
            loss, grads = loss_and_grads(params, X_shuf[start:end], y_shuf[start:end])
            # 파라미터 업데이트
            apply_grads(params, grads, lr=config.lr)
            epoch_loss += loss
            steps += 1

        # 에포크 평균 손실
        train_loss = epoch_loss / max(steps, 1)
        metrics: dict[str, float] = {"epoch": float(epoch), "train_loss": float(train_loss)}

        # 검증 손실 (선택사항)
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_loss = eval_loss(params, X_val, y_val)
            metrics["val_loss"] = float(val_loss)
        history.append(metrics)

    return params, history


def next_token_probs(params: MLPLMParams, context_ids: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    """주어진 컨텍스트에서 다음 토큰의 확률분포를 계산합니다.

    온도(temperature) 조절:
    - temperature < 1: 분포가 뾰족해짐 (더 "자신감 있는" 선택)
    - temperature = 1: 표준 softmax
    - temperature > 1: 분포가 평탄해짐 (더 "다양한" 선택)

    예: temperature=0.1이면 가장 확률 높은 토큰에 집중,
        temperature=2.0이면 여러 토큰에 골고루 확률 분산

    Args:
        params: MLPLMParams - 학습된 파라미터
        context_ids: shape (C,) - 컨텍스트 토큰 ID들
        temperature: 샘플링 온도 (기본값 1.0)

    Returns:
        shape (V,) - 다음 토큰의 확률분포 P(next | context)
    """
    if context_ids.ndim != 1:
        raise ValueError("context_ids must be 1D")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits, _ = forward(params, context_ids.reshape(1, -1))
    # 온도로 logits 스케일 조절
    logits = logits[0] / float(temperature)
    return softmax(logits, axis=0)
