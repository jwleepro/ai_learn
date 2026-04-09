"""(학습 아님) Transformer forward 계산을 numpy로 구현한 데모용 코드.

이 파일의 목적:
- Transformer 블록이 어떤 shape로 이어지는지 확인
- residual/LayerNorm/Attention/FFN 조립 감 잡기

주의:
- 가중치는 랜덤 초기화이며, 학습/역전파는 다루지 않습니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import _path_setup  # noqa: F401  (code/ 하위 폴더 sys.path 등록)
from softmax import softmax


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    max_seq_len: int = 64
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 256
    n_layers: int = 2
    seed: int = 0


@dataclass
class TransformerLayerParams:
    ln1_g: np.ndarray
    ln1_b: np.ndarray
    Wq: np.ndarray
    Wk: np.ndarray
    Wv: np.ndarray
    Wo: np.ndarray
    ln2_g: np.ndarray
    ln2_b: np.ndarray
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray


@dataclass
class TransformerParams:
    tok_emb: np.ndarray  # (V, D)
    pos_emb: np.ndarray  # (T, D)
    layers: list[TransformerLayerParams]
    ln_f_g: np.ndarray
    ln_f_b: np.ndarray
    W_out: np.ndarray  # (D, V)
    b_out: np.ndarray  # (V,)


def layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, *, eps: float = 1e-5) -> np.ndarray:
    """Layer Normalization을 적용합니다.

    Layer Norm의 목적:
    - 네트워크가 깊어질수록 각 층의 입력 분포가 shift/scale되는 현상 발생 (covariate shift)
    - 이로 인해 훈련이 느려지고 그래디언트 흐름이 방해됨
    - 해결: 각 샘플의 특성(feature)별로 평균=0, 분산=1로 정규화
    - 학습 가능한 스케일(g)과 시프트(b)로 표현력 복원

    수식:
    y = g * (x - mean) / sqrt(var + eps) + b

    - mean, var: 특성 차원에서 계산 (각 샘플별로 독립적)
    - eps: 수치 안정성 (var이 0일 때 나누기 오류 방지)
    - g, b: 학습 가능한 파라미터 (affine transformation)

    Args:
        x: shape (..., D) - 입력 (마지막 차원이 특성)
        g: shape (D,) - 스케일 파라미터 (gamma, 보통 1로 초기화)
        b: shape (D,) - 시프트 파라미터 (beta, 보통 0으로 초기화)
        eps: 수치 안정성을 위한 작은 값 (기본값 1e-5)

    Returns:
        정규화되고 affine 변환된 결과 (입력과 같은 shape)

    Shape:
        x: (..., D)  (배치, 시퀀스 길이, 특성 차원 등)
        g, b: (D,)   (특성 차원만)
        output: (..., D)  (같은 shape)
    """
    # mean, var은 특성 차원(axis=-1)을 제외한 모든 차원에서 계산됨
    mean = x.mean(axis=-1, keepdims=True)  # 평균: 각 샘플별로 특성들의 평균

    # 분산: 각 특성이 평균에서 얼마나 떨어져 있는지의 정도
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)

    # 정규화: (x - mean) / sqrt(var)로 평균=0, 표준편차=1로 변환
    # eps 추가: var=0일 때 나누기 오류 방지
    x_hat = (x - mean) / np.sqrt(var + eps)

    # Affine transformation: 스케일 g와 시프트 b로 표현력 복원
    # 이를 통해 네트워크가 정규화된 특성을 다시 필요한 대로 조정 가능
    return x_hat * g + b


def init_params(cfg: TransformerConfig) -> TransformerParams:
    if cfg.vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")

    rng = np.random.default_rng(cfg.seed)
    scale = 0.02

    tok_emb = rng.normal(0.0, scale, size=(cfg.vocab_size, cfg.d_model)).astype(np.float64)
    pos_emb = rng.normal(0.0, scale, size=(cfg.max_seq_len, cfg.d_model)).astype(np.float64)

    layers: list[TransformerLayerParams] = []
    for _ in range(cfg.n_layers):
        ln1_g = np.ones((cfg.d_model,), dtype=np.float64)
        ln1_b = np.zeros((cfg.d_model,), dtype=np.float64)
        Wq = rng.normal(0.0, scale, size=(cfg.d_model, cfg.d_model)).astype(np.float64)
        Wk = rng.normal(0.0, scale, size=(cfg.d_model, cfg.d_model)).astype(np.float64)
        Wv = rng.normal(0.0, scale, size=(cfg.d_model, cfg.d_model)).astype(np.float64)
        Wo = rng.normal(0.0, scale, size=(cfg.d_model, cfg.d_model)).astype(np.float64)
        ln2_g = np.ones((cfg.d_model,), dtype=np.float64)
        ln2_b = np.zeros((cfg.d_model,), dtype=np.float64)
        W1 = rng.normal(0.0, scale, size=(cfg.d_model, cfg.d_ff)).astype(np.float64)
        b1 = np.zeros((cfg.d_ff,), dtype=np.float64)
        W2 = rng.normal(0.0, scale, size=(cfg.d_ff, cfg.d_model)).astype(np.float64)
        b2 = np.zeros((cfg.d_model,), dtype=np.float64)
        layers.append(
            TransformerLayerParams(
                ln1_g=ln1_g,
                ln1_b=ln1_b,
                Wq=Wq,
                Wk=Wk,
                Wv=Wv,
                Wo=Wo,
                ln2_g=ln2_g,
                ln2_b=ln2_b,
                W1=W1,
                b1=b1,
                W2=W2,
                b2=b2,
            )
        )

    ln_f_g = np.ones((cfg.d_model,), dtype=np.float64)
    ln_f_b = np.zeros((cfg.d_model,), dtype=np.float64)
    W_out = rng.normal(0.0, scale, size=(cfg.d_model, cfg.vocab_size)).astype(np.float64)
    b_out = np.zeros((cfg.vocab_size,), dtype=np.float64)
    return TransformerParams(tok_emb=tok_emb, pos_emb=pos_emb, layers=layers, ln_f_g=ln_f_g, ln_f_b=ln_f_b, W_out=W_out, b_out=b_out)


def mha(x: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, Wo: np.ndarray, *, n_heads: int, causal: bool) -> tuple[np.ndarray, np.ndarray]:
    """Multi-Head Attention (MHA) Forward 계산.

    다중 헤드 어텐션의 아이디어:
    - 단일 헤드 어텐션: 한 가지 관점에서만 정보 교환
    - 다중 헤드: 여러 개의 "부분 공간(subspace)"에서 동시에 어텐션 수행
      * 각 헤드는 다른 특성이나 의존성에 집중
      * 예: head 0은 문법 구조, head 1은 의미론, ...
    - 최종 출력: 모든 헤드의 결과를 연결(concatenate)하고 선형 투영

    구현 디테일:
    - D를 n_heads개로 분할: D = n_heads * Dh
    - 각 헤드는 Dh 차원에서 독립적으로 어텐션 수행
    - 배치 처리 효율성을 위해 헤드 차원을 앞에 놓음: (H, T, Dh)

    Args:
        x: shape (T, D) - 입력 시퀀스 (T=길이, D=모델 차원)
        Wq, Wk, Wv: shape (D, D) - Query/Key/Value 투영 행렬
        Wo: shape (D, D) - 출력 투영 행렬 (모든 헤드 연결 후)
        n_heads: 헤드 개수 (D는 n_heads로 나누어떨어져야 함)
        causal: True면 causal mask 적용

    Returns:
        (out, weights) 튜플:
        - out: shape (T, D) - MHA 출력
        - weights: shape (H, T, T) - 각 헤드의 어텐션 가중치

    Information flow:
        x (T, D)
        -> Q = x @ Wq (T, D)
        -> reshape to (T, H, Dh) and transpose to (H, T, Dh)
        -> attention for each head: (H, T, Dh) x (H, Dh, T) = (H, T, T)
        -> output per head: (H, T, Dh)
        -> transpose to (T, H, Dh) and reshape to (T, D)
        -> output projection: (T, D) @ (D, D) = (T, D)
    """
    # ===== 입력 shapes 추출 =====
    T, D = x.shape  # T: 시퀀스 길이, D: 모델 차원
    Dh = D // n_heads  # 각 헤드의 차원

    # ===== Step 1: Q, K, V 투영 및 헤드로 분할 =====
    # (T, D) @ (D, D) = (T, D) -> reshape to (T, H, Dh) -> transpose to (H, T, Dh)
    Q = (x @ Wq).reshape(T, n_heads, Dh).transpose(1, 0, 2)  # (H, T, Dh)
    K = (x @ Wk).reshape(T, n_heads, Dh).transpose(1, 0, 2)  # (H, T, Dh)
    V = (x @ Wv).reshape(T, n_heads, Dh).transpose(1, 0, 2)  # (H, T, Dh)

    # ===== Step 2: 각 헤드에서 어텐션 점수 계산 =====
    # (H, T, Dh) @ (H, Dh, T) = (H, T, T)
    # 각 헤드는 독립적으로 모든 토큰 쌍의 유사성 계산
    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(float(Dh))  # (H, T, T)
    # sqrt(Dh)로 정규화: 차원이 커질수록 내적이 커지는 것을 보정

    # ===== Step 3: Causal mask 적용 (자동회귀용) =====
    if causal:
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = scores.copy()
        scores[:, mask] = -1e9  # 미래 토큰 마스킹

    # ===== Step 4: Softmax로 가중치 계산 =====
    weights = softmax(scores, axis=-1)  # (H, T, T)

    # ===== Step 5: Value의 가중합 =====
    # (H, T, T) @ (H, T, Dh) = (H, T, Dh)
    out = weights @ V  # (H, T, Dh)

    # ===== Step 6: 헤드 결과 연결 및 출력 투영 =====
    # (H, T, Dh) -> transpose to (T, H, Dh) -> reshape to (T, D)
    out = out.transpose(1, 0, 2).reshape(T, D)  # (T, D)
    # 최종 선형 투영
    out = out @ Wo  # (T, D) @ (D, D) = (T, D)

    return out, weights


def ffn(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Feed-Forward Network (FFN) Forward 계산.

    Transformer의 각 블록은 어텐션 다음에 FFN을 적용합니다.
    FFN은 위치별(position-wise)로 독립적으로 작용하는 2층 다층 퍼셉트론입니다.

    구조:
    1. 선형 변환: (T, D) @ (D, d_ff) -> (T, d_ff)
    2. 활성화 함수 (ReLU): 비선형성 추가
    3. 선형 변환: (T, d_ff) @ (d_ff, D) -> (T, D)

    역할:
    - 어텐션은 토큰 간 관계 학습
    - FFN은 각 토큰의 표현력(representation capacity) 증가

    ReLU의 의미:
    - 음수: 0으로 설정 (비활성화)
    - 양수: 그대로 통과 (활성화)
    - 단순하고 효율적인 비선형 활성화

    Args:
        x: shape (T, D) - 입력 시퀀스
        W1: shape (D, d_ff) - 첫번째 선형층 가중치
        b1: shape (d_ff,) - 첫번째 바이어스
        W2: shape (d_ff, D) - 두번째 선형층 가중치
        b2: shape (D,) - 두번째 바이어스

    Returns:
        shape (T, D) - FFN 출력

    Information flow:
        (T, D) @ (D, d_ff) + (d_ff,) = (T, d_ff)  [expand]
        -> ReLU -> (T, d_ff)
        -> (T, d_ff) @ (d_ff, D) + (D,) = (T, D)  [contract]
    """
    # 단계 1: 확장 (expand)
    # 입력을 D 차원에서 더 큰 d_ff 차원으로 확장
    h = x @ W1 + b1  # (T, d_ff)

    # 단계 2: 비선형 활성화 (ReLU)
    h = np.maximum(h, 0.0)  # (T, d_ff) - 음수는 0으로

    # 단계 3: 축소 (contract)
    # 다시 D 차원으로 돌아옴
    return h @ W2 + b2  # (T, D)


def forward(params: TransformerParams, token_ids: np.ndarray, *, n_heads: int, causal: bool = True) -> tuple[np.ndarray, list[np.ndarray]]:
    """Transformer Forward Pass를 수행합니다.

    Transformer의 전체 아키텍처:
    1. 토큰 임베딩 + 위치 임베딩으로 입력 인코딩
    2. N개의 동일한 블록을 순차 적용:
       - Layer Norm -> Multi-Head Attention -> Residual
       - Layer Norm -> Feed-Forward Network -> Residual
    3. 최종 Layer Norm
    4. 선형 투영으로 logits 생성

    핵심 설계 원칙:
    - Residual connections: x = x + f(x)
      * 깊은 네트워크에서 그래디언트 흐름 개선
      * 정체성 매핑(identity mapping) 비용 감소
    - Pre-norm: Layer Norm을 서브모듈 전에 적용
      * 입력을 정규화해 각 서브모듈이 안정적으로 학습
    - 위치 임베딩: 절대 위치 정보 추가
      * 토큰의 순서 정보를 인코딩

    Args:
        params: TransformerParams - 모든 학습 파라미터
        token_ids: shape (T,) - 토큰 ID 시퀀스
        n_heads: 멀티헤드 어텐션의 헤드 개수
        causal: True면 causal mask 적용 (자동회귀용)

    Returns:
        (logits, attn_weights) 튜플:
        - logits: shape (T, V) - 다음 토큰의 logit 점수
        - attn_weights: 각 레이어의 attention weights 리스트

    Information flow:
        token_ids (T,)
        -> token embedding + positional embedding (T, D)
        -> layer 0-n-1 각각:
            -> ln(x) (T, D)
            -> mha (T, D) + residual
            -> ln(x) (T, D)
            -> ffn (T, D) + residual
        -> ln(x) (T, D)
        -> linear projection to vocab (T, V)
    """
    # ===== 입력 검증 =====
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be 1D")
    T = len(token_ids)
    if T == 0:
        raise ValueError("token_ids must not be empty")
    if T > params.pos_emb.shape[0]:
        raise ValueError("Sequence longer than max_seq_len in params")

    # ===== 입력 인코딩 =====
    # 토큰 임베딩: 각 토큰 ID를 D차원 벡터로 변환
    # 위치 임베딩: 각 위치에 고정된 임베딩 추가
    # 이 두 가지를 더하면 "이 위치의 이 토큰"에 대한 정보 획득
    x = params.tok_emb[token_ids] + params.pos_emb[:T]  # (T, D)
    attn_weights: list[np.ndarray] = []

    # ===== Transformer 블록 스택 =====
    for layer in params.layers:
        # --- 어텐션 블록 ---
        # Layer Norm 적용 (pre-norm)
        x_ln = layer_norm(x, layer.ln1_g, layer.ln1_b)

        # 멀티헤드 어텐션
        attn_out, w = mha(x_ln, layer.Wq, layer.Wk, layer.Wv, layer.Wo, n_heads=n_heads, causal=causal)

        # Residual connection: x = x + attention(norm(x))
        x = x + attn_out
        attn_weights.append(w)

        # --- FFN 블록 ---
        # Layer Norm 적용 (pre-norm)
        x_ln2 = layer_norm(x, layer.ln2_g, layer.ln2_b)

        # Residual connection: x = x + ffn(norm(x))
        x = x + ffn(x_ln2, layer.W1, layer.b1, layer.W2, layer.b2)

    # ===== 최종 정규화 및 출력 투영 =====
    # 마지막 Layer Norm
    x = layer_norm(x, params.ln_f_g, params.ln_f_b)

    # 임베딩을 어휘 크기의 logits으로 변환
    logits = x @ params.W_out + params.b_out  # (T, V)

    return logits, attn_weights
