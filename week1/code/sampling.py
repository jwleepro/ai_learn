"""샘플링 유틸리티(temperature / top-k / top-p).

- `sample_from_logits`: logits(점수)에서 샘플링
- `sample_from_probs`: 확률분포에서 샘플링

주의: 확률(probs)에서 temperature를 적용할 때는
`p ** (1/temperature)` 형태로 “뾰족/평평”하게 만듭니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from softmax import softmax


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None


def _validate_sampling_cfg(cfg: SamplingConfig) -> None:
    if cfg.temperature <= 0:
        raise ValueError("temperature must be > 0")
    if cfg.top_k is not None and cfg.top_k <= 0:
        raise ValueError("top_k must be > 0")
    if cfg.top_p is not None and not (0.0 < cfg.top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")


def _apply_top_k(probs: np.ndarray, k: int) -> np.ndarray:
    """
    Top-K 샘플링: 확률이 높은 상위 k개 후보만 남기기

    개념:
    - "생성의 다양성"과 "품질" 사이 균형을 잡는 방법
    - 뾰족한 확률분포에서는 상위 k개로 충분하고 꼬리 확률은 노이즈
    - 예: 확률 [0.5, 0.3, 0.1, 0.05, 0.05]를 k=2로 적용
      -> 상위 2개 [0.5, 0.3]만 남기고 정규화
      -> [0.5/(0.5+0.3), 0.3/(0.5+0.3)] = [0.625, 0.375]

    Args:
        probs: 1D 확률분포 (합 = 1)
        k: 유지할 상위 후보 개수

    Returns:
        상위 k개만 유지한 정규화된 확률분포
    """
    if k >= len(probs):
        return probs
    # argpartition: 가장 큰 k개 요소의 인덱스를 (정렬 없이) 추출
    # -k:는 상위 k개를 의미
    idx = np.argpartition(probs, -k)[-k:]
    # 상위 k개 위치에만 True인 마스크 생성
    mask = np.zeros_like(probs, dtype=bool)
    mask[idx] = True
    # 마스크를 적용: 상위 k개만 유지, 나머지는 0
    out = np.where(mask, probs, 0.0)
    # 다시 정규화: 합 = 1로 만들기
    s = out.sum()
    return out / s if s > 0 else probs


def _apply_top_p(probs: np.ndarray, p: float) -> np.ndarray:
    """
    Nucleus(Top-P) 샘플링: 누적 확률이 p가 될 때까지 필요한 후보만 남기기

    Top-K와의 차이:
    - Top-K: 정확히 k개 후보 (고정)
    - Top-P: 동적으로 후보 수가 결정 (확률 합이 p가 될 때까지)

    개념:
    - "가장 가능성 높은 후보들이 전체 확률의 p를 설명하는가?"
    - 예: p=0.9, 확률 [0.5, 0.3, 0.15, 0.04, 0.01]
      - 누적합: [0.5, 0.8, 0.95, 0.99, 1.0]
      - 0.95에서 p=0.9를 넘으므로 상위 3개 [0.5, 0.3, 0.15]만 유지
      - 나머지 [0.04, 0.01]은 제거
      - 정규화: [0.5/0.95, 0.3/0.95, 0.15/0.95] ≈ [0.526, 0.316, 0.158]

    Args:
        probs: 1D 확률분포 (합 = 1)
        p: 누적 확률 임계값 (0 < p <= 1)

    Returns:
        누적 확률이 p를 설명하는 후보들만 유지한 정규화된 분포
    """
    if p >= 1.0:
        return probs

    # 확률을 내림차순으로 정렬
    order = np.argsort(probs)[::-1]
    sorted_probs = probs[order]

    # 누적합 계산: 큰 확률부터 차례로 더한 값
    # cumsum[i] = 상위 i+1개 확률의 합
    cumsum = np.cumsum(sorted_probs)

    # searchsorted: 누적합이 p를 처음 넘는 인덱스 찾기
    # side="left": p보다 작거나 같은 최대값의 인덱스
    cutoff = int(np.searchsorted(cumsum, p, side="left"))

    # 상위 cutoff+1개까지만 유지 (cumsum[cutoff] >= p가 되도록)
    keep_ids = order[: cutoff + 1]

    # 선택된 확률들만 유지, 나머지는 0
    out = np.zeros_like(probs)
    out[keep_ids] = probs[keep_ids]

    # 다시 정규화
    s = out.sum()
    return out / s if s > 0 else probs


def sample_from_logits(logits: np.ndarray, rng: np.random.Generator, *, cfg: SamplingConfig) -> int:
    """
    로짓(점수)에서 표본 추출

    로짓 -> 온도 조절 -> softmax -> top-k/p -> 샘플링

    Args:
        logits: 1D 배열, 각 후보의 점수 (어떤 범위든 상관없음)
        rng: numpy 난수 생성기
        cfg: 샘플링 설정 (temperature, top_k, top_p)

    Returns:
        선택된 후보의 인덱스
    """
    _validate_sampling_cfg(cfg)
    if logits.ndim != 1:
        raise ValueError("logits must be 1D")

    # 온도 조절: 낮은 온도 = 높은 점수가 더 두드러짐
    scaled = logits / float(cfg.temperature)
    # softmax로 확률분포 변환
    probs = softmax(scaled, axis=0)
    # top-k, top-p 필터링 (선택사항)
    if cfg.top_k is not None:
        probs = _apply_top_k(probs, int(cfg.top_k))
    if cfg.top_p is not None:
        probs = _apply_top_p(probs, float(cfg.top_p))
    # 확률에 따라 표본 추출
    return int(rng.choice(len(probs), p=probs))


def sample_from_probs(probs: np.ndarray, rng: np.random.Generator, *, cfg: SamplingConfig) -> int:
    """
    확률분포에서 표본 추출

    빅램 같은 이미 확률화된 분포에서 온도, top-k, top-p를 적용하여 샘플링

    Args:
        probs: 1D 배열, 확률값들 (음수 없음, 합이 양수)
        rng: numpy 난수 생성기
        cfg: 샘플링 설정 (temperature, top_k, top_p)

    Returns:
        선택된 후보의 인덱스
    """
    _validate_sampling_cfg(cfg)
    if probs.ndim != 1:
        raise ValueError("probs must be 1D")
    if np.any(probs < 0):
        raise ValueError("probs must be non-negative")
    s = float(probs.sum())
    if s <= 0:
        raise ValueError("probs must have positive sum")

    # 정규화 (혹시 합이 정확히 1이 아닐 수도 있으므로)
    p = probs / s

    # 온도 조절 (확률분포에 대해)
    # p^(1/T) 형태로 변환하여 분포를 조절
    if cfg.temperature != 1.0:
        p = p ** (1.0 / float(cfg.temperature))
        p = p / float(p.sum())

    # 필터링 (선택사항)
    if cfg.top_k is not None:
        p = _apply_top_k(p, int(cfg.top_k))
    if cfg.top_p is not None:
        p = _apply_top_p(p, float(cfg.top_p))

    # 최종 확률분포에서 표본 추출
    return int(rng.choice(len(p), p=p))
