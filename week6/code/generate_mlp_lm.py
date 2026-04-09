"""학습된 MLP LM 체크포인트(.npz)로 텍스트를 생성하는 CLI."""

from __future__ import annotations

import argparse

import numpy as np

import _path_setup  # noqa: F401  (code/ 하위 폴더 sys.path 등록)
from mlp_lm import next_token_probs
from model_io import load_mlp_lm
from sampling import SamplingConfig, sample_from_probs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLP LM으로 텍스트 생성(numpy).")
    p.add_argument("--model", default="llm_from_scratch/models/mlp_lm.npz", help="체크포인트 경로(.npz)")
    p.add_argument("--length", type=int, default=400, help="생성할 글자 수")
    p.add_argument("--seed", type=int, default=0, help="난수 시드(seed)")
    p.add_argument("--temperature", type=float, default=1.0, help="샘플링 온도(>0)")
    p.add_argument("--top_k", type=int, default=None, help="top-k 샘플링(선택)")
    p.add_argument("--top_p", type=float, default=None, help="top-p 샘플링(선택)")
    p.add_argument(
        "--start_ids",
        type=str,
        default="",
        help='시작 컨텍스트를 토큰 id로 직접 지정(쉼표 구분). 예: "1,2,3,4"',
    )
    return p.parse_args()


def main() -> None:
    """메인 생성 로직.

    처리 순서:
    1. 학습된 체크포인트 로드 (파라미터 + 토크나이저)
    2. 초기 컨텍스트 결정 (지정 또는 기본값)
    3. 자동회귀 루프 (autoregressive generation):
       - 현재 컨텍스트를 모델에 통과시켜 다음 토큰의 확률분포 계산
       - 확률분포에서 샘플링해 다음 토큰 결정
       - 컨텍스트를 한 칸 오른쪽으로 이동하고 새로운 토큰 추가
       - 이를 반복해 긴 텍스트 생성
    4. 생성된 토큰 ID들을 문자로 디코딩하여 출력

    자동회귀의 개념:
    - "다음 토큰은 이전 컨텍스트에만 의존한다"는 조건부 독립성 가정
    - 이렇게 한 번에 한 토큰씩 생성하면 장기 의존성을 포착 가능
    """
    args = parse_args()

    # 1. 체크포인트 로드: 파라미터 + 토크나이저 + 설정
    ckpt = load_mlp_lm(args.model)

    # 난수 생성기와 샘플링 설정
    rng = np.random.default_rng(args.seed)
    cfg = SamplingConfig(
        temperature=float(args.temperature),
        top_k=args.top_k,
        top_p=args.top_p
    )

    # 2. 초기 컨텍스트 결정
    if args.start_ids:
        # 사용자가 명시적으로 지정한 경우
        start_ids = [int(x.strip()) for x in args.start_ids.split(",") if x.strip() != ""]
        if len(start_ids) != ckpt.context_len:
            raise ValueError(f"--start_ids must have exactly {ckpt.context_len} ids")
        if not all(0 <= token_id < ckpt.tokenizer.vocab_size for token_id in start_ids):
            raise ValueError("--start_ids contains out-of-range token id")
        context = np.array(start_ids, dtype=np.int64)
    else:
        # 기본값: 훈련 텍스트의 첫 context_len개 토큰
        context = ckpt.default_start_ids.copy()
    initial_context = context.copy()

    # 3. 자동회귀 생성 루프
    out_ids: list[int] = []
    for _ in range(args.length):
        # 현재 컨텍스트에서 다음 토큰의 확률분포 계산
        # temperature는 샘플러에서 처리됨
        probs = next_token_probs(ckpt.params, context, temperature=1.0)

        # 확률분포에서 샘플링 (temperature/top-k/top-p 등 적용)
        next_id = sample_from_probs(probs, rng, cfg=cfg)

        # 생성된 토큰 저장
        out_ids.append(next_id)

        # 컨텍스트 슬라이딩: [x0, x1, x2, ..., xn-1] -> [x1, x2, ..., xn-1, next_id]
        # np.roll: 배열을 한 칸 왼쪽으로 회전, 맨 앞 원소가 맨 뒤로 이동
        context = np.roll(context, -1)
        context[-1] = next_id

    # 4. 생성된 토큰 ID들을 문자로 디코딩하여 출력
    print(ckpt.tokenizer.decode(initial_context.tolist()) + ckpt.tokenizer.decode(out_ids))


if __name__ == "__main__":
    main()
