"""학습된 신경망 빅램 체크포인트(.npz)로 텍스트를 생성하는 CLI.

주요 역할:
- 저장된 가중치 행렬 W와 토크나이저 로드
- 자동 회귀 생성: 이전 토큰 -> 다음 토큰 확률 분포 -> 샘플
- 반복하여 긴 텍스트 생성

생성 알고리즘:
1. 시작 토큰 ID로 초기화
2. 루프: N번 반복
   a. W[prev_id]에서 logits 가져오기 -> softmax로 확률 분포
   b. 확률에서 샘플링 (temperature/top-k/top-p 적용)
   c. 샘플된 토큰을 다음 prev_id로 설정
3. 생성된 ID들을 문자로 디코딩
"""

from __future__ import annotations

import argparse

import numpy as np

import _path_setup  # noqa: F401  (code/ 하위 폴더 sys.path 등록)
from bigram_nn import bigram_probs
from model_io import load_bigram_nn
from sampling import SamplingConfig, sample_from_probs


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱합니다.

    반환값:
        argparse.Namespace: --model, --length, --seed, --start_id, --temperature, --top_k, --top_p
    """
    p = argparse.ArgumentParser(description="신경망 빅램 LM으로 텍스트 생성(numpy).")
    p.add_argument("--model", default="llm_from_scratch/models/bigram_nn.npz", help="체크포인트 경로(.npz)")
    p.add_argument("--length", type=int, default=400, help="생성할 글자 수")
    p.add_argument("--seed", type=int, default=0, help="난수 시드(seed)")
    p.add_argument("--start_id", type=int, default=None, help="시작 토큰 id(콘솔 인코딩 이슈 회피용)")
    p.add_argument("--temperature", type=float, default=1.0, help="샘플링 온도(>0)")
    p.add_argument("--top_k", type=int, default=None, help="top-k 샘플링(선택)")
    p.add_argument("--top_p", type=float, default=None, help="top-p 샘플링(선택)")
    return p.parse_args()


def main() -> None:
    """메인 생성 로직.

    처리 순서:
    1. 학습된 체크포인트 로드 (가중치 W + 토크나이저)
    2. 시작 토큰 결정 (지정 또는 기본값)
    3. 자동회귀 루프: 이전 토큰 -> 다음 토큰 샘플링 반복
    4. 생성된 토큰 ID들을 문자로 디코딩하여 출력
    """
    args = parse_args()

    # 체크포인트 로드: 가중치 W와 토크나이저 객체 포함
    ckpt = load_bigram_nn(args.model)

    # 난수 생성기와 샘플링 설정
    rng = np.random.default_rng(args.seed)
    cfg = SamplingConfig(temperature=float(args.temperature), top_k=args.top_k, top_p=args.top_p)

    # 시작 토큰 결정
    if args.start_id is None:
        # 기본값: 훈련 텍스트의 첫 문자
        prev_id = int(ckpt.default_start_id)
        start_text = ckpt.tokenizer.vocab[prev_id]
    else:
        # 명시적 지정
        prev_id = int(args.start_id)
        if not (0 <= prev_id < ckpt.tokenizer.vocab_size):
            raise ValueError("--start_id out of range for vocab")
        start_text = ckpt.tokenizer.vocab[prev_id]

    # 자동회귀 생성 루프
    out_ids: list[int] = []
    for _ in range(args.length):
        # 이전 토큰에서 다음 토큰의 확률 분포 계산
        # bigram_probs는 W[prev_id]를 softmax로 변환
        probs = bigram_probs(ckpt.W, prev_id, temperature=1.0)  # temperature은 샘플러에서 처리

        # 확률 분포에서 샘플링 (top-k/top-p 등 적용)
        next_id = sample_from_probs(probs, rng, cfg=cfg)

        # 생성된 토큰 저장 및 다음 스텝의 입력으로 사용
        out_ids.append(next_id)
        prev_id = next_id

    # 생성된 토큰 ID들을 문자로 디코딩하여 출력
    print(start_text + ckpt.tokenizer.decode(out_ids))


if __name__ == "__main__":
    main()
