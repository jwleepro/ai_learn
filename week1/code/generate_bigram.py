"""카운트 기반 빅램(글자 단위)으로 텍스트 생성하는 CLI.

학습(카운트/확률표 만들기)과 생성을 한 번에 수행합니다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _path_setup  # noqa: F401  (code/ 하위 폴더 sys.path 등록)
from bigram_counts import build_bigram_counts, counts_to_probs
from sampling import SamplingConfig, sample_from_probs
from tokenizer_char import CharTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="카운트 기반 빅램 언어모델로 텍스트 생성(글자 단위).")
    p.add_argument("--input", required=True, help="입력 UTF-8 텍스트 파일 경로")
    p.add_argument("--length", type=int, default=400, help="생성할 글자 수")
    p.add_argument("--seed", type=int, default=0, help="난수 시드(seed)")
    p.add_argument("--start", type=str, default="", help="시작 텍스트(마지막 글자를 컨텍스트로 사용)")
    p.add_argument("--smoothing", type=float, default=0.0, help="Add-k smoothing (0=끄기)")
    p.add_argument("--temperature", type=float, default=1.0, help="샘플링 온도(>0)")
    p.add_argument("--top_k", type=int, default=None, help="top-k 샘플링(k개 후보만 유지; 선택)")
    p.add_argument("--top_p", type=float, default=None, help="top-p(nucleus) 샘플링(0<p<=1; 선택)")
    return p.parse_args()


def main() -> None:
    """
    빅램 언어모델로 텍스트 생성하는 주요 흐름

    단계:
    1. 입력 텍스트 로드
    2. 문자 토크나이저 구축 (어휘 = 텍스트에 등장한 모든 문자)
    3. 텍스트를 토큰 ID로 인코딩
    4. 빅램 등장 횟수 계산
    5. 확률분포 변환
    6. 시작 토큰부터 시작하여 한 글자씩 샘플링으로 생성
    """
    args = parse_args()

    # 1단계: 입력 텍스트 로드
    text_path = Path(args.input)
    text = text_path.read_text(encoding="utf-8")
    if not text:
        raise ValueError("Input text is empty")

    # 2단계: 토크나이저 구축 (문자 단위)
    # 이 텍스트에 등장한 고유한 모든 문자를 어휘로 사용
    tokenizer = CharTokenizer.from_text(text)

    # 3단계: 텍스트 인코딩
    # "안녕하세요" -> [an_id, nyeong_id, ha_id, se_id, yo_id] 같은 식
    token_ids = np.array(tokenizer.encode(text), dtype=np.int64)

    # 4단계: 빅램 등장 횟수 계산
    # counts[i, j] = "토큰i 다음에 토큰j가 나온 횟수"
    counts = build_bigram_counts(token_ids, tokenizer.vocab_size)

    # 5단계: 확률분포로 변환
    # probs[i, j] = P(next=j | prev=i)
    # smoothing: 학습 데이터에 없던 전이를 가능하게 함 (0=끄기, 1=Laplace)
    probs = counts_to_probs(counts, smoothing=args.smoothing)

    # 6단계: 생성 시작 준비
    # 시작 텍스트의 마지막 글자를 초기 context로 사용
    start_text = args.start if args.start else text[:1]
    start_ids = tokenizer.encode(start_text)
    prev_id = start_ids[-1]  # 마지막 글자의 토큰 ID

    # 난수 생성기 설정 (재현 가능하도록 seed 고정)
    rng = np.random.default_rng(args.seed)

    # 샘플링 옵션 설정
    # - temperature: 확률 분포의 날카로움/부드러움 조절
    # - top_k, top_p: 샘플링할 후보를 상위 k개/누적확률 p로 제한
    cfg = SamplingConfig(temperature=float(args.temperature), top_k=args.top_k, top_p=args.top_p)

    # 7단계: 글자 생성 반복
    out_ids: list[int] = []
    for _ in range(args.length):
        # 이전 토큰의 확률분포에서 다음 토큰 샘플링
        next_id = sample_from_probs(probs[prev_id], rng, cfg=cfg)
        out_ids.append(next_id)
        # 다음 반복을 위해 현재 토큰을 이전 토큰으로 업데이트
        prev_id = next_id

    # 결과 출력: 시작 텍스트 + 생성된 텍스트
    print(start_text + tokenizer.decode(out_ids))


if __name__ == "__main__":
    main()
