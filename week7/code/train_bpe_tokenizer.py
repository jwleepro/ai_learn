"""BPE 토크나이저를 학습하고 JSON으로 저장하는 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from bpe_tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="(학습용) 단순 BPE 토크나이저 학습.")
    p.add_argument("--input", required=True, help="입력 UTF-8 텍스트 파일 경로")
    p.add_argument("--out", default="llm_from_scratch/models/bpe_tokenizer.json", help="출력 JSON 경로")
    p.add_argument("--merges", type=int, default=200, help="merge 반복 횟수")
    return p.parse_args()


def main() -> None:
    """메인 학습 로직.

    처리 순서:
    1. 텍스트 파일 로드
    2. BPE 토크나이저 학습 (병합 횟수 지정)
    3. 학습된 토크나이저를 JSON으로 저장
    4. 학습 결과 출력 (어휘 크기, 병합 수)

    BPE의 장점:
    - 문자 기반 모델보다 효율적 (토큰이 더 큼)
    - 서브워드 토크나이저보다 단순 (공백 기반)
    - 훈련 데이터에 특화된 어휘 구성
    """
    args = parse_args()

    # 1. 텍스트 로드
    text = Path(args.input).read_text(encoding="utf-8")

    # 2. BPE 토크나이저 학습
    # num_merges: 병합 횟수가 많을수록 어휘 크기 증가
    tok = BPETokenizer.train(text, num_merges=int(args.merges))

    # 3. 학습된 토크나이저 저장
    # JSON 형식: merges와 token_to_id 매핑 포함
    tok.save_json(args.out)

    # 4. 학습 결과 출력
    print(f"saved={args.out}  vocab_size={tok.vocab_size}  merges={len(tok.merges)}")


if __name__ == "__main__":
    main()
