"""BPE encode/decode를 눈으로 확인하는 데모 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from bpe_tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BPE encode/decode 데모.")
    p.add_argument("--tokenizer", required=True, help="토크나이저 JSON 경로")
    p.add_argument("--text_file", required=True, help="인코딩할 텍스트 파일 경로")
    p.add_argument("--max_tokens", type=int, default=60, help="앞에서부터 N개 토큰 출력")
    return p.parse_args()


def main() -> None:
    """메인 데모 로직.

    목표:
    1. 학습된 BPE 토크나이저 로드
    2. 텍스트를 인코딩하고 토큰들 확인
    3. 디코딩해서 원본 텍스트 복원 가능함을 확인

    BPE 분석:
    - 문자 토크나이저보다 토큰 수가 적음 (더 큰 단위)
    - 자주 나타나는 부분 문자열들이 단일 토큰으로 나타남
    """
    args = parse_args()

    # 1. 학습된 토크나이저 로드
    tok = BPETokenizer.load_json(args.tokenizer)

    # 2. 텍스트 로드 및 인코딩
    text = Path(args.text_file).read_text(encoding="utf-8")

    # 토큰 시퀀스 (문자열 리스트)
    tokens = tok.encode_tokens(text)

    # 토큰 ID 시퀀스 (정수 리스트)
    ids = tok.encode(text)

    # 3. 결과 출력
    print(f"tokens={len(tokens)}  ids={len(ids)}  vocab_size={tok.vocab_size}")

    # 앞의 N개 토큰 출력
    shown = tokens[: int(args.max_tokens)]
    print("first tokens:")
    for i, t in enumerate(shown):
        print(f"  [{i:02d}] {t!r}")

    print("")
    print("decode (from ids):")
    # 첫 200개 ID를 다시 텍스트로 디코딩
    print(tok.decode(ids[: 200]))


if __name__ == "__main__":
    main()
