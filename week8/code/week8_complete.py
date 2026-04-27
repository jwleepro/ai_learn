"""Week 8 Complete: SFT(Supervised Fine-Tuning) JSONL Validator.

이 파일은 SFT 학습을 위해 준비된 JSONL 데이터셋의 형식을 검증하는 과정을
하나의 파일에서 순서대로 읽고 실행할 수 있도록 통합한 교육용 코드입니다.

주요 내용:
1. JSONL Format: 각 줄이 독립된 JSON 객체로 구성된 데이터 형식
2. Validation: 필수 키(instruction, output 등) 존재 여부 및 값의 유효성 검사
3. JSON Output Check: 모델의 출력이 JSON 형식이어야 하는 경우(구조화된 추출)에 대한 특수 검증

실행 방법:
- 검증: python week8/code/week8_complete.py --input week8/data/sft_toy.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# ============================================================================
# 1. Main Execution Flow
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="SFT JSONL 데이터셋 검증기")
    parser.add_argument("--input", required=True, help="입력 JSONL 파일 경로")
    parser.add_argument("--required", default="instruction,output", help="필수 키 (쉼표 구분)")
    parser.add_argument("--expect_json", action="store_true", help="output이 JSON 형식이어야 함")
    args = parser.parse_args()

    # 데이터 로드
    data_path = Path(args.input)
    if not data_path.exists():
        # 교육용 샘플 데이터가 없을 경우를 위해 더미 생성 (실제 환경에서는 데이터가 있어야 함)
        if "sft_toy.jsonl" in str(data_path):
            print("샘플 데이터를 생성합니다...")
            data_path.parent.mkdir(parents=True, exist_ok=True)
            sample_content = (
                '{"instruction": "안녕?", "output": "안녕하세요!"}\n'
                '{"instruction": "날씨 어때?", "output": "좋아요."}\n'
            )
            data_path.write_text(sample_content, encoding="utf-8")
        else:
            print(f"파일이 없습니다: {data_path}")
            return

    # 리스트 컴프리헨션 [식 for x in 시퀀스]. 자바/C# 에는 없는 문법.
    # str.split(",") 으로 자른 뒤 각 토막에 strip() 으로 공백 제거.
    required_keys = [k.strip() for k in args.required.split(",")]
    lines = data_path.read_text(encoding="utf-8").strip().splitlines()

    # f-string: 중괄호 안에 표현식을 넣어 보간. C# 의 $"..." 와 비슷.
    print(f"--- SFT 데이터 검증 시작 (파일: {data_path.name}) ---")
    print(f"필수 키: {required_keys}")

    success_count = 0
    error_count = 0

    # enumerate(시퀀스, start=1) : (인덱스, 값) 쌍 순회. start=1 은 1부터 세겠다는 키워드 인자.
    # `for i, line in ...` 는 튜플 언패킹.
    for i, line in enumerate(lines, start=1):
        try:
            # 1. JSON 문법 검사
            data = json.loads(line)
            
            # 2. 필수 키 존재 여부 검사
            # `[k for k in ... if 조건]` 형태의 필터링 컴프리헨션 (자바 stream filter, C# LINQ Where 와 같은 의미).
            # `k not in data` 는 dict 멤버십 검사 — 자바 `containsKey`, C# `ContainsKey` 에 해당.
            missing = [k for k in required_keys if k not in data]
            if missing:
                print(f"라인 {i} 에러: 누락된 키 {missing}")
                error_count += 1
                continue

            # 3. 값의 유효성 검사 (비어있는 문자열 등)
            empty = [k for k in required_keys if not str(data[k]).strip()]
            if empty:
                print(f"라인 {i} 에러: 빈 값 {empty}")
                error_count += 1
                continue
                
            # 4. (선택) output이 JSON 형식인지 검사
            if args.expect_json:
                try:
                    json.loads(data["output"])
                except json.JSONDecodeError:
                    print(f"라인 {i} 에러: output이 JSON 형식이 아님")
                    error_count += 1
                    continue
            
            success_count += 1
            
        except json.JSONDecodeError:
            print(f"라인 {i} 에러: 유효하지 않은 JSON 문법")
            error_count += 1

    print(f"\n--- 검증 완료 ---")
    print(f"총 라인: {len(lines)}")
    print(f"성공: {success_count}")
    print(f"실패: {error_count}")


if __name__ == "__main__":
    main()
