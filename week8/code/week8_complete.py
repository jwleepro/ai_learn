"""Week 8 Complete: SFT(Supervised Fine-Tuning) JSONL Validator.

SFT 데이터셋(JSONL) 형식 검증기.

JSONL 이란:
- 한 줄에 JSON 객체 한 개씩 들어있는 텍스트 파일.
- 예) {"instruction": "안녕?", "output": "안녕하세요!"}

검증 항목:
1) JSON 문법이 올바른지
2) 필수 키(instruction, output 등) 가 모두 있는지
3) 그 키의 값이 비어있지 않은지
4) (선택) output 자체가 다시 JSON 형식이어야 하는 경우 (구조화된 출력 학습용)

실행 방법:
- 검증: python week8/code/week8_complete.py --input week8/data/sft_toy.jsonl
"""

import argparse
import json
from pathlib import Path


def parse_required_keys(required_arg):
    """쉼표로 구분된 문자열을 키 리스트로 변환.

    예) "instruction,output" -> ["instruction", "output"]
    """
    keys = []
    for part in required_arg.split(","):
        key = part.strip()  # 양 끝 공백 제거
        keys.append(key)
    return keys


def find_missing_keys(data, required_keys):
    """data 에서 빠진 필수 키들을 리스트로 반환."""
    missing = []
    for key in required_keys:
        if key not in data:
            missing.append(key)
    return missing


def find_empty_keys(data, required_keys):
    """값이 빈 문자열(공백만 있어도 비어있는 것으로 본다) 인 키들을 리스트로 반환."""
    empty = []
    for key in required_keys:
        value = str(data[key])
        if len(value.strip()) == 0:
            empty.append(key)
    return empty


def is_valid_json(text):
    """text 가 JSON 으로 파싱 가능한지 여부."""
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def main():
    parser = argparse.ArgumentParser(description="SFT JSONL 데이터셋 검증기")
    parser.add_argument("--input", required=True, help="입력 JSONL 파일 경로")
    parser.add_argument("--required", default="instruction,output", help="필수 키 (쉼표 구분)")
    parser.add_argument("--expect_json", action="store_true", help="output이 JSON 형식이어야 함")
    args = parser.parse_args()

    data_path = Path(args.input)

    # 데이터 파일이 없을 때: 교육용 샘플은 자동 생성
    if not data_path.exists():
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

    required_keys = parse_required_keys(args.required)

    file_text = data_path.read_text(encoding="utf-8")
    lines = file_text.strip().splitlines()

    print(f"--- SFT 데이터 검증 시작 (파일: {data_path.name}) ---")
    print(f"필수 키: {required_keys}")

    success_count = 0
    error_count = 0

    line_number = 0
    for line in lines:
        line_number += 1

        # 1) JSON 문법 검사
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print(f"라인 {line_number} 에러: 유효하지 않은 JSON 문법")
            error_count += 1
            continue

        # 2) 필수 키 존재 여부
        missing = find_missing_keys(data, required_keys)
        if len(missing) > 0:
            print(f"라인 {line_number} 에러: 누락된 키 {missing}")
            error_count += 1
            continue

        # 3) 빈 값 검사
        empty = find_empty_keys(data, required_keys)
        if len(empty) > 0:
            print(f"라인 {line_number} 에러: 빈 값 {empty}")
            error_count += 1
            continue

        # 4) (선택) output 이 JSON 형식인지 확인
        if args.expect_json:
            if not is_valid_json(data["output"]):
                print(f"라인 {line_number} 에러: output이 JSON 형식이 아님")
                error_count += 1
                continue

        success_count += 1

    print("\n--- 검증 완료 ---")
    print(f"총 라인: {len(lines)}")
    print(f"성공: {success_count}")
    print(f"실패: {error_count}")


if __name__ == "__main__":
    main()
