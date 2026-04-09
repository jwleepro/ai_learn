"""SFT(지도 파인튜닝)용 JSONL 데이터셋을 빠르게 검증하는 CLI.

- 각 줄이 JSON으로 파싱되는지
- 필수 키가 존재하는지
- (선택) output이 JSON 문자열이라면 파싱 가능한지

표준 라이브러리만 사용합니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT JSONL 데이터셋 검증기(가벼운 포맷 검사).")
    p.add_argument("--input", required=True, help="입력 JSONL 파일 경로")
    p.add_argument(
        "--required",
        default="instruction,output",
        help="필수 키(콤마로 구분). 기본: instruction,output",
    )
    p.add_argument(
        "--expect_output_json",
        action="store_true",
        help="output 값이 JSON 문자열(또는 JSON 오브젝트)이어야 함(구조화 추출 태스크용).",
    )
    p.add_argument("--max_errors", type=int, default=20, help="최대 에러 출력 개수")
    return p.parse_args()


def _looks_like_json_text(s: str) -> bool:
    s = s.strip()
    return s.startswith("{") or s.startswith("[")


def _parse_output_json(value: Any) -> tuple[bool, str]:
    """output 필드가 유효한 JSON인지 검증합니다.

    구조화 추출(structured extraction) 작업에서는 모델의 출력이
    JSON 형식이어야 합니다. 이 함수는 그를 검증합니다.

    지원하는 형식:
    - JSON 문자열: '{"key": "value"}'
    - JSON 오브젝트: dict 또는 list (Python 객체)

    Args:
        value: output 필드의 값

    Returns:
        (is_valid, error_message) 튜플
        - is_valid: True면 유효한 JSON, False면 검증 실패
        - error_message: 실패 시 오류 메시지, 성공 시 빈 문자열
    """
    # dict나 list는 이미 파이썬 객체로 파싱된 것 (유효)
    if isinstance(value, (dict, list)):
        return True, ""

    # 문자열이 아니면 실패
    if not isinstance(value, str):
        return False, f"output must be str/dict/list for JSON, got {type(value).__name__}"

    # JSON 문자열처럼 보이는지 확인 ('{' 또는 '[' 시작)
    if not _looks_like_json_text(value):
        return False, "output does not look like JSON text (expected '{' or '[')"

    # 실제로 파싱 가능한지 확인
    try:
        json.loads(value)
    except json.JSONDecodeError as exc:
        return False, f"output JSON parse error: {exc.msg} (pos {exc.pos})"

    return True, ""


def main() -> None:
    """메인 검증 로직.

    SFT(Supervised Fine-Tuning) 데이터셋은 JSONL 형식이어야 합니다.
    각 줄은 {필수_키: 값, ...} 형태의 JSON 오브젝트입니다.

    검증 항목:
    1. JSON 형식 유효성
    2. 필수 키 존재 확인
    3. 필수 값이 비어있지 않은지 확인
    4. (선택) output이 JSON 형식인지 확인
    """
    args = parse_args()

    # 필수 키 목록 파싱
    required = [k.strip() for k in str(args.required).split(",") if k.strip()]
    if not required:
        raise SystemExit("required keys is empty")

    path = Path(args.input)
    lines = path.read_text(encoding="utf-8").splitlines()

    total = 0      # 총 (비어있지 않은) 줄 수
    ok = 0         # 검증 성공한 줄 수
    errors: list[str] = []  # 에러 메시지들

    # ===== 각 줄 검증 =====
    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            # 빈 줄은 무시
            continue

        total += 1

        # Step 1: JSON 파싱
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {i}: JSON parse error: {exc.msg} (pos {exc.pos})")
            continue

        # Step 2: JSON 오브젝트 확인 (dict 타입)
        if not isinstance(obj, dict):
            errors.append(f"line {i}: expected JSON object(dict), got {type(obj).__name__}")
            continue

        # Step 3: 필수 키 존재 확인
        missing = [k for k in required if k not in obj]
        if missing:
            errors.append(f"line {i}: missing keys: {', '.join(missing)}")
            continue

        # Step 4: 필수 값의 유효성 검증
        # - 보통은 비어있지 않은 문자열
        # - expect_output_json이 켜져있으면 output은 str/dict/list 모두 가능
        bad = []
        for k in required:
            v = obj.get(k)
            if bool(args.expect_output_json) and k == "output":
                # output이 JSON 형식이어야 함
                if isinstance(v, str):
                    if not v.strip():
                        bad.append(k)
                elif not isinstance(v, (dict, list)):
                    bad.append(k)
            else:
                # 다른 필수 키: 비어있지 않은 문자열
                if not isinstance(v, str) or not v.strip():
                    bad.append(k)

        if bad:
            errors.append(f"line {i}: invalid required values: {', '.join(bad)}")
            continue

        # Step 5: (선택) output JSON 형식 검증
        if bool(args.expect_output_json):
            ok_json, why = _parse_output_json(obj.get("output"))
            if not ok_json:
                errors.append(f"line {i}: {why}")
                continue

        ok += 1

    # ===== 결과 출력 =====
    print(f"file={path}  total={total}  ok={ok}  errors={len(errors)}")

    # 에러 메시지 출력 (최대 max_errors개)
    for msg in errors[: int(args.max_errors)]:
        print(f"ERROR: {msg}")

    # 에러가 있으면 종료 코드 1로 반환
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
