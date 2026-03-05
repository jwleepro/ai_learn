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
    if isinstance(value, (dict, list)):
        return True, ""
    if not isinstance(value, str):
        return False, f"output must be str/dict/list for JSON, got {type(value).__name__}"
    if not _looks_like_json_text(value):
        return False, "output does not look like JSON text (expected '{' or '[')"
    try:
        json.loads(value)
    except json.JSONDecodeError as exc:
        return False, f"output JSON parse error: {exc.msg} (pos {exc.pos})"
    return True, ""


def main() -> None:
    args = parse_args()
    required = [k.strip() for k in str(args.required).split(",") if k.strip()]
    if not required:
        raise SystemExit("required keys is empty")

    path = Path(args.input)
    lines = path.read_text(encoding="utf-8").splitlines()

    total = 0
    ok = 0
    errors: list[str] = []

    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {i}: JSON parse error: {exc.msg} (pos {exc.pos})")
            continue

        if not isinstance(obj, dict):
            errors.append(f"line {i}: expected JSON object(dict), got {type(obj).__name__}")
            continue

        missing = [k for k in required if k not in obj]
        if missing:
            errors.append(f"line {i}: missing keys: {', '.join(missing)}")
            continue

        # Light sanity checks: required values should be non-empty strings.
        # If --expect_output_json is on, output may also be dict/list.
        bad = []
        for k in required:
            v = obj.get(k)
            if bool(args.expect_output_json) and k == "output":
                if isinstance(v, str):
                    if not v.strip():
                        bad.append(k)
                elif not isinstance(v, (dict, list)):
                    bad.append(k)
            else:
                if not isinstance(v, str) or not v.strip():
                    bad.append(k)
        if bad:
            errors.append(f"line {i}: invalid required values: {', '.join(bad)}")
            continue

        if bool(args.expect_output_json):
            ok_json, why = _parse_output_json(obj.get("output"))
            if not ok_json:
                errors.append(f"line {i}: {why}")
                continue

        ok += 1

    print(f"file={path}  total={total}  ok={ok}  errors={len(errors)}")
    for msg in errors[: int(args.max_errors)]:
        print(f"ERROR: {msg}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
