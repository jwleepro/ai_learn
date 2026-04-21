"""Week 8: SFT JSONL 검증기 (초보용 단순 버전)."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--required", default="instruction,output")
    parser.add_argument("--expect_json", action="store_true")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"파일이 없습니다: {path}")
        return

    required_keys = [x.strip() for x in args.required.split(",")]
    lines = path.read_text(encoding="utf-8").splitlines()

    total = len(lines)
    ok = 0
    fail = 0

    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            print(f"{line_no}행: JSON 문법 오류")
            fail += 1
            continue

        missing = []
        for key in required_keys:
            if key not in item:
                missing.append(key)
        if missing:
            print(f"{line_no}행: 키 누락 {missing}")
            fail += 1
            continue

        empty = []
        for key in required_keys:
            if not str(item[key]).strip():
                empty.append(key)
        if empty:
            print(f"{line_no}행: 빈 값 {empty}")
            fail += 1
            continue

        if args.expect_json:
            try:
                json.loads(item["output"])
            except json.JSONDecodeError:
                print(f"{line_no}행: output이 JSON 문자열이 아님")
                fail += 1
                continue

        ok += 1

    print("검증 완료")
    print(f"총 라인: {total}")
    print(f"성공: {ok}")
    print(f"실패: {fail}")


if __name__ == "__main__":
    main()
