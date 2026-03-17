"""tests/ 에서 code/ 모듈을 import할 수 있도록 sys.path를 설정합니다."""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

_CODE_DIR = _TESTS_DIR.parent / "code"
for _sub in sorted(_CODE_DIR.iterdir()):
    if _sub.is_dir() and not _sub.name.startswith("_"):
        p = str(_sub)
        if p not in sys.path:
            sys.path.insert(0, p)
