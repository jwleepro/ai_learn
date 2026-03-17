"""tests/ 에서 code/ 모듈을 import할 수 있도록 sys.path를 설정합니다."""

import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent / "code"
for _sub in sorted(_CODE_DIR.iterdir()):
    if _sub.is_dir() and not _sub.name.startswith("_"):
        p = str(_sub)
        if p not in sys.path:
            sys.path.insert(0, p)
