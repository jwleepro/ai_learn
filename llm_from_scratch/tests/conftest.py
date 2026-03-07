"""tests/ 에서 code/ 모듈을 import할 수 있도록 sys.path를 설정합니다."""

import sys
from pathlib import Path

_CODE_DIR = str(Path(__file__).resolve().parent.parent / "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
