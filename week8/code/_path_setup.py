"""code/ 하위 모든 폴더를 sys.path에 추가 — cross-folder import 지원."""
import sys
from pathlib import Path

_code = Path(__file__).resolve().parent.parent
for _d in sorted(_code.iterdir()):
    if _d.is_dir() and not _d.name.startswith("_"):
        _p = str(_d)
        if _p not in sys.path:
            sys.path.insert(0, _p)
