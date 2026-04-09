"""글자(character) 단위 토크나이저.

- vocab: 등장한 글자 목록(중복 없음)
- encode: text -> list[int]
- decode: list[int] -> text

주의:
- 기본은 vocab을 정렬하여(stable) 같은 텍스트면 항상 같은 id가 나오게 합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class CharTokenizer:
    """가장 단순한 토크나이저(글자 단위).

    학습/생성 코드에서 공통으로 쓰는 최소 기능만 제공합니다.
    """

    vocab: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.vocab) == 0:
            raise ValueError("vocab must not be empty")
        if len(set(self.vocab)) != len(self.vocab):
            raise ValueError("vocab must not contain duplicates")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def char_to_id(self) -> dict[str, int]:
        return {ch: i for i, ch in enumerate(self.vocab)}

    @classmethod
    def from_text(cls, text: str, *, sort_vocab: bool = True) -> CharTokenizer:
        if not text:
            raise ValueError("text must not be empty")
        unique_chars = set(text)
        vocab = sorted(unique_chars) if sort_vocab else tuple(unique_chars)
        return cls(tuple(vocab))

    def encode(self, text: str) -> list[int]:
        mapping = self.char_to_id
        ids: list[int] = []
        for ch in text:
            try:
                ids.append(mapping[ch])
            except KeyError as exc:
                raise KeyError(f"Unknown character {ch!r}. Rebuild vocab from data?") from exc
        return ids

    def decode(self, ids: list[int]) -> str:
        out_chars: list[str] = []
        for token_id in ids:
            if not (0 <= token_id < self.vocab_size):
                raise ValueError(f"token_id out of range: {token_id}")
            out_chars.append(self.vocab[token_id])
        return "".join(out_chars)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        payload = {"type": "CharTokenizer", "vocab": list(self.vocab)}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> CharTokenizer:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("type") != "CharTokenizer":
            raise ValueError("Not a CharTokenizer json")
        vocab = payload.get("vocab")
        if not isinstance(vocab, list) or not all(isinstance(x, str) for x in vocab):
            raise ValueError("Invalid vocab in json")
        return cls(tuple(vocab))
