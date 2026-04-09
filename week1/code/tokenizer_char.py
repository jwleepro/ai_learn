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

    토크나이저의 역할:
    - 텍스트 <-> 정수 배열 간 변환
    - 텍스트를 모델이 처리할 수 있는 수치 형태로 변환

    설계:
    - 글자(character) 단위로 분할 (BPE, SentencePiece 같은 고급 방식 아님)
    - 어휘(vocab)는 텍스트에 나타난 모든 고유 글자
    - 각 글자는 0~(V-1) 사이의 ID로 매핑

    예: "안녕하세요"
    - 고유 글자: {안, 녕, 하, 세, 요}
    - 어휘 크기: 5
    - "안녕"을 인코딩하면: [안_id, 녕_id]
    """

    vocab: tuple[str, ...]

    def __post_init__(self) -> None:
        """초기화 검증: 어휘가 비어있지 않고 중복이 없는지 확인"""
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
        """
        텍스트에서 토크나이저 생성

        텍스트의 모든 고유 글자를 추출하여 어휘로 사용합니다.

        Args:
            text: 입력 텍스트
            sort_vocab: True면 어휘를 알파벳 순서로 정렬 (재현성 위해 기본값 True)

        Returns:
            CharTokenizer 인스턴스

        예: from_text("hello") -> vocab = ('e', 'h', 'l', 'o') (정렬됨)
        """
        if not text:
            raise ValueError("text must not be empty")
        unique_chars = set(text)
        vocab = sorted(unique_chars) if sort_vocab else tuple(unique_chars)
        return cls(tuple(vocab))

    def encode(self, text: str) -> list[int]:
        """
        텍스트를 토큰 ID의 리스트로 변환

        각 글자를 해당 ID로 매핑합니다.

        Args:
            text: 인코딩할 텍스트

        Returns:
            토큰 ID 리스트

        예: encode("안녕") -> [안_id, 녕_id]

        예외: 어휘에 없는 글자가 있으면 KeyError 발생
        """
        mapping = self.char_to_id
        ids: list[int] = []
        for ch in text:
            try:
                ids.append(mapping[ch])
            except KeyError as exc:
                raise KeyError(f"Unknown character {ch!r}. Rebuild vocab from data?") from exc
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        토큰 ID 리스트를 텍스트로 변환

        encode의 역함수입니다.

        Args:
            ids: 토큰 ID 리스트

        Returns:
            복원된 텍스트

        예: decode([안_id, 녕_id]) -> "안녕"
        """
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
