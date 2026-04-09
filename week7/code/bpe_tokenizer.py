"""학습용 단순 BPE(Byte Pair Encoding) 토크나이저.

이 구현은 교육 목적이라서 몇 가지를 단순화했습니다:
- 공백 기준으로 단어를 나누며, 공백/줄바꿈을 원문 그대로 복원하지는 않습니다.
- 단어 끝에 `</w>` 토큰을 붙여 “단어 경계”를 표현합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


def _get_pairs(symbols: tuple[str, ...]) -> set[tuple[str, str]]:
    return {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}


def _merge_symbols(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    a, b = pair
    merged: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
            merged.append(a + b)
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


def _text_to_word_symbols(text: str) -> list[tuple[str, ...]]:
    # Very simple whitespace tokenization; keeps punctuation attached to words.
    words = [w for w in text.split() if w]
    out: list[tuple[str, ...]] = []
    for w in words:
        out.append(tuple(list(w) + ["</w>"]))
    return out


@dataclass(frozen=True)
class BPETokenizer:
    """BPE 토크나이저(간단 버전)."""

    merges: tuple[tuple[str, str], ...]
    token_to_id: dict[str, int]
    id_to_token: tuple[str, ...]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def ranks(self) -> dict[tuple[str, str], int]:
        return {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def train(cls, text: str, *, num_merges: int = 200) -> BPETokenizer:
        """BPE(Byte Pair Encoding) 토크나이저를 학습합니다.

        BPE 알고리즘:
        1. 초기 어휘를 모든 고유 문자로 설정
        2. num_merges번 반복:
           a. 텍스트에서 가장 자주 나타나는 인접 토큰 쌍(pair) 찾기
           b. 그 쌍을 하나의 토큰으로 합치기(merge)
        3. 학습된 병합 순서를 저장해 인코딩/디코딩에 사용

        핵심 아이디어:
        - 자주 나타나는 부분 문자열을 새로운 토큰으로 점진적으로 생성
        - 훈련 데이터에 특화된 어휘 구성
        - 좋은 압축률과 언어 모델 성능의 균형

        예:
        - 초기: ['l', 'o', 'w', ' ', 'w', 'i', 'd', 'e', ' ', 'l', 'o', 'w']
        - merge 1: 'l' + 'o' = 'lo'
        - merge 2: 'lo' + 'w' = 'low'
        - merge 3: ' ' + 'w' = ' w'
        - ... (계속)

        Args:
            text: 학습할 원본 텍스트
            num_merges: 수행할 병합 횟수 (더 많을수록 어휘 크기 증가)

        Returns:
            학습된 BPETokenizer 객체

        Complexity:
        - 시간: O(num_merges * vocab_size * avg_symbols_per_word)
        - 공간: O(vocab_size)
        """
        if not text:
            raise ValueError("text must not be empty")
        if num_merges <= 0:
            raise ValueError("num_merges must be > 0")

        # ===== Step 1: 초기 어휘 구성 (단위: 문자) =====
        # 각 단어를 문자 리스트 + "</w>" 토큰으로 분할
        # vocab: {(문자들...): 출현 빈도}
        vocab: dict[tuple[str, ...], int] = {}
        for symbols in _text_to_word_symbols(text):
            vocab[symbols] = vocab.get(symbols, 0) + 1
        # 예: {('l', 'o', 'w', '</w>'): 5, ('w', 'i', 'd', 'e', '</w>'): 3, ...}

        # ===== Step 2: 병합 반복 =====
        merges: list[tuple[str, str]] = []
        for _ in range(num_merges):
            # Step 2a: 모든 인접 쌍의 빈도 계산
            pair_freq: dict[tuple[str, str], int] = {}
            for symbols, freq in vocab.items():
                # 각 단어의 인접 쌍을 찾고 빈도 누적
                for pair in _get_pairs(symbols):
                    pair_freq[pair] = pair_freq.get(pair, 0) + freq

            if not pair_freq:
                # 더 이상 쌍이 없으면 종료
                break

            # Step 2b: 가장 자주 나타나는 쌍 선택
            best_pair = max(pair_freq.items(), key=lambda kv: kv[1])[0]
            merges.append(best_pair)

            # Step 2c: 선택된 쌍을 모든 단어에서 병합
            new_vocab: dict[tuple[str, ...], int] = {}
            for symbols, freq in vocab.items():
                # 이 쌍의 모든 출현을 하나의 토큰으로 병합
                merged = _merge_symbols(symbols, best_pair)
                new_vocab[merged] = new_vocab.get(merged, 0) + freq
            vocab = new_vocab

        # ===== Step 3: 최종 어휘에서 토큰 ID 매핑 생성 =====
        # 최종 vocab에서 나타나는 모든 고유 토큰 수집
        tokens: set[str] = set()
        for symbols in vocab:
            tokens.update(symbols)

        # 안정적인 ID 할당을 위해 정렬
        id_to_token = tuple(sorted(tokens))
        token_to_id = {t: i for i, t in enumerate(id_to_token)}

        return cls(merges=tuple(merges), token_to_id=token_to_id, id_to_token=id_to_token)

    def _encode_word_to_tokens(self, word: str) -> tuple[str, ...]:
        """단일 단어를 BPE 토큰 시퀀스로 인코딩합니다.

        인코딩 알고리즘:
        1. 단어를 문자 리스트로 분할하고 "</w>" 추가 (단어 끝 표시)
        2. 학습 중 발견된 병합들을 훈련 순서에 따라 그리디하게 적용
           - 각 단계에서 가장 먼저 학습된 (가장 낮은 rank의) 쌍을 병합
        3. 더 이상 병합할 수 있는 쌍이 없을 때까지 반복

        예시:
        - word = "hello"
        - 초기: ('h', 'e', 'l', 'l', 'o', '</w>')
        - 병합 'he': ('he', 'l', 'l', 'o', '</w>')
        - 병합 'll': ('he', 'll', 'o', '</w>')
        - 병합 'o</w>': ('he', 'll', 'o</w>')
        - 결과: ('he', 'll', 'o</w>')

        Greedy 병합:
        - 각 단계에서 가장 먼저 학습된 쌍을 선택
        - 최적이 아닐 수도 있지만, 높은 성능과 빠른 인코딩 제공

        Args:
            word: 인코딩할 단어

        Returns:
            BPE 토큰들의 튜플
        """
        # 단어를 초기 심볼(문자) 시퀀스로 변환
        symbols: tuple[str, ...] = tuple(list(word) + ["</w>"])
        # 병합 순서별 rank 맵 (먼저 학습될수록 작은 rank)
        ranks = self.ranks

        # Greedy 병합: 더 이상 병합할 수 없을 때까지 반복
        while True:
            # 현재 심볼 시퀀스의 모든 인접 쌍 찾기
            pairs = _get_pairs(symbols)
            if not pairs:
                # 쌍이 없으면 종료
                break

            # 이 쌍들 중 rank가 가장 낮은 것(가장 먼저 학습된 것) 선택
            best = None
            best_rank = 10**18
            for p in pairs:
                r = ranks.get(p)  # 이 쌍이 학습 중 어디서 나타났는지 확인
                if r is not None and r < best_rank:
                    best = p
                    best_rank = r

            if best is None:
                # 학습된 쌍이 없으면 종료
                break

            # 선택된 쌍을 병합
            symbols = _merge_symbols(symbols, best)

        return symbols

    def encode_tokens(self, text: str) -> list[str]:
        out: list[str] = []
        for word in [w for w in text.split() if w]:
            out.extend(self._encode_word_to_tokens(word))
        return out

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for tok in self.encode_tokens(text):
            if tok not in self.token_to_id:
                raise KeyError(f"Unknown token {tok!r}. Train tokenizer on larger data?")
            ids.append(self.token_to_id[tok])
        return ids

    def decode_tokens(self, tokens: list[str]) -> str:
        pieces: list[str] = []
        for tok in tokens:
            if tok.endswith("</w>"):
                pieces.append(tok[: -len("</w>")])
                pieces.append(" ")
            else:
                pieces.append(tok)
        return "".join(pieces).rstrip()

    def decode(self, ids: list[int]) -> str:
        tokens: list[str] = []
        for token_id in ids:
            if not (0 <= token_id < self.vocab_size):
                raise ValueError(f"token_id out of range: {token_id}")
            tokens.append(self.id_to_token[token_id])
        return self.decode_tokens(tokens)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "type": "BPETokenizer",
            "merges": [list(p) for p in self.merges],
            "id_to_token": list(self.id_to_token),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> BPETokenizer:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("type") != "BPETokenizer":
            raise ValueError("Not a BPETokenizer json")

        merges_raw = payload.get("merges")
        if not isinstance(merges_raw, list):
            raise ValueError("Invalid merges")
        merges: list[tuple[str, str]] = []
        for item in merges_raw:
            if not (isinstance(item, list) and len(item) == 2 and all(isinstance(x, str) for x in item)):
                raise ValueError("Invalid merge pair")
            merges.append((item[0], item[1]))

        id_to_token_raw = payload.get("id_to_token")
        if not isinstance(id_to_token_raw, list) or not all(isinstance(x, str) for x in id_to_token_raw):
            raise ValueError("Invalid id_to_token")
        id_to_token = tuple(id_to_token_raw)
        token_to_id = {t: i for i, t in enumerate(id_to_token)}
        return cls(merges=tuple(merges), token_to_id=token_to_id, id_to_token=id_to_token)
