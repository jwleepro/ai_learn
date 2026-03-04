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
        if not text:
            raise ValueError("text must not be empty")
        if num_merges <= 0:
            raise ValueError("num_merges must be > 0")

        # Word vocabulary: map symbol-tuples to frequency.
        vocab: dict[tuple[str, ...], int] = {}
        for symbols in _text_to_word_symbols(text):
            vocab[symbols] = vocab.get(symbols, 0) + 1

        merges: list[tuple[str, str]] = []
        for _ in range(num_merges):
            pair_freq: dict[tuple[str, str], int] = {}
            for symbols, freq in vocab.items():
                for pair in _get_pairs(symbols):
                    pair_freq[pair] = pair_freq.get(pair, 0) + freq
            if not pair_freq:
                break

            best_pair = max(pair_freq.items(), key=lambda kv: kv[1])[0]
            merges.append(best_pair)

            new_vocab: dict[tuple[str, ...], int] = {}
            for symbols, freq in vocab.items():
                merged = _merge_symbols(symbols, best_pair)
                new_vocab[merged] = new_vocab.get(merged, 0) + freq
            vocab = new_vocab

        # Collect tokens from final vocab.
        tokens: set[str] = set()
        for symbols in vocab:
            tokens.update(symbols)
        # Sort for stable ids.
        id_to_token = tuple(sorted(tokens))
        token_to_id = {t: i for i, t in enumerate(id_to_token)}
        return cls(merges=tuple(merges), token_to_id=token_to_id, id_to_token=id_to_token)

    def _encode_word_to_tokens(self, word: str) -> tuple[str, ...]:
        symbols: tuple[str, ...] = tuple(list(word) + ["</w>"])
        ranks = self.ranks

        while True:
            pairs = _get_pairs(symbols)
            if not pairs:
                break
            best = None
            best_rank = 10**18
            for p in pairs:
                r = ranks.get(p)
                if r is not None and r < best_rank:
                    best = p
                    best_rank = r
            if best is None:
                break
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
