"""Week 7 Complete: BPE Tokenizer.

이 통합 파일은 bpe_tokenizer.py, train_bpe_tokenizer.py, demo_bpe.py의
모든 코드를 포함합니다.

BPE(Byte Pair Encoding) 토크나이저의 학습, 인코딩, 디코딩을 지원합니다.

BPE 알고리즘:
1. 초기 어휘를 모든 고유 문자로 설정
2. num_merges번 반복:
   a. 텍스트에서 가장 자주 나타나는 인접 토큰 쌍(pair) 찾기
   b. 그 쌍을 하나의 토큰으로 합치기(merge)
3. 학습된 병합 순서를 저장해 인코딩/디코딩에 사용
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# ============================================================================
# Section 1: bpe_tokenizer.py - BPE Core Implementation
# ============================================================================

def _get_pairs(symbols: tuple[str, ...]) -> set[tuple[str, str]]:
    """인접한 토큰 쌍을 찾습니다."""
    return {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}


def _merge_symbols(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    """주어진 쌍을 모든 위치에서 병합합니다."""
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
    """간단한 공백 토큰화를 사용하여 단어를 심볼 리스트로 변환합니다."""
    words = [w for w in text.split() if w]
    out: list[tuple[str, ...]] = []
    for w in words:
        out.append(tuple(list(w) + ["</w>"]))
    return out


class BPETokenizer:
    """BPE 토크나이저(간단 버전)."""

    def __init__(self, merges: tuple[tuple[str, str], ...], token_to_id: dict[str, int], id_to_token: tuple[str, ...]):
        self.merges = merges
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def ranks(self) -> dict[tuple[str, str], int]:
        return {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def train(cls, text: str, *, num_merges: int = 200) -> BPETokenizer:
        """BPE 토크나이저를 학습합니다."""
        if not text:
            raise ValueError("text must not be empty")
        if num_merges <= 0:
            raise ValueError("num_merges must be > 0")

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

        tokens: set[str] = set()
        for symbols in vocab:
            tokens.update(symbols)

        id_to_token = tuple(sorted(tokens))
        token_to_id = {t: i for i, t in enumerate(id_to_token)}

        return cls(merges=tuple(merges), token_to_id=token_to_id, id_to_token=id_to_token)

    def _encode_word_to_tokens(self, word: str) -> tuple[str, ...]:
        """단일 단어를 BPE 토큰 시퀀스로 인코딩합니다."""
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
        """텍스트를 BPE 토큰 리스트로 인코딩합니다."""
        out: list[str] = []
        for word in [w for w in text.split() if w]:
            out.extend(self._encode_word_to_tokens(word))
        return out

    def encode(self, text: str) -> list[int]:
        """텍스트를 토큰 ID 리스트로 인코딩합니다."""
        ids: list[int] = []
        for tok in self.encode_tokens(text):
            if tok not in self.token_to_id:
                raise KeyError(f"Unknown token {tok!r}. Train tokenizer on larger data?")
            ids.append(self.token_to_id[tok])
        return ids

    def decode_tokens(self, tokens: list[str]) -> str:
        """토큰 리스트를 텍스트로 디코딩합니다."""
        pieces: list[str] = []
        for tok in tokens:
            if tok.endswith("</w>"):
                pieces.append(tok[: -len("</w>")])
                pieces.append(" ")
            else:
                pieces.append(tok)
        return "".join(pieces).rstrip()

    def decode(self, ids: list[int]) -> str:
        """토큰 ID 리스트를 텍스트로 디코딩합니다."""
        tokens: list[str] = []
        for token_id in ids:
            if not (0 <= token_id < self.vocab_size):
                raise ValueError(f"token_id out of range: {token_id}")
            tokens.append(self.id_to_token[token_id])
        return self.decode_tokens(tokens)

    def save_json(self, path: str | Path) -> None:
        """토크나이저를 JSON으로 저장합니다."""
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
        """JSON에서 토크나이저를 로드합니다."""
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


# ============================================================================
# Section 2: train_bpe_tokenizer.py - Training script
# ============================================================================

def train_main(args: argparse.Namespace) -> None:
    """BPE 토크나이저 학습.

    처리 순서:
    1. 텍스트 파일 로드
    2. BPE 토크나이저 학습 (병합 횟수 지정)
    3. 학습된 토크나이저를 JSON으로 저장
    4. 학습 결과 출력 (어휘 크기, 병합 수)

    BPE의 장점:
    - 문자 기반 모델보다 효율적 (토큰이 더 큼)
    - 서브워드 토크나이저보다 단순 (공백 기반)
    - 훈련 데이터에 특화된 어휘 구성
    """
    text = Path(args.input).read_text(encoding="utf-8")
    tok = BPETokenizer.train(text, num_merges=int(args.merges))
    tok.save_json(args.out)

    print(f"saved={args.out}  vocab_size={tok.vocab_size}  merges={len(tok.merges)}")


# ============================================================================
# Section 3: demo_bpe.py - Demo script
# ============================================================================

def demo_main(args: argparse.Namespace) -> None:
    """BPE encode/decode 데모.

    목표:
    1. 학습된 BPE 토크나이저 로드
    2. 텍스트를 인코딩하고 토큰들 확인
    3. 디코딩해서 원본 텍스트 복원 가능함을 확인

    BPE 분석:
    - 문자 토크나이저보다 토큰 수가 적음 (더 큰 단위)
    - 자주 나타나는 부분 문자열들이 단일 토큰으로 나타남
    """
    tok = BPETokenizer.load_json(args.tokenizer)

    text = Path(args.text_file).read_text(encoding="utf-8")

    tokens = tok.encode_tokens(text)
    ids = tok.encode(text)

    print(f"tokens={len(tokens)}  ids={len(ids)}  vocab_size={tok.vocab_size}")

    shown = tokens[: int(args.max_tokens)]
    print("first tokens:")
    for i, t in enumerate(shown):
        print(f"  [{i:02d}] {t!r}")

    print("")
    print("decode (from ids):")
    print(tok.decode(ids[: 200]))


# ============================================================================
# Main entry point
# ============================================================================

def parse_args() -> argparse.Namespace:
    """메인 커맨드라인 인자를 파싱합니다."""
    p = argparse.ArgumentParser(description="Week 7: BPE 토크나이저 (학습 및 데모).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train 서브커맨드
    p_train = sub.add_parser("train", help="BPE 토크나이저 학습")
    p_train.add_argument("--input", required=True, help="입력 UTF-8 텍스트 파일 경로")
    p_train.add_argument("--out", default="llm_from_scratch/models/bpe_tokenizer.json", help="출력 JSON 경로")
    p_train.add_argument("--merges", type=int, default=200, help="merge 반복 횟수")

    # demo 서브커맨드
    p_demo = sub.add_parser("demo", help="BPE encode/decode 데모")
    p_demo.add_argument("--tokenizer", required=True, help="토크나이저 JSON 경로")
    p_demo.add_argument("--text_file", required=True, help="인코딩할 텍스트 파일 경로")
    p_demo.add_argument("--max_tokens", type=int, default=60, help="앞에서부터 N개 토큰 출력")

    return p.parse_args()


def main() -> None:
    """메인 통합 CLI 인터페이스."""
    args = parse_args()

    if args.cmd == "train":
        train_main(args)
    elif args.cmd == "demo":
        demo_main(args)
    else:
        raise AssertionError("unreachable")


if __name__ == "__main__":
    main()
