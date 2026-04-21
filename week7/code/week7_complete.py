"""Week 7: BPE 토크나이저 (초보용 단순 버전)."""

import argparse
import json
from pathlib import Path


def replace_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[tuple[int, int], int]]):
        self.vocab = vocab
        self.merges = merges

    @classmethod
    def train(cls, text: str, vocab_size: int):
        ids = list(text.encode("utf-8"))
        vocab = {i: bytes([i]) for i in range(256)}
        merges = []

        merge_count = max(0, vocab_size - 256)
        for i in range(merge_count):
            pair_count = {}
            for j in range(len(ids) - 1):
                pair = (ids[j], ids[j + 1])
                pair_count[pair] = pair_count.get(pair, 0) + 1

            if not pair_count:
                break

            best_pair = max(pair_count, key=pair_count.get)
            new_id = 256 + i
            vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
            merges.append((best_pair, new_id))
            ids = replace_pair(ids, best_pair, new_id)

            if (i + 1) % 50 == 0:
                piece = vocab[new_id].decode("utf-8", errors="replace")
                print(f"merge {i + 1}/{merge_count}: {best_pair} -> {new_id} ({piece!r})")

        return cls(vocab, merges)

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        for pair, new_id in self.merges:
            ids = replace_pair(ids, pair, new_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        b = b"".join(self.vocab[idx] for idx in ids)
        return b.decode("utf-8", errors="replace")

    def save(self, path: str):
        data = {
            "vocab": {str(k): v.hex() for k, v in self.vocab.items()},
            "merges": [[a, b, new_id] for (a, b), new_id in self.merges],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
        merges = []
        for a, b, new_id in data["merges"]:
            merges.append(((int(a), int(b)), int(new_id)))
        return cls(vocab, merges)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--data", default="week7/data/tiny_corpus_ko.txt")
    parser.add_argument("--vocab_size", type=int, default=500)
    parser.add_argument("--model_path", default="week7/code/bpe_tokenizer.json")
    args = parser.parse_args()

    if args.train:
        path = Path(args.data)
        if not path.exists():
            print(f"파일이 없습니다: {path}")
            return
        text = path.read_text(encoding="utf-8")
        tok = BPETokenizer.train(text, vocab_size=args.vocab_size)
        tok.save(args.model_path)
        print(f"학습 저장 완료: {args.model_path}")
        return

    if args.demo:
        if not Path(args.model_path).exists():
            print("모델 파일이 없습니다. 먼저 --train을 실행하세요.")
            return
        tok = BPETokenizer.load(args.model_path)
        sample = "안녕하세요, BPE 토크나이저 테스트입니다."
        ids = tok.encode(sample)
        back = tok.decode(ids)
        print(f"원문: {sample}")
        print(f"토큰 ID: {ids}")
        print(f"복원: {back}")
        print(f"토큰 수: {len(ids)} / 원본 바이트 수: {len(sample.encode('utf-8'))}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
