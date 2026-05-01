"""Week 7 Complete: BPE(Byte Pair Encoding) Tokenizer.

BPE 알고리즘 한 줄 요약:
"가장 자주 같이 나오는 두 토큰을 하나로 합치기" 를 vocab_size 가 될 때까지 반복.

학습 과정:
1) 텍스트를 UTF-8 바이트로 바꾼다 (각 바이트 0-255 가 초기 토큰).
2) 인접한 (앞, 뒤) 쌍의 등장 횟수를 센다.
3) 가장 많이 나온 쌍을 새 ID 로 만들고, 텍스트에서 그 쌍을 모두 새 ID 로 치환.
4) 2-3 을 (목표 vocab 크기 - 256) 번 반복.

인코딩: 학습 때 만든 병합 규칙들을 가능한 빨리 학습된 순서대로 적용.
디코딩: ID -> 바이트 -> UTF-8 텍스트.

실행 방법:
- 학습: python week7/code/week7_complete.py --train
- 데모: python week7/code/week7_complete.py --demo
"""

import argparse
import json
from pathlib import Path


# ============================================================================
# 1. BPE Tokenizer
# ============================================================================

class BPETokenizer:
    """BPE 토크나이저.

    - vocab:  ID -> bytes (예: 256 -> b'th')
    - merges: (id1, id2) -> new_id (학습 순서대로 매겨진 새 ID)
    """

    def __init__(self, vocab, merges):
        self.vocab = vocab    # dict: int -> bytes
        self.merges = merges  # dict: (int, int) -> int

    @staticmethod
    def train(text, vocab_size):
        """텍스트로 BPE 토크나이저를 학습한다."""
        # 1) 초기 토큰: UTF-8 바이트 0..255
        tokens = list(text.encode("utf-8"))

        vocab = {}
        for i in range(256):
            vocab[i] = bytes([i])

        merges = {}
        num_merges = vocab_size - 256

        for step in range(num_merges):
            # 2) 인접 쌍 빈도수 세기
            pair_counts = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in pair_counts:
                    pair_counts[pair] += 1
                else:
                    pair_counts[pair] = 1

            # 더 셀 쌍이 없으면 종료
            if len(pair_counts) == 0:
                break

            # 가장 자주 나온 쌍 찾기
            top_pair = None
            top_count = -1
            for pair in pair_counts:
                if pair_counts[pair] > top_count:
                    top_count = pair_counts[pair]
                    top_pair = pair

            # 3) 새 ID 부여 + vocab 갱신
            new_id = 256 + step
            merges[top_pair] = new_id
            vocab[new_id] = vocab[top_pair[0]] + vocab[top_pair[1]]

            # 4) tokens 안의 top_pair 를 모두 new_id 로 치환
            new_tokens = []
            i = 0
            while i < len(tokens):
                # 현재 위치에서 top_pair 가 시작되는지 확인
                if i + 1 < len(tokens) and tokens[i] == top_pair[0] and tokens[i + 1] == top_pair[1]:
                    new_tokens.append(new_id)
                    i = i + 2  # 두 칸 점프 (쌍을 합쳤으니)
                else:
                    new_tokens.append(tokens[i])
                    i = i + 1
            tokens = new_tokens

            if (step + 1) % 50 == 0:
                merged_str = vocab[new_id].decode("utf-8", errors="replace")
                print(f"Merge {step + 1}/{num_merges}: {top_pair} -> {new_id} ({merged_str})")

        return BPETokenizer(vocab, merges)

    def encode(self, text):
        """텍스트 -> 토큰 ID 리스트.

        학습한 병합 규칙을 학습된 순서대로 가능한 모두 적용한다.
        """
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            # 적용 가능한 병합 규칙 중 가장 먼저 학습된(=ID 값이 작은) 규칙 선택
            best_pair = None
            best_rank = None  # 작은 값일수록 먼저 학습된 것

            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                if pair in self.merges:
                    rank = self.merges[pair]
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            # 더 적용할 규칙이 없으면 종료
            if best_pair is None:
                break

            new_id = self.merges[best_pair]

            # ids 안의 best_pair 를 모두 new_id 로 치환
            new_ids = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and ids[i] == best_pair[0] and ids[i + 1] == best_pair[1]:
                    new_ids.append(new_id)
                    i = i + 2
                else:
                    new_ids.append(ids[i])
                    i = i + 1
            ids = new_ids

        return ids

    def decode(self, ids):
        """토큰 ID 리스트 -> 텍스트.

        각 ID 의 바이트를 이어붙인 뒤 UTF-8 로 디코딩.
        """
        all_bytes = b""
        for token_id in ids:
            all_bytes = all_bytes + self.vocab[token_id]
        return all_bytes.decode("utf-8", errors="replace")

    def save(self, path):
        """JSON 파일로 저장.

        JSON 은 튜플 키를 직접 못 쓰므로:
        - bytes 값은 hex 문자열로 변환 (예: b'th' -> '7468')
        - (id1, id2) 키는 "id1,id2" 문자열로 변환
        """
        vocab_str = {}
        for token_id in self.vocab:
            vocab_str[str(token_id)] = self.vocab[token_id].hex()

        merges_str = {}
        for pair in self.merges:
            key = f"{pair[0]},{pair[1]}"
            merges_str[key] = self.merges[pair]

        data = {"vocab": vocab_str, "merges": merges_str}
        Path(path).write_text(json.dumps(data, indent=2))

    @staticmethod
    def load(path):
        data = json.loads(Path(path).read_text())

        vocab = {}
        for key in data["vocab"]:
            vocab[int(key)] = bytes.fromhex(data["vocab"][key])

        merges = {}
        for key in data["merges"]:
            parts = key.split(",")
            id1 = int(parts[0])
            id2 = int(parts[1])
            merges[(id1, id2)] = data["merges"][key]

        return BPETokenizer(vocab, merges)


# ============================================================================
# 2. Main Execution Flow
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="학습 실행")
    parser.add_argument("--demo", action="store_true", help="데모 실행")
    parser.add_argument("--data", default="week7/data/tiny_corpus_ko.txt", help="데이터 경로")
    parser.add_argument("--vocab_size", type=int, default=500, help="목표 어휘 사전 크기")
    parser.add_argument("--model_path", default="week7/code/bpe_tokenizer.json", help="모델 저장 경로")
    args = parser.parse_args()

    if args.train:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"데이터 파일이 없습니다: {data_path}")
            return
        text = data_path.read_text(encoding="utf-8")

        print(f"--- BPE 학습 시작 (Target Vocab Size: {args.vocab_size}) ---")
        tokenizer = BPETokenizer.train(text, args.vocab_size)
        tokenizer.save(args.model_path)
        print(f"학습 완료 및 저장: {args.model_path}")

    elif args.demo:
        if not Path(args.model_path).exists():
            print("모델 파일이 없습니다. 먼저 --train 을 실행하세요.")
            return

        tokenizer = BPETokenizer.load(args.model_path)
        test_text = "안녕하세요, BPE 토크나이저 테스트입니다. 딥러닝은 재미있어요!"

        ids = tokenizer.encode(test_text)
        decoded = tokenizer.decode(ids)

        print(f"원본 텍스트: {test_text}")
        print(f"인코딩 IDs: {ids}")
        print(f"토큰 수: {len(ids)} (바이트 수: {len(test_text.encode('utf-8'))})")
        print(f"디코딩 결과: {decoded}")

        # 개별 토큰 확인
        print("\n[개별 토큰 분석]")
        for token_id in ids:
            token_bytes = tokenizer.vocab[token_id]
            token_str = token_bytes.decode("utf-8", errors="replace")
            print(f"ID {token_id:3d}: {repr(token_str)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
