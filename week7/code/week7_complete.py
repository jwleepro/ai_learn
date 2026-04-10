"""Week 7 Complete: BPE(Byte Pair Encoding) Tokenizer.

이 파일은 BPE 토크나이저의 모든 과정(학습, 인코딩, 디코딩, 저장/로드)을
하나의 파일에서 순서대로 읽고 실행할 수 있도록 통합한 교육용 코드입니다.

주요 내용:
1. BPETokenizer: 빈도 기반의 서브워드(Subword) 토크나이징 알고리즘
2. Training: 가장 자주 발생하는 인접 쌍을 반복적으로 병합하여 어휘 사전(Vocab) 확장
3. Encoding/Decoding: 학습된 병합 규칙을 적용하여 텍스트를 토큰 ID로 변환 및 복원

실행 방법:
- 학습: python week7/code/week7_complete.py --train
- 데모: python week7/code/week7_complete.py --demo
"""

from __future__ import annotations

import argparse
import json
import collections
from dataclasses import dataclass
from pathlib import Path


# ============================================================================
# 1. BPE Tokenizer Implementation
# ============================================================================

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: dict[tuple[int, int], int]):
        self.vocab = vocab # id -> bytes
        self.merges = merges # (id, id) -> merged_id
        
    @classmethod
    def train(cls, text: str, vocab_size: int) -> BPETokenizer:
        """BPE 알고리즘을 사용하여 토크나이저를 학습합니다."""
        # 1. 초기 어휘 사전: 각 바이트(0-255)를 개별 토큰으로 설정
        tokens = list(text.encode("utf-8"))
        vocab = {i: bytes([i]) for i in range(256)}
        merges = {}
        
        num_merges = vocab_size - 256
        current_ids = list(tokens)
        
        for i in range(num_merges):
            # 가장 빈번한 인접 쌍 찾기
            stats = collections.Counter()
            for pair in zip(current_ids, current_ids[1:]):
                stats[pair] += 1
            
            if not stats:
                break
                
            top_pair = max(stats, key=stats.get)
            new_id = 256 + i
            
            # 병합 규칙 저장
            merges[top_pair] = new_id
            vocab[new_id] = vocab[top_pair[0]] + vocab[top_pair[1]]
            
            # 텍스트 내의 쌍을 새로운 ID로 교체
            new_ids = []
            skip = False
            for j in range(len(current_ids)):
                if skip:
                    skip = False
                    continue
                if j < len(current_ids) - 1 and (current_ids[j], current_ids[j+1]) == top_pair:
                    new_ids.append(new_id)
                    skip = True
                else:
                    new_ids.append(current_ids[j])
            current_ids = new_ids
            
            if (i + 1) % 50 == 0:
                print(f"Merge {i+1}/{num_merges}: {top_pair} -> {new_id} ({vocab[new_id].decode('utf-8', errors='replace')})")
                
        return cls(vocab, merges)

    def encode(self, text: str) -> list[int]:
        """학습된 병합 규칙을 사용하여 텍스트를 토큰 ID로 인코딩합니다."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            # 적용 가능한 병합 규칙 중 가장 먼저 학습된(순위가 높은) 규칙 찾기
            stats = {}
            for i, pair in enumerate(zip(ids, ids[1:])):
                if pair in self.merges:
                    # (순위, 위치) 저장
                    if pair not in stats:
                        stats[pair] = self.merges[pair]
            
            if not stats:
                break
                
            # 가장 순위가 높은(값이 작은) 병합 규칙 선택
            best_pair = min(stats, key=stats.get)
            new_id = self.merges[best_pair]
            
            new_ids = []
            skip = False
            for i in range(len(ids)):
                if skip:
                    skip = False
                    continue
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == best_pair:
                    new_ids.append(new_id)
                    skip = True
                else:
                    new_ids.append(ids[i])
            ids = new_ids
        return ids

    def decode(self, ids: list[int]) -> str:
        """토큰 ID들을 바이트 시퀀스로 결합한 후 텍스트로 디코딩합니다."""
        parts = [self.vocab[idx] for idx in ids]
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, path: str):
        # JSON 저장을 위해 튜플 키를 문자열로 변환
        data = {
            "vocab": {str(k): v.hex() for k, v in self.vocab.items()},
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> BPETokenizer:
        data = json.loads(Path(path).read_text())
        vocab = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
        merges = {}
        for k, v in data["merges"].items():
            p1, p2 = map(int, k.split(","))
            merges[(p1, p2)] = v
        return cls(vocab, merges)


# ============================================================================
# 2. Main Execution Flow
# ============================================================================

def main() -> None:
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
            print("모델 파일이 없습니다. 먼저 --train을 실행하세요.")
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
        for idx in ids:
            token_bytes = tokenizer.vocab[idx]
            print(f"ID {idx:3d}: {token_bytes.decode('utf-8', errors='replace')!r}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
