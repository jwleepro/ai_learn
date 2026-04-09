"""신경망 빅램 모델을 학습하고 체크포인트(.npz)로 저장하는 CLI.

주요 역할:
- 입력 텍스트를 토큰화
- 이전 토큰 -> 다음 토큰 매핑 생성
- 학습/검증 데이터 분할
- 신경망 빅램 모델 훈련
- 최종 가중치와 토크나이저를 .npz 형식으로 저장

데이터 구조:
- prev_ids: 이전 토큰 ID 배열 (shape: N)
- next_ids: 다음 토큰 ID 배열 (shape: N) - 정답/라벨
- W: 학습된 가중치 행렬 (shape: vocab_size x vocab_size)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _path_setup  # noqa: F401  (code/ 하위 폴더 sys.path 등록)
from bigram_nn import BigramNNConfig, train_bigram_nn
from model_io import BigramNNCheckpoint, save_bigram_nn
from tokenizer_char import CharTokenizer


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱합니다.

    반환값:
        argparse.Namespace: --input, --out, --epochs, --lr, --batch, --seed, --val_frac
    """
    p = argparse.ArgumentParser(description="신경망 빅램 LM 학습(numpy).")
    p.add_argument("--input", required=True, help="입력 UTF-8 텍스트 파일 경로")
    p.add_argument("--out", default="llm_from_scratch/models/bigram_nn.npz", help="체크포인트 저장 경로(.npz)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2.0, help="학습률(learning rate)")
    p.add_argument("--batch", type=int, default=2048, help="배치 크기(batch size)")
    p.add_argument("--seed", type=int, default=0, help="난수 시드(seed)")
    p.add_argument("--val_frac", type=float, default=0.1, help="검증 데이터 비율(0~0.5)")
    return p.parse_args()


def main() -> None:
    """메인 훈련 로직.

    처리 순서:
    1. 텍스트 파일 읽기
    2. 문자 토크나이저 생성 (사전학습된 토크나이저 없으므로 텍스트에서 직접 생성)
    3. 텍스트를 토큰 ID 배열로 변환
    4. (prev_id, next_id) 쌍 생성 (빅램)
    5. 데이터를 훈련/검증으로 분할
    6. 신경망 빅램 모델 훈련
    7. 체크포인트 저장 (.npz 파일에 토크나이저와 가중치 포함)
    """
    args = parse_args()

    # 텍스트 파일 읽기
    text = Path(args.input).read_text(encoding="utf-8")
    if not text:
        raise ValueError("Input text is empty")

    # 토크나이저 생성: 텍스트의 고유한 문자들을 어휘로 사용
    tok = CharTokenizer.from_text(text)

    # 텍스트를 정수 토큰 ID로 변환 (각 문자 -> ID)
    ids = np.array(tok.encode(text), dtype=np.int64)

    # 빅램 데이터셋 생성: (이전 토큰, 다음 토큰) 쌍
    # 예: [x0, x1, x2, x3] -> prev=[x0, x1, x2], next=[x1, x2, x3]
    prev_ids = ids[:-1]  # 마지막 토큰 제외
    next_ids = ids[1:]   # 첫 토큰 제외

    # 훈련/검증 데이터 분할
    if not (0.0 <= args.val_frac < 0.5):
        raise ValueError("--val_frac must be in [0, 0.5)")
    split = int(len(prev_ids) * (1.0 - args.val_frac))
    prev_train, prev_val = prev_ids[:split], prev_ids[split:]
    next_train, next_val = next_ids[:split], next_ids[split:]

    # 훈련 설정 객체 생성
    config = BigramNNConfig(
        lr=float(args.lr),
        epochs=int(args.epochs),
        batch_size=int(args.batch),
        seed=int(args.seed),
    )

    # 신경망 빅램 모델 훈련
    # 반환: W (학습된 가중치), history (각 에포크의 손실값)
    W, history = train_bigram_nn(
        prev_train,
        next_train,
        tok.vocab_size,
        config=config,
        prev_val=prev_val,
        next_val=next_val,
    )

    # 체크포인트 저장: 토크나이저 + 가중치 + 시작 토큰 ID
    save_bigram_nn(args.out, BigramNNCheckpoint(tokenizer=tok, W=W, default_start_id=int(ids[0])))

    # 최종 손실값 출력
    last = history[-1]
    if "val_loss" in last:
        print(f"saved={args.out}  train_loss={last['train_loss']:.4f}  val_loss={last['val_loss']:.4f}")
    else:
        print(f"saved={args.out}  train_loss={last['train_loss']:.4f}")


if __name__ == "__main__":
    main()
