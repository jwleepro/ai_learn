"""MLP LM을 학습하고 체크포인트(.npz)로 저장하는 CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _path_setup  # noqa: F401  (code/ 하위 폴더 sys.path 등록)
from dataset_lm import make_context_dataset
from mlp_lm import MLPLMConfig, train_mlp_lm
from model_io import MLPLMCheckpoint, save_mlp_lm
from tokenizer_char import CharTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="미니 MLP 언어모델 학습(numpy).")
    p.add_argument("--input", required=True, help="입력 UTF-8 텍스트 파일 경로")
    p.add_argument("--out", default="llm_from_scratch/models/mlp_lm.npz", help="체크포인트 저장 경로(.npz)")
    p.add_argument("--context", type=int, default=8, help="컨텍스트 길이(k)")
    p.add_argument("--embed", type=int, default=32, help="임베딩 차원(D)")
    p.add_argument("--hidden", type=int, default=128, help="은닉 차원(H)")
    p.add_argument("--epochs", type=int, default=60, help="epoch 수")
    p.add_argument("--lr", type=float, default=0.2, help="학습률(learning rate)")
    p.add_argument("--batch", type=int, default=256, help="배치 크기(batch size)")
    p.add_argument("--seed", type=int, default=0, help="난수 시드(seed)")
    p.add_argument("--val_frac", type=float, default=0.1, help="검증 데이터 비율(0~0.5)")
    return p.parse_args()


def main() -> None:
    """메인 훈련 로직.

    처리 순서:
    1. 텍스트 파일 읽기
    2. 문자 토크나이저 생성 (사전학습 없이 텍스트의 고유 문자들을 어휘로 사용)
    3. 텍스트를 토큰 ID 배열로 변환
    4. 컨텍스트 데이터셋 생성: (컨텍스트, 정답) 쌍
       - 예: 컨텍스트 길이가 8이면 [x0~x7] -> x8, [x1~x8] -> x9, ...
    5. 데이터를 훈련/검증으로 분할
    6. MLP LM 모델 훈련 (SGD)
    7. 체크포인트 저장 (.npz 파일에 파라미터와 토크나이저 포함)
    """
    args = parse_args()

    # 1. 텍스트 파일 읽기
    text = Path(args.input).read_text(encoding="utf-8")
    if not text:
        raise ValueError("Input text is empty")

    # 2. 토크나이저 생성: 텍스트의 고유한 문자들을 어휘로 사용
    tok = CharTokenizer.from_text(text)
    # 3. 텍스트를 정수 토큰 ID로 변환 (각 문자 -> ID)
    ids = np.array(tok.encode(text), dtype=np.int64)

    # 4. 컨텍스트 데이터셋 생성
    # X: (N, C) - N개의 컨텍스트 (각각 context_len개 토큰)
    # y: (N,) - N개의 정답 토큰
    X, y = make_context_dataset(ids, int(args.context))

    # 5. 훈련/검증 데이터 분할
    if not (0.0 <= args.val_frac < 0.5):
        raise ValueError("--val_frac must be in [0, 0.5)")
    split = int(len(X) * (1.0 - args.val_frac))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # 훈련 설정 객체 생성
    config = MLPLMConfig(
        context_len=int(args.context),       # 컨텍스트 길이 (과거 몇 개 토큰을 봐야 하는가)
        embed_dim=int(args.embed),            # 임베딩 차원 (각 토큰을 몇 차원으로 표현)
        hidden_dim=int(args.hidden),          # 은닉층 차원 (내부 표현의 복잡도)
        lr=float(args.lr),                    # 학습률 (파라미터 업데이트 속도)
        epochs=int(args.epochs),              # 에포크 수 (데이터를 몇 번 반복)
        batch_size=int(args.batch),           # 배치 크기 (한 번에 처리할 샘플 수)
        seed=int(args.seed),                  # 난수 시드 (재현성)
    )

    # 6. MLP LM 모델 훈련
    params, history = train_mlp_lm(
        X_train, y_train, tok.vocab_size,
        config=config,
        X_val=X_val, y_val=y_val
    )

    # 7. 체크포인트 저장
    # 생성 시에 사용할 초기 컨텍스트 (훈련 텍스트의 첫 context_len개 토큰)
    default_start_ids = ids[: config.context_len]
    save_mlp_lm(
        args.out,
        MLPLMCheckpoint(
            tokenizer=tok,
            context_len=config.context_len,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            params=params,
            default_start_ids=default_start_ids,
        ),
    )

    # 최종 손실값 출력
    last = history[-1]
    if "val_loss" in last:
        print(f"saved={args.out}  train_loss={last['train_loss']:.4f}  val_loss={last['val_loss']:.4f}")
    else:
        print(f"saved={args.out}  train_loss={last['train_loss']:.4f}")


if __name__ == "__main__":
    main()
