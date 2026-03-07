"""MiniGPT 학습 CLI(PyTorch).

char-level(글자 단위) 토큰화로 작은 GPT를 학습합니다.

처음 읽을 때 추천 순서:
1. `parse_args`
2. `get_batch`
3. `estimate_loss`
4. 맨 아래 학습 루프
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "PyTorch가 설치되어 있지 않습니다. "
            "`llm_from_scratch/requirements-torch.txt`를 참고해 torch를 설치하세요."
        ) from exc
    return torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MiniGPT 학습(PyTorch, 글자 단위).")
    p.add_argument("--input", required=True, help="입력 UTF-8 텍스트 파일 경로")
    p.add_argument("--out", default="llm_from_scratch/models/minigpt.pt", help="체크포인트 저장 경로(.pt)")
    p.add_argument("--block", type=int, default=128, help="block_size(T)")
    p.add_argument("--embd", type=int, default=128, help="임베딩/모델 차원(n_embd)")
    p.add_argument("--heads", type=int, default=4, help="헤드 수(n_head)")
    p.add_argument("--layers", type=int, default=4, help="레이어 수(n_layer)")
    p.add_argument("--dropout", type=float, default=0.1, help="dropout 확률")
    p.add_argument("--batch", type=int, default=64, help="배치 크기(batch size)")
    p.add_argument("--steps", type=int, default=2000, help="학습 스텝 수")
    p.add_argument("--lr", type=float, default=3e-4, help="학습률(learning rate)")
    p.add_argument("--seed", type=int, default=0, help="난수 시드(seed)")
    p.add_argument("--eval_every", type=int, default=200, help="평가 출력 주기(스텝 단위)")
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    return p.parse_args()


def main() -> None:
    torch = _require_torch()
    from minigpt import GPTConfig, MiniGPT

    args = parse_args()
    text = Path(args.input).read_text(encoding="utf-8")
    if not text:
        raise ValueError("Input text is empty")

    # 가장 단순한 글자 단위 vocab입니다.
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    _ = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    if len(train_data) < 2 or len(val_data) < 2:
        raise ValueError("Dataset is too short after train/val split; need at least 2 chars per split.")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    torch.manual_seed(int(args.seed))

    cfg = GPTConfig(
        vocab_size=len(vocab),
        block_size=int(args.block),
        n_layer=int(args.layers),
        n_head=int(args.heads),
        n_embd=int(args.embd),
        dropout=float(args.dropout),
    )
    model = MiniGPT(cfg).to(device_t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    def get_batch(split: str):
        src = train_data if split == "train" else val_data
        seq_len = min(cfg.block_size, len(src) - 1)
        if seq_len <= 0:
            raise ValueError(f"{split} split is too short for language-model training.")
        # 임의 시작점 여러 개를 뽑아,
        # x는 현재 구간, y는 한 칸 오른쪽으로 민 정답 구간으로 만듭니다.
        ix = torch.randint(0, len(src) - seq_len, (int(args.batch),))
        x = torch.stack([src[int(i) : int(i) + seq_len] for i in ix])
        y = torch.stack([src[int(i) + 1 : int(i) + seq_len + 1] for i in ix])
        return x.to(device_t), y.to(device_t)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = torch.zeros(20, device=device_t)
            for k in range(20):
                x, y = get_batch(split)
                _, loss = model(x, y)
                losses[k] = loss
            out[split] = float(losses.mean().item())
        model.train()
        return out

    t0 = time.time()
    for step in range(1, int(args.steps) + 1):
        if step % int(args.eval_every) == 0 or step == 1:
            losses = estimate_loss()
            dt = time.time() - t0
            print(f"step={step}  train_loss={losses['train']:.4f}  val_loss={losses['val']:.4f}  dt={dt:.1f}s")

        x, y = get_batch("train")
        logits, loss = model(x, y)
        _ = logits
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    ckpt = {
        "cfg": cfg.__dict__,
        "vocab": vocab,
        "state_dict": model.state_dict(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
