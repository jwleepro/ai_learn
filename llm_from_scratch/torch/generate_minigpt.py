"""학습된 MiniGPT 체크포인트로 텍스트를 생성하는 CLI(PyTorch)."""

from __future__ import annotations

import argparse
from pathlib import Path


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
    p = argparse.ArgumentParser(description="MiniGPT로 텍스트 생성(PyTorch).")
    p.add_argument("--model", default="llm_from_scratch/models/minigpt.pt", help="체크포인트 경로(.pt)")
    p.add_argument("--length", type=int, default=400, help="생성할 글자 수")
    p.add_argument("--seed", type=int, default=0, help="난수 시드(seed)")
    p.add_argument("--temperature", type=float, default=1.0, help="샘플링 온도(>0)")
    p.add_argument("--top_k", type=int, default=None, help="top-k 샘플링(선택)")
    p.add_argument("--top_p", type=float, default=None, help="top-p 샘플링(선택)")
    p.add_argument("--prompt_file", type=str, default="", help="프롬프트 텍스트 파일(선택)")
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    return p.parse_args()


def main() -> None:
    torch = _require_torch()
    from minigpt import GPTConfig, MiniGPT

    args = parse_args()
    top_k = None if args.top_k is None else int(args.top_k)
    top_p = None if args.top_p is None else float(args.top_p)
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be > 0")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")

    ckpt = torch.load(Path(args.model), map_location="cpu")
    vocab = ckpt["vocab"]
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    cfg = GPTConfig(**ckpt["cfg"])

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    model = MiniGPT(cfg).to(device_t)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    prompt = ""
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    # If prompt has unseen chars, we fail early.
    for ch in prompt:
        if ch not in stoi:
            raise KeyError(f"Prompt has unseen character {ch!r}")

    torch.manual_seed(int(args.seed))

    if prompt:
        idx = torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long, device=device_t)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device_t)

    def sample_next(logits_1d):
        import torch.nn.functional as F

        temperature = float(args.temperature)
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        logits_1d = logits_1d / temperature
        probs = F.softmax(logits_1d, dim=-1)

        if top_k is not None:
            k = min(top_k, int(probs.numel()))
            v, ix = torch.topk(probs, k)
            mask = torch.zeros_like(probs)
            mask[ix] = v
            probs = mask / mask.sum()

        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_ix = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = int((cumsum >= top_p).nonzero(as_tuple=False)[0].item())
            keep_ix = sorted_ix[: cutoff + 1]
            mask = torch.zeros_like(probs)
            mask[keep_ix] = probs[keep_ix]
            probs = mask / mask.sum()

        return int(torch.multinomial(probs, num_samples=1).item())

    with torch.no_grad():
        for _ in range(int(args.length)):
            idx_cond = idx[:, -cfg.block_size :]
            logits, _ = model(idx_cond)
            next_id = sample_next(logits[0, -1])
            idx = torch.cat([idx, torch.tensor([[next_id]], device=device_t)], dim=1)

    out = "".join(itos[int(i)] for i in idx[0].tolist())
    print(out)


if __name__ == "__main__":
    main()
