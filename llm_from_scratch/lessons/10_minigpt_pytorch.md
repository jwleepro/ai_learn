# 10. (선택) PyTorch로 MiniGPT 학습하기

목표: 지금까지의 개념(토큰화→학습→샘플링)을 Transformer(GPT)로 묶어서,
작은 데이터/작은 모델로 **실제로 학습되는 LLM**을 직접 만듭니다.

> 현재 작업 폴더의 Python(3.14) 환경에는 torch가 없고, 버전 호환 이슈가 있을 수 있습니다.  
> 권장: **Python 3.12/3.13 가상환경 + PyTorch**.
>
> 이 파트부터는 **수학 난도보다 PyTorch 문법 난도**가 먼저 올라갑니다.
> Core 트랙(특히 06, 07)을 끝낸 뒤 보는 것을 권장합니다.

---

## 0) 읽기 전에

이 파트에서 새로 어려운 것은 공식을 더 외우는 일이 아니라,
**PyTorch가 텐서와 레이어를 어떻게 표현하는지 읽는 일**입니다.

첫 번째 읽기에서는 아래 3가지만 잡으면 충분합니다.

1. 입력 `idx`가 임베딩을 거쳐 `(B, T, C)` 텐서가 된다.
2. `Block`이 attention + MLP를 감싼 반복 단위다.
3. 마지막에 `(B, T, V)` logits를 만들고, `cross_entropy`로 학습한다.
   - `cross_entropy`는 "정답 글자에 높은 점수를 주면 좋아지는 채점 방식" 정도로 보면 됩니다.

코드 읽는 순서:

1. [train_minigpt.py](../torch/train_minigpt.py): 학습 루프 전체 흐름
2. [minigpt.py](../torch/minigpt.py): `MiniGPT` → `Block` → `CausalSelfAttention` → `MLP`
3. [generate_minigpt.py](../torch/generate_minigpt.py): 학습된 모델로 생성

첫 번째 읽기에서 잠시 넘어가도 되는 것:

- `register_buffer` = 학습은 안 하지만 모델과 같이 들고 다니는 값
- `.transpose(...)` = 텐서 축 순서 바꾸기
- `.contiguous().view(...)` = 텐서 모양 다시 정리하기
- `dropout` = 학습 중 일부 연결을 잠깐 끄는 규제 장치

---

## 1) 설치(예시)

```powershell
python -m venv .venv_torch
.\.venv_torch\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r llm_from_scratch/requirements-torch.txt
```

---

## 2) 학습

```powershell
python llm_from_scratch/torch/train_minigpt.py --input llm_from_scratch/data/tiny_corpus_ko.txt --steps 2000 --block 128 --embd 128 --heads 4 --layers 4
```

명령의 옵션을 평범한 말로 번역하면:

- `--steps`: 몇 번 업데이트할지
- `--block`: 한 번에 최대 몇 글자까지 보고 다음 글자를 맞힐지
- `--embd`: 토큰 하나를 몇 칸짜리 벡터로 표현할지
- `--heads`: attention을 몇 갈래 시선으로 나눌지
- `--layers`: Transformer 블록을 몇 번 반복할지

처음에는 아래처럼 더 작게 시작해도 됩니다:

```powershell
python llm_from_scratch/torch/train_minigpt.py --input llm_from_scratch/data/tiny_corpus_ko.txt --steps 300 --block 64 --embd 64 --heads 4 --layers 2
```

이 설정은 "성능"보다 **학습이 실제로 돌아가는지 확인**하는 데 더 적합합니다.

체크포인트:

- `llm_from_scratch/models/minigpt.pt`

학습 로그를 볼 때는:

- `train_loss`, `val_loss`가 내려가는지
- 너무 빨리 과적합되지 않는지
- `block`, `embd`, `heads`, `layers`를 키우면 시간이 얼마나 늘어나는지

부터 보면 됩니다.

처음에는 아래처럼 생각하면 충분합니다.

- `block` 증가 = 더 긴 문맥을 볼 수 있음, 대신 느려짐
- `embd` 증가 = 각 글자를 더 풍부하게 표현, 대신 무거워짐
- `heads` 증가 = 여러 관점으로 attention, 대신 계산 증가
- `layers` 증가 = 더 깊은 모델, 대신 학습 시간 증가

---

## 3) 생성

```powershell
python llm_from_scratch/torch/generate_minigpt.py --model llm_from_scratch/models/minigpt.pt --length 400 --seed 0 --temperature 1.0 --top_p 0.9
```

프롬프트를 파일로 넣기:

```powershell
python llm_from_scratch/torch/generate_minigpt.py --model llm_from_scratch/models/minigpt.pt --prompt_file llm_from_scratch/data/tiny_corpus_ko.txt --length 200
```

생성 옵션은 이렇게 보면 됩니다.

- `temperature`: 보수적으로 뽑을지, 더 랜덤하게 뽑을지
- `top_k`: 후보를 상위 몇 개로 자를지
- `top_p`: 확률 큰 후보들을 누적합 기준으로 어디까지 남길지

---

## 4) 다음 확장 아이디어

- 더 큰 데이터로 학습(뉴스/소설/위키 등)
- BPE 토크나이저를 붙여서 시퀀스 길이 줄이기
- 평가(perplexity) + 간단 벤치마크 만들기
- (선택) 파인튜닝 입문(SFT/LoRA/QLoRA 용어/운영): [11_finetuning_essentials.md](11_finetuning_essentials.md) → [12_lora_qlora_and_ops.md](12_lora_qlora_and_ops.md)

> 파인튜닝/운영 쪽으로 넘어갈 때는 [GLOSSARY.md](../GLOSSARY.md)의 확장 섹션을 같이 보면 용어가 덜 튑니다.

---

[← 이전: BPE 토크나이저](09_bpe_tokenizer.md) | [목차](INDEX.md) | [다음: 파인튜닝 필수 용어 →](11_finetuning_essentials.md)
