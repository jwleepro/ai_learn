# 10. (선택) PyTorch로 MiniGPT 학습하기

목표: 지금까지의 개념(토큰화→학습→샘플링)을 Transformer(GPT)로 묶어서,
작은 데이터/작은 모델로 **실제로 학습되는 LLM**을 직접 만듭니다.

> 현재 작업 폴더의 Python(3.14) 환경에는 torch가 없고, 버전 호환 이슈가 있을 수 있습니다.  
> 권장: **Python 3.12/3.13 가상환경 + PyTorch**.

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

체크포인트:

- `llm_from_scratch/models/minigpt.pt`

---

## 3) 생성

```powershell
python llm_from_scratch/torch/generate_minigpt.py --model llm_from_scratch/models/minigpt.pt --length 400 --seed 0 --temperature 1.0 --top_p 0.9
```

프롬프트를 파일로 넣기:

```powershell
python llm_from_scratch/torch/generate_minigpt.py --model llm_from_scratch/models/minigpt.pt --prompt_file llm_from_scratch/data/tiny_corpus_ko.txt --length 200
```

---

## 4) 다음 확장 아이디어

- 더 큰 데이터로 학습(뉴스/소설/위키 등)
- BPE 토크나이저를 붙여서 시퀀스 길이 줄이기
- 평가(perplexity) + 간단 벤치마크 만들기

