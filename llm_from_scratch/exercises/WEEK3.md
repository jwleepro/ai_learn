# Week 3 과제 (Core / numpy): MLP LM

> 관련 레슨: [05_mlp_context_lm — MLP 언어모델](../lessons/05_mlp_context_lm.md)

목표: 컨텍스트 길이 `k`를 늘려보며, “문맥을 본다”가 모델 품질에 어떤 영향을 주는지 경험합니다.

---

## 0) 학습 실행(기본)

```powershell
python llm_from_scratch/code/train_mlp_lm.py --input llm_from_scratch/data/tiny_corpus_ko.txt --context 8 --embed 32 --hidden 128 --epochs 60 --lr 0.2
```

저장:

- `llm_from_scratch/models/mlp_lm.npz`

생성:

```powershell
python llm_from_scratch/code/generate_mlp_lm.py --model llm_from_scratch/models/mlp_lm.npz --length 300 --seed 0 --temperature 1.0
```

---

## 1) 실험 1: 컨텍스트 길이 바꾸기

아래 3개를 학습/생성해서 비교해보세요.

- `--context 2`
- `--context 8`
- `--context 16`

질문:

- “더 문장 같아졌다”가 느껴지나요?
- 반복/깨짐/랜덤함은 어떻게 바뀌나요?

---

## 2) 실험 2: 모델 크기 바꾸기

고정: `--context 8`, 나머지를 바꿉니다.

- 작은 모델: `--embed 16 --hidden 64`
- 큰 모델: `--embed 64 --hidden 256`

질문:

- train loss / val loss가 어떻게 달라지나요?
- 큰 모델이 val에서 더 나빠지는(과적합) 신호가 보이나요?

---

## 3) 내 코퍼스로 학습(권장)

`llm_from_scratch/data/my_corpus.txt`로 바꿔서:

- 카운트 빅램
- 신경망 빅램
- MLP LM

3개를 모두 생성 비교해보세요.

