# Week 2 과제 (Core / numpy): 신경망 빅램 LM

> 관련 레슨: [04_neural_bigram — 신경망 빅램 LM](../lessons/04_neural_bigram.md)

목표: 카운트 대신 **학습되는 가중치(W)**로 `P(next|prev)`를 만들고, loss가 실제로 내려가는 경험을 합니다.

---

## 0) 학습 실행

```powershell
python llm_from_scratch/code/train_bigram_nn.py --input llm_from_scratch/data/tiny_corpus_ko.txt --epochs 40 --lr 2.0 --batch 2048
```

학습이 끝나면 기본 경로로 저장됩니다:

- `llm_from_scratch/models/bigram_nn.npz`

---

## 1) 생성 실행

```powershell
python llm_from_scratch/code/generate_bigram_nn.py --model llm_from_scratch/models/bigram_nn.npz --length 300 --seed 0 --temperature 1.0
```

---

## 2) 관찰 질문

1. 카운트 빅램(`generate_bigram.py`)과 비교했을 때, 결과가 “실제로” 달라졌나요?
2. 학습률(`--lr`)을 너무 크게/작게 하면 어떤 현상이 보이나요?
3. `--val_frac`를 0.0 / 0.1 / 0.3으로 바꾸면 val loss는 어떻게 변하나요?

---

## 3) 내 코퍼스로 확장(권장)

`llm_from_scratch/data/my_corpus.txt`를 만들고 (200~5000자),
같은 방식으로 학습/생성을 해보세요.

---

## 자기 점검(자동)

```powershell
python -m unittest discover -s llm_from_scratch/tests -p "test_core_week2.py" -v
```

