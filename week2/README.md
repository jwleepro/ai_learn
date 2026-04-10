# Week 2 과제 (Core / numpy)

> 관련 레슨: [04_neural_bigram](lessons/04_neural_bigram.md)

목표: 단순히 빈도를 세는 것이 아니라, **학습(training)**을 통해 파라미터를 최적화하는 과정을 이해합니다.

---

## 과제 1) 신경망 빅램 학습해보기

아래 명령을 실행하여 모델을 학습시키고, 손실(loss)이 줄어드는지 확인하세요.

> 소스: [`week2_complete.py`](code/week2_complete.py)
```powershell
python code/week2_complete.py --train --data data/tiny_corpus_ko.txt
```

---

## 과제 2) 학습된 모델로 생성하기

학습이 끝난 후, 저장된 모델(`bigram_nn_model.npz`)을 로드하여 텍스트를 생성해보세요.

> 소스: [`week2_complete.py`](code/week2_complete.py)
```powershell
python code/week2_complete.py --generate
```

질문:
1. 카운트 빅램(`week1_complete.py`)과 비교했을 때, 결과가 “실제로” 달라졌나요? (이론적으로는 같아야 합니다!)
2. 학습률($lr$)을 0.1이나 10.0으로 바꿨을 때 어떤 현상이 벌어지나요?

---

## (선택) 과제 3) 초기화 효과 관찰

`init_W` 함수에서 가중치 초기화 scale을 `0.01`이 아니라 `1.0` 혹은 `10.0`처럼 아주 크게 잡으면 학습 초기에 어떤 일이 벌어지는지 확인해보세요.

---

## 자기 점검(자동)

```powershell
python -m unittest discover -s llm_from_scratch/tests -p "test_core_week2.py" -v
```
