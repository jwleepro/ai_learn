# Week 3 과제 (Core / numpy)

> 관련 레슨: [05_mlp_context_lm](lessons/05_mlp_context_lm.md)

목표: 단일 글자가 아닌 **여러 글자(context)**를 보고 다음 글자를 예측하는 MLP 모델을 구현하고 학습시킵니다.

---

## 과제 1) MLP 언어모델 학습하기

아래 명령을 실행하여 모델을 학습시키세요. 컨텍스트 길이($C$), 임베딩 차원($D$), 은닉층 크기($H$) 등이 결과에 미치는 영향을 확인합니다.

> 소스: [`week3_complete.py`](code/week3_complete.py)
```powershell
python code/week3_complete.py --train --data data/tiny_corpus_ko.txt
```

---

## 과제 2) 학습된 모델로 생성하기

학습된 모델(`mlp_model.npz`)을 사용하여 텍스트를 생성해보세요.

> 소스: [`week3_complete.py`](code/week3_complete.py)
```powershell
python code/week3_complete.py --generate
```

질문:
1. 빅램 모델(Week 1, 2)과 비교했을 때, 문장이 더 자연스러워졌나요?
2. 특정 단어나 조사가 반복되는 현상이 있나요?

---

## (선택) 과제 3) 하이퍼파라미터 변경

`week3_complete.py` 코드 내의 `C`(컨텍스트 길이)를 16으로 늘리거나, `D`(임베딩 차원)를 64로 늘려보고 다시 학습시켰을 때 loss가 어떻게 변하는지 관찰하세요.

---

## 자기 점검(자동)

```powershell
python -m unittest discover -s llm_from_scratch/tests -p "test_core_week3.py" -v
```
